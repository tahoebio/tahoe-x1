from functools import lru_cache
from typing import Any, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
from llmfoundry.models.layers.attention import ATTN_CLASS_REGISTRY
from llmfoundry.models.layers.ffn import resolve_ffn_hidden_size, resolve_ffn_act_fn
from llmfoundry.models.layers.norm import NORM_CLASS_REGISTRY
from torch import Tensor
from torch.nn.modules.transformer import _get_clones

attn_config_defaults: Dict = {
    "attn_type": "grouped_query_attention",
    "attn_pdrop": 0.0,
    "attn_impl": "triton",
    "qk_ln": False,
    "qk_gn": False,
    "clip_qkv": None,
    "softmax_scale": None,
}

norm_config_defaults: Dict = {
    "norm_type": "low_precision_layernorm",
    "eps": 1e-5,
}


class SCGPTBlock(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        n_heads: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """

    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        expansion_ratio: int,
        attn_config: Optional[Dict] = None,
        norm_config: Optional[Dict] = None,
        dropout: Optional[float] = 0.0,
        activation: Optional[str] = "gelu",
        device: Optional[str] = None,
        dtype=None,
        norm_scheme="pre",
        use_glu: bool = False,
        **kwargs: Any,
    ) -> None:
        if attn_config is None:
            attn_config = attn_config_defaults
        if norm_config is None:
            norm_config = norm_config_defaults
        del kwargs  # unused, just to capture any extra args from the config
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        attn_class = ATTN_CLASS_REGISTRY[attn_config["attn_type"]]
        self.d_model = d_model
        self.n_heads = n_heads
        self.device = device
        self.self_attn = attn_class(
            d_model=d_model,
            n_heads=n_heads,
            kv_n_heads=attn_config.get("kv_n_heads", n_heads),
            attn_impl=attn_config.get("attn_impl", "triton"),
            device=device,
        )
        # Implementation of Feedforward model
        dim_feedforward = resolve_ffn_hidden_size(d_model, expansion_ratio)
        self.up_proj = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.down_proj = nn.Linear(dim_feedforward, d_model, **factory_kwargs)
        self.use_glu = use_glu
        if self.use_glu:
            self.gate_proj = nn.Linear(d_model, dim_feedforward, **factory_kwargs)

        # Norms
        norm_class = NORM_CLASS_REGISTRY[norm_config["norm_type"].lower()]
        self.norm1 = norm_class(
            d_model, device=device, eps=norm_config.get("eps", 1e-5)
        )
        self.norm2 = norm_class(
            d_model, device=device, eps=norm_config.get("eps", 1e-5)
        )
        self.post_sa_dropout = nn.Dropout(dropout)
        self.post_ffn_dropout = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be either pre or post")

    @staticmethod
    def _get_activation_fn(activation):
        return resolve_ffn_act_fn({"name": activation})

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        if self.norm_scheme == "pre":
            x = x + self._sa_block(self.norm1(x), attn_bias=attn_bias)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, attn_bias=attn_bias))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x: Tensor, attn_bias: Optional[Tensor] = None) -> Tensor:
        x, _, _ = self.self_attn(x, attn_bias=attn_bias, is_causal=False)
        return self.post_sa_dropout(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        if self.use_glu:
            x = self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))
        else:
            x = self.down_proj(self.activation(self.up_proj(x)))
        return self.post_ffn_dropout(x)


class SCGPTEncoder(nn.Module):
    # takes in the set of different inputs in an mapping
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: SCGPTBlock,
        num_layers: int,
        use_norm: bool = False,
        norm_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_norm = use_norm
        if self.use_norm:
            if norm_config is None:
                norm_config = norm_config_defaults
            norm_class = NORM_CLASS_REGISTRY[norm_config["norm_type"].lower()]
            self.norm = norm_class(
                encoder_layer.d_model,
                device=encoder_layer.device,
                eps=norm_config.get("eps", 1e-5),
            )

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Optional[Tensor] = None,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if gen_total_embs is None:
            pcpt_only = True
            total_embs = pcpt_total_embs
        else:
            pcpt_only = False
            total_embs = torch.cat([pcpt_total_embs, gen_total_embs], dim=1)

        if pcpt_key_padding_mask is None and gen_key_padding_mask is None:
            key_padding_mask = None
        else:
            if pcpt_key_padding_mask is None:
                pcpt_key_padding_mask = torch.ones(
                    (pcpt_total_embs.shape[0], pcpt_total_embs.shape[1]),
                    device=pcpt_total_embs.device,
                    dtype=torch.bool,
                )  # 1 means attention is allowed
            elif gen_key_padding_mask is None:
                gen_key_padding_mask = torch.ones(
                    (gen_total_embs.shape[0], gen_total_embs.shape[1]),
                    device=gen_total_embs.device,
                    dtype=torch.bool,
                )  # 1 means attention is allowed
            key_padding_mask = torch.cat(
                [pcpt_key_padding_mask, gen_key_padding_mask], dim=1
            )  # (B, S)
        p_len = pcpt_total_embs.shape[1]
        total_len = total_embs.shape[1]
        g_len = total_len - p_len
        attention_mask = self._make_mask(p_len, g_len, total_embs.device)
        attn_bias = torch.zeros_like(
            attention_mask,
            dtype=total_embs.dtype,
            device=attention_mask.device,
            requires_grad=False,
        ).masked_fill(
            ~attention_mask, torch.finfo(total_embs.dtype).min
        )  # Matrix with -inf at the place of masked values and 0 elsewhere
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(
            1
        )  # Broadcastable to (B,H, S_Q, S_K) dimensions

        if key_padding_mask is not None:  # NOTE: handle when key_padding_mask is None
            # Merge the key_padding_mask into attn_bias
            b_size, s_k = key_padding_mask.shape[:2]
            attn_bias = attn_bias.masked_fill(
                ~key_padding_mask.view((b_size, 1, 1, s_k)),
                torch.finfo(total_embs.dtype).min,
            )
        for mod in self.layers:
            total_embs = mod(total_embs, attn_bias=attn_bias)

        if self.use_norm:
            total_embs = self.norm(total_embs)
        if pcpt_only:
            return total_embs
        else:
            pcpt_total_embs = total_embs[:, :p_len, :]
            gen_total_embs = total_embs[:, p_len:, :]
            return pcpt_total_embs, gen_total_embs

    @torch.no_grad()
    @lru_cache(maxsize=1)
    def _make_mask(self, p_len, g_len, device):
        # Mask follows the LLM Foundry convention
        # ie: 0 indicates no-attention, 1 indicates attention is allowed
        total_len = p_len + g_len
        attention_mask = torch.ones(
            (total_len, total_len), device=device, dtype=torch.bool
        )  # (pcpt_len+gen_len, pcpt_len+gen_len)

        if g_len > 0:
            # pcpt genes should not see gen genes
            # Equivalent to dense self-attention on pcpt genes
            attention_mask[0:p_len, -g_len:] = False
            # gen genes can see all pcpt genes and themselves, not other gen genes.
            # make the last gen_len by gen_gen to be an identity matrix, attention allowed along the diagonal
            # Equivalent to cross-attention from pcpt genes to gen genes
            attention_mask[-g_len:, -g_len:] = torch.eye(
                g_len, device=device, dtype=torch.bool
            )
        return attention_mask


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        use_norm: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.use_norm = use_norm
        if self.use_norm:
            self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        if self.use_norm:
            x = self.enc_norm(
                x
            )  # Norm for embedding is not used when using pre-norm transformer.
        return x


class ContinuousValueEncoder(nn.Module):
    """
    Encode real number values to a vector using neural nets projection.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_value: int = 512,
        activation: str = "relu",
        use_norm: bool = False,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = resolve_ffn_act_fn({"name": activation})
        self.linear2 = nn.Linear(d_model, d_model)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        if self.use_norm:
            x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        use_norm: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.use_norm = use_norm
        if self.use_norm:
            self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
        if self.use_norm:
            x = self.enc_norm(x)
        return x


class ExprDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_outputs: int = 1,
        n_layers: int = 2,
        activation: str = "leaky_relu",
    ):
        super().__init__()
        d_in = d_model
        self.activation = resolve_ffn_act_fn({"name": activation})
        self.linear_layers = nn.ModuleList(
            [nn.Linear(d_in, d_model) for _ in range(n_layers)]
        )
        self.out_proj = nn.Linear(d_model, n_outputs)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """x is the output of the transformer, (batch, seq_len, d_model)"""
        for layer in self.linear_layers:
            x = self.activation(layer(x))
        pred_value = self.out_proj(x)  # (batch, seq_len, n_outputs)
        if pred_value.shape[-1] == 1:
            pred_value = pred_value.squeeze(-1)
        return dict(pred=pred_value)


class MVCDecoder(nn.Module):
    """
    Decoder for the masked value prediction for cell embeddings.
    """

    def __init__(
        self,
        d_model: int,
        arch_style: str = "inner product",
        query_activation: str = "sigmoid",
    ) -> None:
        """
        Args:
            d_model (:obj:`int`): dimension of the gene embedding.
            arch_style (:obj:`str`): architecture style of the decoder, choice from
                1. "inner product" or 2. "concat query" or 3. "sum query".
            query_activation (:obj:`nn.Module`): activation function for the query
                vectors.
            hidden_activation (:obj:`nn.Module`): activation function for the hidden
                layers.
        """
        super().__init__()
        d_in = d_model

        if arch_style == "inner product":
            self.gene2query = nn.Linear(d_model, d_model)
            self.query_activation = resolve_ffn_act_fn({"name": query_activation})
            self.W = nn.Linear(d_model, d_in, bias=False)
        else:
            raise ValueError(f"Unknown arch_style: {arch_style}")

        self.arch_style = arch_style

    def forward(
        self, cell_emb: Tensor, gene_embs: Tensor
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Args:
            cell_emb: Tensor, shape (batch, embsize=d_model)
            gene_embs: Tensor, shape (batch, seq_len, embsize=d_model)
        """
        if self.arch_style == "inner product":
            query_vecs = self.query_activation(
                self.gene2query(gene_embs)
            )  # (batch, seq_len, embsize)
            cell_emb = cell_emb.unsqueeze(2)  # (batch, embsize, 1)
            pred_value = torch.bmm(self.W(query_vecs), cell_emb).squeeze(
                2
            )  # (batch, seq_len)
            return dict(pred=pred_value)
        else:
            raise ValueError(f"Unknown arch_style: {self.arch_style}")
