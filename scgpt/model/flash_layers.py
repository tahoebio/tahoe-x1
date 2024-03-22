from functools import lru_cache
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones
from llmfoundry.models.layers.attention import GroupedQueryAttention



class SCGPTBlock(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    The class is modified from torch.nn.TransformerEncoderLayer to support the
    FlashAttention.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
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
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=nhead,
            kv_n_heads=nhead,
            attn_impl="triton",
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be either pre or post")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu

        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

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
        x = self.self_attn(x, attn_bias=attn_bias)
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class FlashscGPTGenerator(nn.Module):
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
    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        mask_check=True,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        total_embs = torch.cat([pcpt_total_embs, gen_total_embs], dim=1)
        if pcpt_key_padding_mask is None and gen_key_padding_mask is None:
            key_padding_mask = None
        else:
            if pcpt_key_padding_mask is None:
                pcpt_key_padding_mask = torch.ones(
                    (pcpt_total_embs.shape[0], pcpt_total_embs.shape[1]),
                    device=pcpt_total_embs.device,
                    dtype=torch.bool,
                )
            elif gen_key_padding_mask is None:
                gen_key_padding_mask = torch.ones(
                    (gen_total_embs.shape[0], gen_total_embs.shape[1]),
                    device=gen_total_embs.device,
                    dtype=torch.bool,
                )
            key_padding_mask = ~torch.cat(
                [pcpt_key_padding_mask, gen_key_padding_mask], dim=1
            )  # (B, S)
        p_len = pcpt_total_embs.shape[1]
        total_len = total_embs.shape[1]
        g_len = total_len - p_len
        attention_mask = _make_mask(p_len, g_len, total_embs.device)
        attn_bias = torch.zeros_like(
            attention_mask, dtype=total_embs.dtype, device=attention_mask.device, requires_grad=False
        ).masked_fill(
            attention_mask, torch.finfo(total_embs.dtype).min
        )  # Matrix with -inf at the place of masked values and 0 elsewhere
        # FIXME: verify what should be 1 in the triton key_padding_mask
        attn_bias = attn_bias.unsqueeze(0).unsqueeze(
            1
        )  # Broadcastable to (B,H, S_Q, S_K) dimensions

        # Merge the key_padding_mask into attn_bias
        b_size, s_k = key_padding_mask.shape[:2]
        attn_bias = attn_bias.masked_fill(
            ~key_padding_mask.view((b_size, 1, 1, s_k)),
            torch.finfo(total_embs.dtype).min,
        )

        for mod in self.layers:
            total_embs = mod(total_embs, attn_bias=attn_bias)

        if self.norm is not None:
            total_embs = self.norm(total_embs)

        pcpt_total_embs = total_embs[:, :p_len, :]
        gen_total_embs = total_embs[:, p_len:, :]
        return pcpt_total_embs, gen_total_embs

@torch.no_grad()
@lru_cache(maxsize=1)
def _make_mask(p_len, g_len, device):
    total_len = p_len + g_len
    attention_mask = torch.zeros(
        (g_len, total_len), device=device, dtype=torch.bool
    )  # (gen_len, pcpt_len+gen_len)
    # make the last gen_len by gen_gen to be true, only the diagonal is allowed with false
    attention_mask[:, -g_len:] = ~torch.eye(g_len, device=device, dtype=torch.bool)
    upper_mask = torch.zeros(
        (p_len, total_len), device=device, dtype=torch.bool
    )  # (pcpt_len, pcpt_len+gen_len)
    upper_mask[:, -g_len:] = True

    attention_mask = torch.cat([upper_mask, attention_mask], dim=0)
    return attention_mask
