import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Dict, Mapping, Optional, Tuple, Any, Union
from composer.models import ComposerModel
from scgpt.loss import masked_mse_loss, MaskedMseMetric
from .flash_layers import SCGPTBlock, FlashscGPTGenerator
from llmfoundry.models.layers.ffn import resolve_ffn_hidden_size, resolve_ffn_act_fn
from omegaconf import DictConfig


class SCGPTModel(nn.Module):
    def __init__(
        self,
        model_config: DictConfig,
        collator_config: DictConfig,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model_type = "Transformer"
        self.device = device
        self.vocab_size = model_config.vocab_size
        self.n_layers = model_config.n_layers
        self.n_heads = model_config.n_heads
        self.d_model = model_config.d_model
        self.expansion_ratio = model_config.expansion_ratio
        self.norm_scheme = model_config.get("norm_scheme", "pre")
        self.transformer_activation = model_config.get("transformer_activation", "gelu")
        self.use_generative_training = model_config.get("use_generative_training", True)

        self.cell_emb_style = model_config.get("cell_emb_style", "cls")

        self.pad_token_id = collator_config.pad_token_id
        self.pad_value = collator_config.pad_value
        self.n_input_bins = collator_config.num_bins

        self.attn_config = model_config.get("attn_config", None)
        self.norm_config = model_config.get("norm_config", None)
        if self.cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")

        # TODO: add dropout in the GeneEncoder
        self.gene_encoder = GeneEncoder(
            self.vocab_size, self.d_model, padding_idx=self.pad_token_id
        )
        self.flag_encoder = nn.Embedding(2, self.d_model)

        expression_encoder_config = model_config.expression_encoder
        self.input_emb_style = expression_encoder_config.get(
            "input_emb_style", "continuous"
        )
        if self.input_emb_style not in ["category", "continuous"]:
            raise ValueError(
                f"input_emb_style should be one of category or continuous"
                f"got {self.input_emb_style}"
            )
        if self.input_emb_style == "continuous":
            self.expression_encoder = ContinuousValueEncoder(
                d_model=self.d_model,
                dropout=expression_encoder_config.get("dropout", 0.1),
                max_value=expression_encoder_config.get("max_value", 512),
                activation=expression_encoder_config.get("activation", "relu"),
            )
        elif self.input_emb_style == "category":
            assert self.n_input_bins > 0
            self.expression_encoder = CategoryValueEncoder(
                self.n_input_bins, self.d_model, padding_idx=self.pad_value
            )
        else:
            raise ValueError(f"Unknown input_emb_style: {self.input_emb_style}")

        if self.use_generative_training:
            encoder_layers = SCGPTBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                expansion_ratio=self.expansion_ratio,
                attn_config=self.attn_config,
                norm_config=self.norm_config,
                activation=self.transformer_activation,
                device=self.device,
                norm_scheme=self.norm_scheme,
            )
            self.transformer_encoder = FlashscGPTGenerator(
                encoder_layers, self.n_layers
            )
        else:
            raise NotImplementedError("Only generative training is supported")

        expression_decoder_config = model_config.expression_decoder
        self.expression_decoder = ExprDecoder(
            d_model=self.d_model,
            n_outputs=expression_decoder_config.get("n_outputs", 1),
            n_layers=expression_decoder_config.get("n_layers", 2),
            activation=expression_decoder_config.get("activation", "leaky_relu"),
        )

        if model_config.mvc is not None:
            mvc_config = model_config.mvc
            self.mvc_decoder = MVCDecoder(
                d_model=self.d_model,
                arch_style=mvc_config.arch_style,
                query_activation=mvc_config.query_activation,
            )

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # TODO: check if this initialization is helpful and shall we apply to all?
        self.gene_encoder.embedding.weight.data.uniform_(-initrange, initrange)

    def _encode(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
    ) -> Tensor:
        src = self.gene_encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src
        values = self.expression_encoder(values)  # (batch, seq_len, embsize)
        total_embs = src + values
        output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )
        return output  # (batch, seq_len, embsize)

    def transformer_generate(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        input_cell_emb: Optional[Tensor] = None,  # (batch, seq_len, embsize)
    ) -> Tuple[Tensor, Tensor]:
        pcpt_token_embs = self.gene_encoder(pcpt_genes)  # (batch, pcpt_len, embsize)
        pcpt_values = self.expression_encoder(pcpt_values)  # (batch, pcpt_len, embsize)
        pcpt_total_embs = pcpt_token_embs + pcpt_values
        if gen_genes is not None:
            gen_token_embs = self.gene_encoder(gen_genes)  # (batch, gen_len, embsize)
            self.cur_gene_token_embs = torch.cat(
                [pcpt_token_embs, gen_token_embs], dim=1
            )
            gen_flags = self.flag_encoder(
                torch.tensor(1, device=pcpt_values.device)
            ).expand(gen_genes.shape[0], gen_genes.shape[1], -1)

            gen_total_embs = gen_token_embs + gen_flags
        else:
            self.cur_gene_token_embs = pcpt_token_embs
            gen_total_embs = None

        if input_cell_emb is not None:
            pcpt_total_embs[:, 0, :] = input_cell_emb

        pcpt_output, gen_output = self.transformer_encoder(
            pcpt_total_embs,
            gen_total_embs,
            pcpt_key_padding_mask=pcpt_key_padding_mask,
            gen_key_padding_mask=gen_key_padding_mask,
        )

        return pcpt_output, gen_output

    def _get_cell_emb_from_layer(
        self, layer_output: Tensor, weights: Tensor = None
    ) -> Tensor:
        """
        Args:
            layer_output(:obj:`Tensor`): shape (batch, seq_len, embsize)
            weights(:obj:`Tensor`): shape (batch, seq_len), optional and only used
                when :attr:`self.cell_emb_style` is "w-pool".

        Returns:
            :obj:`Tensor`: shape (batch, embsize)
        """
        if self.cell_emb_style == "cls":
            cell_emb = layer_output[:, 0, :]  # (batch, embsize)
        elif self.cell_emb_style == "avg-pool":
            cell_emb = torch.mean(layer_output, dim=1)
        elif self.cell_emb_style == "w-pool":
            if weights is None:
                raise ValueError("weights is required when cell_emb_style is w-pool")
            if weights.dim() != 2:
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

    def generate(
        self,
        cell_emb: Tensor,
        src: Tensor,
        values: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            cell_emb(:obj:`Tensor`): shape (batch, embsize)
            src(:obj:`Tensor`): shape (batch, seq_len)
            values(:obj:`Tensor`): shape (batch, seq_len), optional
            src_key_padding_mask(:obj:`Tensor`): shape (batch, seq_len), optional
        """
        # TODO: should have a tag indicate the generation mode
        # TODO: if gen_iters > 1, should have a tag indicate the current iteration

        src = self.gene_encoder(src)  # (batch, seq_len, embsize)
        if values is not None:
            values = self.expression_encoder(values)  # (batch, seq_len, embsize)
            total_embs = src + values
        else:
            total_embs = src

        total_embs = self.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

        total_embs[:, 0, :] = cell_emb

        if src_key_padding_mask is None:
            src_key_padding_mask = torch.zeros(
                total_embs.shape[:2], dtype=torch.bool, device=total_embs.device
            )
        transformer_output = self.transformer_encoder(
            total_embs, src_key_padding_mask=src_key_padding_mask
        )

        mlm_output = self.expression_decoder(transformer_output)
        output = mlm_output["pred"]  # (batch, seq_len)

        return output  # (batch, seq_len)

    def _extend_output(
        self,
        output: Mapping[str, Tensor],
        transformer_output: Tensor,
        CLS: bool = False,
        MVC: bool = False,
    ) -> Mapping[str, Tensor]:
        cell_emb = self._get_cell_emb_from_layer(transformer_output)
        output["cell_emb"] = cell_emb

        if CLS:
            output["cls_output"] = self.cls_decoder(cell_emb)  # (batch, n_cls)
        if MVC:
            mvc_output = self.mvc_decoder(
                cell_emb,
                self.cur_gene_token_embs,
            )
            output["mvc_output"] = mvc_output["pred"]  # (batch, seq_len)
        return output

    def forward(
        self,
        *args,
        **kwargs,
    ) -> Mapping[str, Tensor]:
        """
        Wrapper to call either generative_forward or perceptual_forward, depending
        on the value of the "generative_training" kwarg.
        """
        if "generative_training" not in kwargs:
            # raise ValueError("generative_training kwarg is required")
            warnings.warn(
                "generative_training kwarg is required but not provided! "
                "Using False and calling perceptual_forward instead"
            )
            return self.perceptual_forward(*args, **kwargs)

        # get the generative training flag and pop it out
        do_generative_training = kwargs.pop("generative_training")
        if do_generative_training:
            return self.generative_forward(*args, **kwargs)
        else:
            return self.perceptual_forward(*args, **kwargs)

    def generative_forward(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        CLS: bool = False,
        MVC: bool = False,
        input_cell_emb: Optional[Tensor] = None,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            pcpt_genes (:obj:`Tensor`): token ids of the perceptual part, shape
                [batch_size, seq_len]
            pcpt_values (:obj:`Tensor`): token values of the perceptual part, shape
                [batch_size, seq_len]
            pcpt_key_padding_mask (:obj:`Tensor`): mask for pcpt_genes, shape
                [batch_size, seq_len]
            gen_genes (:obj:`Tensor`): token ids of the generative part, shape
                [batch_size, seq_len]
            gen_key_padding_mask (:obj:`Tensor`): mask for gen_genes, shape
                [batch_size, seq_len]
            input_cell_emb (:obj:`Tensor`): cell embeddings, shape [batch_size,
                embsize]

        Returns:
            :obj:`Mapping[str, Tensor]`:
                - pred (:obj:`Tensor`): prediction, shape [batch_size, seq_len]
                - cell_emb (:obj:`Tensor`): cell embeddings, shape [batch_size,
                    embsize]
        """

        pcpt_output, gen_output = self.transformer_generate(
            pcpt_genes,
            pcpt_values,
            pcpt_key_padding_mask,
            gen_genes,
            gen_key_padding_mask,
            input_cell_emb=input_cell_emb,
        )
        if gen_output is None:
            transformer_output = pcpt_output
        else:
            transformer_output = torch.cat([pcpt_output, gen_output], dim=1)

        output = {}
        decoder_output = self.expression_decoder(transformer_output)
        full_preds = decoder_output["pred"]  # (batch, seq_len)
        output["pcpt_preds"] = full_preds[:, : pcpt_genes.shape[1]]
        output["gen_preds"] = full_preds[:, pcpt_genes.shape[1] :]

        output = self._extend_output(
            output,
            transformer_output,
            CLS=CLS,
            MVC=MVC,
        )

        return output

    def perceptual_forward(
        self,
        src: Tensor,
        values: Tensor,
        src_key_padding_mask: Tensor,
        CLS: bool = False,
        MVC: bool = False,
    ) -> Mapping[str, Tensor]:
        """
        Args:
            src (:obj:`Tensor`): token ids, shape [batch_size, seq_len]
            values (:obj:`Tensor`): token values, shape [batch_size, seq_len]
            src_key_padding_mask (:obj:`Tensor`): mask for src, shape [batch_size,
                seq_len]
            CLS (:obj:`bool`): if True, return the celltype classification objective
                (CLS) output
            MVC (:obj:`bool`): if True, return the masked value prediction for cell
                embedding MVC output

        Returns:
            dict of output Tensors.
        """
        transformer_output = self._encode(src, values, src_key_padding_mask)

        output = {}
        mlm_output = self.expression_decoder(transformer_output)
        output["mlm_output"] = mlm_output["pred"]  # (batch, seq_len)

        output = self._extend_output(
            output,
            transformer_output,
            CLS=CLS,
            MVC=MVC,
        )

        return output

    def fsdp_wrap_fn(self, module):
        return isinstance(module, SCGPTBlock)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, SCGPTBlock)


class FlashTransformerEncoderLayer(nn.Module):
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
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",  # "pre" or "post"
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = FlashMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
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
        if self.norm_scheme not in ["pre", "post"]:
            raise ValueError(f"norm_scheme should be pre or post, not {norm_scheme}")

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
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        if src_mask is not None:
            raise ValueError("FlashTransformerEncoderLayer does not support src_mask")

        if not src_key_padding_mask.any().item():
            # no padding tokens in src
            src_key_padding_mask_ = None
        else:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            # NOTE: the FlashMHA uses mask 0 for padding tokens, which is the opposite
            src_key_padding_mask_ = ~src_key_padding_mask

        if self.norm_scheme == "pre":
            src = self.norm1(src)
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm2(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
        else:
            src2 = self.self_attn(src, key_padding_mask=src_key_padding_mask_)[0]
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)
            src = self.norm2(src)

        return src


class GeneEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)  # (batch, seq_len, embsize)
        x = self.enc_norm(x)  # TODO remove this for pre-norm
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
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(1, d_model)
        self.activation = resolve_ffn_act_fn({"name": activation})
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.max_value = max_value

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len]
        """
        # TODO: test using actual embedding layer if input is categorical
        # expand last dimension
        x = x.unsqueeze(-1)
        # clip x to [-inf, max_value]
        x = torch.clamp(x, max=self.max_value)
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x)
        return self.dropout(x)


class CategoryValueEncoder(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim, padding_idx=padding_idx
        )
        self.enc_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x.long()
        x = self.embedding(x)  # (batch, seq_len, embsize)
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


class ComposerSCGPTModel(ComposerModel):
    def __init__(self, model_config, collator_config, device=None):
        super().__init__()
        self.criterion = masked_mse_loss
        self.pad_token_id = collator_config.pad_token_id
        self.use_cell_conditioned_generation = model_config.get("use_cell_conditioned_generation", False)
        self.model = SCGPTModel(
            model_config=model_config,
            collator_config=collator_config,
            device=device,
        )
        self.n_active_params = sum(p.numel() for p in self.model.parameters())
        self.train_mse = MaskedMseMetric(name="MSE")
        self.train_mvc = MaskedMseMetric(name="MVC")
        self.val_mse = MaskedMseMetric(name="MSE")
        self.val_mvc = MaskedMseMetric(name="MVC")
        if self.use_cell_conditioned_generation:
            self.train_gen = MaskedMseMetric(name="GEN")
        if self.use_cell_conditioned_generation:
            self.val_gen = MaskedMseMetric(name="GEN")

    def forward(self, batch):  # batch is the output of the dataloader
        # specify how batches are passed through the model
        pcpt_gene = batch["pcpt_gene"]
        pcpt_expr = batch["pcpt_expr"]
        pcpt_key_padding_mask = pcpt_gene.eq(self.pad_token_id)
        gen_gene = batch["gen_gene"]
        gen_key_padding_mask = gen_gene.eq(self.pad_token_id)
        output_dict = self.model(
            pcpt_gene,
            pcpt_expr,
            pcpt_key_padding_mask,
            gen_gene,
            gen_key_padding_mask,
            MVC=True,
            generative_training=True,
        )
        if self.use_cell_conditioned_generation:
            previous_cell_embs = output_dict["cell_emb"].detach()
            preds = self.model(
                pcpt_gene,
                pcpt_expr,
                pcpt_key_padding_mask,
                gen_gene,
                gen_key_padding_mask,
                MVC=False,
                input_cell_emb=previous_cell_embs,
                generative_training=True,
            )["gen_preds"]
            output_dict["GEPC"] = preds
        return output_dict

    def loss(self, outputs, batch):
        # pass batches and `forward` outputs to the loss
        pcpt_gene = batch["pcpt_gene"]
        gen_gene = batch["gen_gene"]
        gen_expr_target = batch["gen_expr_target"]
        gen_key_padding_mask = gen_gene.eq(self.pad_token_id)
        positions_to_match = ~gen_key_padding_mask
        gen_expr_preds = outputs["gen_preds"]
        loss_mse = self.criterion(gen_expr_preds, gen_expr_target, positions_to_match)
        loss_mvc = self.criterion(
            outputs["mvc_output"][:, pcpt_gene.shape[1] :],
            gen_expr_target,
            positions_to_match,
        )
        if self.use_cell_conditioned_generation:
            loss_gen = self.criterion(outputs["GEPC"], gen_expr_target, positions_to_match)
            loss = (loss_mse + loss_mvc + loss_gen) / 3
        else:
            loss = (loss_mse + loss_mvc) / 2
        return loss

    def update_metric(self, batch, outputs, metric):
        pcpt_gene = batch["pcpt_gene"]
        gen_gene = batch["gen_gene"]
        mask = ~gen_gene.eq(self.pad_token_id)
        target = batch["gen_expr_target"]
        if metric.name == "MSE":
            preds = outputs["gen_preds"]
        elif metric.name == "MVC":
            preds = outputs["mvc_output"][:, pcpt_gene.shape[1] :]
        elif metric.name == "GEN":
            assert self.use_cell_conditioned_generation
            preds = outputs["GEPC"]
        else:
            raise ValueError(f"metric {metric.name} not recognized")
        metric.update(preds=preds, target=target, mask=mask)

    def get_metrics(self, is_train=False):
        # defines which metrics to use in each phase of training
        if is_train:
            if self.use_cell_conditioned_generation:
                metric_dict = {
                    "MSE": self.train_mse,
                    "MVC": self.train_mvc,
                    "GEN": self.train_gen,
                }
            else:
                metric_dict = {
                    "MSE": self.train_mse,
                    "MVC": self.train_mvc,
                }
        else:
            if self.use_cell_conditioned_generation:
                metric_dict = {
                    "MSE": self.val_mse,
                    "MVC": self.val_mvc,
                    "GEN": self.val_gen,
                }
            else:
                metric_dict = {
                    "MSE": self.val_mse,
                    "MVC": self.val_mvc,
                }
        return metric_dict

    def flops_per_batch(self, batch: Mapping) -> int:
        # specify how to compute the number of FLOPs for a batch
        # This assumes non cell-conditioned generation (single forward pass)
        bs = batch["pcpt_gene"].shape[0]
        pcpt_len = batch["pcpt_gene"].shape[1]
        gen_len = batch["gen_gene"].shape[1]
        msl = pcpt_len + gen_len  # Assumes no-padding (as an approximation)
        params = self.n_active_params
        params_flops_per_token = 2 * params
        params_flops_per_seq = params_flops_per_token * msl
        attn_flops_per_seq = (
            self.model.n_layers * 2 * 2 * (self.model.d_model * (msl**2))
        )
        return (params_flops_per_seq + attn_flops_per_seq) * 3 * bs
