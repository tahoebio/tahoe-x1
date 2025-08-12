# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import logging
from typing import Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from composer.utils import dist
from llmfoundry.layers_registry import param_init_fns
from omegaconf import DictConfig
from torch import Tensor, nn

from mosaicfm.loss import MaskedMseMetric, MaskedSpearmanMetric, masked_mse_loss
from mosaicfm.model.blocks import (
    AffineExprDecoder,
    CategoryValueEncoder,
    ChemEncoder,
    ContinuousValueEncoder,
    ExprDecoder,
    GeneEncoder,
    MVCDecoder,
    SCGPTBlock,
    SCGPTEncoder,
    gene_encoder_defaults,
    init_config_defaults,
)

log = logging.getLogger(__name__)


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
        self.use_chem_token = collator_config.get("use_chem_token", False)
        assert (
            not self.use_chem_token or "chemical_encoder" in model_config
        ), "If use_chem_token is set to True, chemical_encoder submodule needs to be specified!"
        assert (
            "chemical_encoder" not in model_config or self.use_chem_token
        ), "If chemical_encoder submodule is specified, use_chem_token needs to be set to True!"

        self.init_device = model_config.get("init_device", "cpu")
        if self.init_device == "mixed":
            if dist.get_local_rank() == 0:
                self.init_device = "cpu"
            else:
                self.init_device = "meta"
        self.cell_emb_style = model_config.get("cell_emb_style", "cls")
        self.pad_token_id = collator_config.pad_token_id
        self.pad_value = collator_config.pad_value
        self.n_input_bins = collator_config.num_bins
        self.attn_config = model_config.get("attn_config", None)
        self.norm_config = model_config.get("norm_config", None)
        self.init_config = model_config.get("init_config", None)
        self.gene_encoder_config = model_config.get("gene_encoder", None)
        if self.init_config is None:
            self.init_config = init_config_defaults
        if self.gene_encoder_config is None:
            self.gene_encoder_config = gene_encoder_defaults
        if self.cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")

        self.gene_encoder = GeneEncoder(
            self.vocab_size,
            self.d_model,
            padding_idx=self.pad_token_id,
            use_norm=self.gene_encoder_config["use_norm"],
            gene_encoder_cfg=self.gene_encoder_config,
        )
        self.flag_encoder = nn.Embedding(2, self.d_model)

        expression_encoder_config = model_config.expression_encoder
        self.input_emb_style = expression_encoder_config.get(
            "input_emb_style",
            "continuous",
        )
        if self.input_emb_style not in ["category", "continuous"]:
            raise ValueError(
                f"input_emb_style should be one of category or continuous"
                f"got {self.input_emb_style}",
            )
        if self.input_emb_style == "continuous":
            self.expression_encoder = ContinuousValueEncoder(
                d_model=self.d_model,
                dropout=expression_encoder_config.get("dropout", 0.1),
                max_value=expression_encoder_config.get("max_value", 512),
                activation=expression_encoder_config.get("activation", "relu"),
                use_norm=expression_encoder_config.get("use_norm", False),
            )
        elif self.input_emb_style == "category":
            assert self.n_input_bins > 0
            self.expression_encoder = CategoryValueEncoder(
                self.n_input_bins,
                self.d_model,
                padding_idx=self.pad_value,
                use_norm=False,
            )
        else:
            raise ValueError(f"Unknown input_emb_style: {self.input_emb_style}")

        if self.use_chem_token:
            chem_encoder_config = model_config.chemical_encoder
            self.chem_encoder = ChemEncoder(
                drug_fps_path=chem_encoder_config.get("drug_fps_path"),
                d_out=self.d_model,
                padding_idx=chem_encoder_config.get("padding_idx", 0),
                activation=chem_encoder_config.get("activation", "leaky_relu"),
                freeze=chem_encoder_config.get("freeze", False),
            )

        encoder_layers = SCGPTBlock(
            d_model=self.d_model,
            n_heads=self.n_heads,
            expansion_ratio=self.expansion_ratio,
            attn_config=self.attn_config,
            norm_config=self.norm_config,
            activation=self.transformer_activation,
            device=self.device,
            norm_scheme=self.norm_scheme,
            use_glu=model_config.get("use_glu", False),
        )
        self.transformer_encoder = SCGPTEncoder(
            encoder_layers,
            self.n_layers,
            use_norm=self.norm_scheme == "pre",
            norm_config=self.norm_config,
            attn_config=self.attn_config,
        )

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
                scaled_dot_product=mvc_config.get("scaled_dot_product", False),
            )

        if self.init_device != "meta":
            log.info(
                'MosaicML recommends using config.init_device="meta" with Composer + FSDP for faster initialization.',
            )
            self.apply(self.param_init_fn)

    def param_init_fn(self, module: nn.Module):
        # skip initialization for modules that has skip_init=True
        if hasattr(module, "skip_init") and module.skip_init:
            log.info(f"Skipping re-initializing for {module._get_name()}")
            return
        init_fn_name = self.init_config["name"]
        param_init_fns.get(init_fn_name)(
            module=module,
            n_layers=self.n_layers,
            d_model=self.d_model,
            **self.init_config,
        )

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
            total_embs=total_embs,
            gen_mask=None,
            key_padding_mask=src_key_padding_mask,
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
        drug_ids: Optional[
            Tensor
        ] = None,  # drug_ids is None if use_chem_token is set to False
    ) -> Tuple[Tensor, Tensor]:

        pcpt_token_embs = self.gene_encoder(pcpt_genes)  # (batch, pcpt_len, embsize)
        pcpt_values = self.expression_encoder(pcpt_values)  # (batch, pcpt_len, embsize)
        pcpt_total_embs = pcpt_token_embs + pcpt_values  # (batch, pcpt_len, embsize)

        if self.use_chem_token:
            # calculate chemical embedding and put it in its correct place (after <cls>)
            drug_embs = self.chem_encoder(drug_ids)  # (batch, embsize)
            pcpt_total_embs[:, 1, :] = drug_embs  # (batch, pcpt_len, embsize)

        if gen_genes is not None:
            gen_token_embs = self.gene_encoder(gen_genes)  # (batch, gen_len, embsize)
            self.cur_gene_token_embs = torch.cat(
                [pcpt_token_embs, gen_token_embs],
                dim=1,
            )
            gen_flags = self.flag_encoder(
                torch.tensor(1, device=pcpt_values.device),
            ).expand(gen_genes.shape[0], gen_genes.shape[1], -1)

            gen_total_embs = gen_token_embs + gen_flags
        else:
            self.cur_gene_token_embs = pcpt_token_embs
            gen_total_embs = None

        if input_cell_emb is not None:
            pcpt_total_embs[:, 0, :] = input_cell_emb

        if gen_total_embs is not None:
            total_embs = torch.cat([pcpt_total_embs, gen_total_embs], dim=1)
            p_len = pcpt_total_embs.shape[1]
            gen_mask = torch.zeros(
                total_embs.shape[0],
                total_embs.shape[1],
                dtype=torch.bool,
                device=total_embs.device,
            )
            gen_mask[:, p_len:] = True

            if pcpt_key_padding_mask is None and gen_key_padding_mask is None:
                key_padding_mask = None
            else:
                if pcpt_key_padding_mask is None:
                    pcpt_key_padding_mask = torch.ones(
                        (pcpt_total_embs.shape[0], pcpt_total_embs.shape[1]),
                        device=pcpt_total_embs.device,
                        dtype=torch.bool,
                    )
                if gen_key_padding_mask is None:
                    gen_key_padding_mask = torch.ones(
                        (gen_total_embs.shape[0], gen_total_embs.shape[1]),
                        device=gen_total_embs.device,
                        dtype=torch.bool,
                    )
                key_padding_mask = torch.cat(
                    [pcpt_key_padding_mask, gen_key_padding_mask],
                    dim=1,
                )
        else:
            total_embs = pcpt_total_embs
            gen_mask = None
            key_padding_mask = pcpt_key_padding_mask
            p_len = pcpt_total_embs.shape[1]

        total_output = self.transformer_encoder(
            total_embs=total_embs,
            gen_mask=gen_mask,
            key_padding_mask=key_padding_mask,
        )

        pcpt_output = total_output[:, :p_len, :]
        gen_output = total_output[:, p_len:, :] if gen_total_embs is not None else None

        return pcpt_output, gen_output

    def _get_cell_emb_from_layer(
        self,
        layer_output: Tensor,
        weights: Tensor = None,
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
            if weights.dim() != 2:  # noqa: PLR2004
                raise ValueError("weights should be 2D")
            cell_emb = torch.sum(layer_output * weights.unsqueeze(2), dim=1)
            cell_emb = F.normalize(cell_emb, p=2, dim=1)  # (batch, embsize)

        return cell_emb

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
        """Wrapper to call either generative_forward or perceptual_forward,
        depending on the value of the "generative_training" kwarg."""
        if "generative_training" not in kwargs:
            raise ValueError(
                "Please specify generative_training argument and set to False if doing inference",
            )
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
        drug_ids: Optional[Tensor] = None,
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
            drug_ids (:obj:`Tensor`): drug ids corresponding to chem_encoder embedding layer, shape
                [batch_size]

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
            drug_ids=drug_ids,
        )
        if gen_output is None:  # type: ignore
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
        expression_decoder_output = self.expression_decoder(transformer_output)
        output["pcpt_preds"] = expression_decoder_output["pred"]  # (batch, seq_len)

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


class ComposerSCGPTModel(ComposerModel):
    def __init__(self, model_config, collator_config, device=None):
        super().__init__()
        self.criterion = masked_mse_loss
        self.pad_token_id = collator_config.pad_token_id
        self.use_cell_conditioned_generation = model_config.get(
            "use_cell_conditioned_generation",
            False,
        )
        self.model = SCGPTModel(
            model_config=model_config,
            collator_config=collator_config,
            device=device,
        )
        self.n_active_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.train_metrics = {
            "MSE": MaskedMseMetric(name="MSE"),
            "MVC": MaskedMseMetric(name="MVC"),
        }
        self.standard_scale_outputs = model_config.get("standard_scale_outputs", False)
        self.collator_config = collator_config
        self.model_config = model_config

        self.val_metrics = {
            "MSE": MaskedMseMetric(name="MSE"),
            "MVC": MaskedMseMetric(name="MVC"),
            "Spearman": MaskedSpearmanMetric(name="Spearman"),
        }
        if self.use_cell_conditioned_generation:
            self.train_gen = MaskedMseMetric(name="GEN")
            self.train_metrics.update({"GEN": self.train_gen})
        if self.use_cell_conditioned_generation:
            self.val_gen = MaskedMseMetric(name="GEN")
            self.val_metrics.update({"GEN": self.val_gen})

    def forward(self, batch):  # batch is the output of the dataloader
        # specify how batches are passed through the model
        pcpt_gene = batch["pcpt_gene"]
        pcpt_expr = batch["pcpt_expr"]
        pcpt_key_padding_mask = ~pcpt_gene.eq(self.pad_token_id)
        drug_ids = (
            batch["drug_ids"] if "drug_ids" in batch else None
        )  # drug_ids is None if use_chem_token is set to False
        gen_gene = batch["gen_gene"]
        gen_key_padding_mask = ~gen_gene.eq(self.pad_token_id)
        output_dict = self.model(
            pcpt_gene,
            pcpt_expr,
            pcpt_key_padding_mask,
            gen_gene,
            gen_key_padding_mask,
            drug_ids=drug_ids,
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
                drug_ids=drug_ids,
                MVC=False,
                input_cell_emb=previous_cell_embs,
                generative_training=True,
            )["gen_preds"]
            output_dict["cell_conditioned_gen_preds"] = preds
        return output_dict

    def eval_forward(self, batch, outputs: Optional = None):
        if outputs:
            return outputs

        self.model.zero_grad(set_to_none=True)

        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs, batch):
        # pass batches and `forward` outputs to the loss
        pcpt_gene = batch["pcpt_gene"]
        gen_gene = batch["gen_gene"]
        gen_expr_target = batch["gen_expr_target"]
        if self.standard_scale_outputs:
            gen_expr_target = self.scale_outputs(gen_expr_target)
        gen_key_padding_mask = ~gen_gene.eq(self.pad_token_id)
        positions_to_match = gen_key_padding_mask
        gen_expr_preds = outputs["gen_preds"]
        loss_mse = self.criterion(gen_expr_preds, gen_expr_target, positions_to_match)
        loss_mvc = self.criterion(
            outputs["mvc_output"][:, pcpt_gene.shape[1] :],
            gen_expr_target,
            positions_to_match,
        )
        if self.use_cell_conditioned_generation:
            loss_gen = self.criterion(
                outputs["cell_conditioned_gen_preds"],
                gen_expr_target,
                positions_to_match,
            )
            loss = (loss_mse + loss_mvc + loss_gen) / 3
        else:
            loss = (loss_mse + loss_mvc) / 2
        return loss

    def update_metric(self, batch, outputs, metric):
        pcpt_gene = batch["pcpt_gene"]
        gen_gene = batch["gen_gene"]
        gen_expr_raw = batch["gen_expr_raw"]
        mask = ~gen_gene.eq(self.pad_token_id)
        target = batch["gen_expr_target"]
        if self.standard_scale_outputs:
            target = self.scale_outputs(target)
        if metric.name == "MSE":
            preds = outputs["gen_preds"]
        elif metric.name == "MVC":
            preds = outputs["mvc_output"][:, pcpt_gene.shape[1] :]
        elif metric.name == "GEN":
            assert self.use_cell_conditioned_generation
            preds = outputs["cell_conditioned_gen_preds"]
        elif metric.name == "Spearman":
            preds = outputs["gen_preds"]
            target = gen_expr_raw
        else:
            raise ValueError(f"metric {metric.name} not recognized")
        metric.update(preds=preds, target=target, mask=mask)

    def get_metrics(self, is_train=False):
        # defines which metrics to use in each phase of training
        metric_dict = self.train_metrics if is_train else self.val_metrics
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

    def scale_outputs(self, x: torch.Tensor) -> torch.Tensor:
        min_value = 1
        max_value = self.collator_config.num_bins - 1
        normalized_value = (x - min_value) / (max_value - min_value)
        # Scale to -1..1
        return 2 * normalized_value - 1


class ComposerSCGPTPerturbationModel(ComposerModel):
    def __init__(self, model_config, collator_config, device="cuda"):
        super().__init__()
        self.device = device
        self.criterion = masked_mse_loss
        self.pad_token_id = collator_config.pad_token_id
        self.use_cell_conditioned_generation = model_config.get(
            "use_cell_conditioned_generation",
            False,
        )
        self.model = SCGPTModel(
            model_config=model_config,
            collator_config=collator_config,
            device=device,
        )
        self.pert_encoder = nn.Embedding(3, self.model.d_model, padding_idx=0)
        self.pert_decoder = AffineExprDecoder(self.model.d_model)

    def forward(self, batch):
        gene_ids = batch["genes"]
        ctrl_expr = batch["expressions_ctrl"]
        perturbation_flags = batch["perturb_flags"]

        gene_token_emb = self.model.gene_encoder(gene_ids.to(self.device))
        gene_expr_emb = self.model.expression_encoder(ctrl_expr.to(self.device))
        pert_flag_emb = self.pert_encoder(perturbation_flags.to(self.device))
        combined_input_embs = gene_token_emb + gene_expr_emb + pert_flag_emb

        transformer_encoding = self.model.transformer_encoder(
            total_embs=combined_input_embs,
            gen_mask=None,
            key_padding_mask=None,
        )
        predicted_post_expr = self.pert_decoder(transformer_encoding, ctrl_expr)
        output = {
            "predicted_expr_perturbed": predicted_post_expr["pred"],
        }
        return output

    def loss(self, outputs, batch):
        expr_target = batch["expressions_perturbed"]
        gene_ids = batch["genes"]
        mask = torch.ones_like(gene_ids, dtype=torch.bool)

        expr_pred = outputs["predicted_expr_perturbed"]

        loss_mse = self.criterion(
            expr_pred,
            expr_target,
            mask,
        )
        return loss_mse
