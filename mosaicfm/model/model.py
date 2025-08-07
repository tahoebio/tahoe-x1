# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import logging
from typing import Any, Dict, Mapping, Optional, Tuple, Union

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
    """Single-Cell Gene Programming Transformer (SCGPT) Model.

    A transformer-based foundation model for single-cell RNA-seq data analysis.
    This model processes gene expression data by tokenizing genes and their expression
    values, then using transformer architecture to learn representations and perform
    various downstream tasks including generative modeling, masked value prediction,
    and cell type classification.

    The model supports both perceptual (discriminative) and generative training modes:
    - Perceptual mode: Processes full gene expression profiles for embedding extraction
      and discriminative tasks
    - Generative mode: Uses a perception-generation framework where a subset of genes
      are used as context (perception) to predict the remaining genes (generation)

    Key architectural components:
    - Gene encoder: Embeds gene identities using learned embeddings
    - Expression encoder: Encodes gene expression values (continuous or categorical)
    - Transformer encoder: Multi-layer transformer blocks for learning gene interactions
    - Expression decoder: Predicts gene expression values
    - Optional decoders: MVC (Masked Value Prediction), CLS (Classification)
    - Optional chemical encoder: For drug perturbation modeling

    Args:
        model_config: Configuration containing model architecture parameters including:
            - vocab_size: Number of genes in vocabulary (~60K for human genome)
            - n_layers: Number of transformer layers
            - n_heads: Number of attention heads
            - d_model: Hidden dimension size
            - expansion_ratio: Feed-forward network expansion ratio
            - norm_scheme: Normalization scheme ('pre' or 'post')
            - transformer_activation: Activation function for transformer blocks
            - use_generative_training: Whether to enable generative training mode
            - cell_emb_style: Cell embedding aggregation method ('cls', 'avg-pool', 'w-pool')
            - attn_config: Attention mechanism configuration
            - norm_config: Layer normalization configuration
            - init_config: Parameter initialization configuration
            - gene_encoder: Gene encoder configuration
            - expression_encoder: Expression encoder configuration
            - expression_decoder: Expression decoder configuration
            - mvc: Masked value prediction decoder configuration (optional)
            - chemical_encoder: Chemical encoder configuration (optional)
        collator_config: Data collation configuration containing:
            - pad_token_id: Padding token ID for genes
            - pad_value: Padding value for expressions
            - num_bins: Number of bins for categorical expression encoding
            - use_chem_token: Whether to use chemical tokens for drug perturbation
        device: Device to place model parameters on ('cpu', 'cuda', etc.)

    Attributes:
        model_type: Model type identifier ('Transformer')
        vocab_size: Size of gene vocabulary
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        d_model: Hidden dimension size
        expansion_ratio: Feed-forward expansion ratio
        norm_scheme: Normalization scheme
        transformer_activation: Transformer activation function
        use_generative_training: Whether generative training is enabled
        use_chem_token: Whether chemical tokens are used
        init_device: Device for parameter initialization
        cell_emb_style: Cell embedding aggregation method
        pad_token_id: Gene padding token ID
        pad_value: Expression padding value
        n_input_bins: Number of expression bins (for categorical encoding)
        input_emb_style: Expression encoding style ('continuous' or 'category')

    Examples:
        >>> model_cfg = DictConfig({
        ...     'vocab_size': 60000, 'n_layers': 12, 'n_heads': 8, 'd_model': 512,
        ...     'expansion_ratio': 4, 'expression_encoder': {'input_emb_style': 'continuous'},
        ...     'expression_decoder': {'n_outputs': 1}
        ... })
        >>> collator_cfg = DictConfig({'pad_token_id': 0, 'pad_value': -1, 'num_bins': 256})
        >>> model = SCGPTModel(model_cfg, collator_cfg)
        >>> # Forward pass for perceptual mode
        >>> genes = torch.randint(0, 60000, (32, 2000))  # (batch, seq_len)
        >>> expressions = torch.randn(32, 2000)  # (batch, seq_len)
        >>> mask = torch.ones(32, 2000, dtype=torch.bool)  # (batch, seq_len)
        >>> output = model(genes, expressions, mask, generative_training=False)
    """

    def __init__(
        self,
        model_config: DictConfig,
        collator_config: DictConfig,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_type: str = "Transformer"
        self.device: Optional[str] = device
        self.vocab_size: int = model_config.vocab_size
        self.n_layers: int = model_config.n_layers
        self.n_heads: int = model_config.n_heads
        self.d_model: int = model_config.d_model
        self.expansion_ratio: int = model_config.expansion_ratio
        self.norm_scheme: str = model_config.get("norm_scheme", "pre")
        self.transformer_activation: str = model_config.get(
            "transformer_activation",
            "gelu",
        )
        self.use_generative_training: bool = model_config.get(
            "use_generative_training",
            True,
        )
        self.use_chem_token: bool = collator_config.get("use_chem_token", False)
        assert (
            not self.use_chem_token or "chemical_encoder" in model_config
        ), "If use_chem_token is set to True, chemical_encoder submodule needs to be specified!"
        assert (
            "chemical_encoder" not in model_config or self.use_chem_token
        ), "If chemical_encoder submodule is specified, use_chem_token needs to be set to True!"

        self.init_device: str = model_config.get("init_device", "cpu")
        if self.init_device == "mixed":
            if dist.get_local_rank() == 0:
                self.init_device = "cpu"
            else:
                self.init_device = "meta"
        self.cell_emb_style: str = model_config.get("cell_emb_style", "cls")
        self.pad_token_id: int = collator_config.pad_token_id
        self.pad_value: Union[int, float] = collator_config.pad_value
        self.n_input_bins: int = collator_config.num_bins
        self.attn_config: Optional[Dict[str, Any]] = model_config.get(
            "attn_config",
            None,
        )
        self.norm_config: Optional[Dict[str, Any]] = model_config.get(
            "norm_config",
            None,
        )
        self.init_config: Optional[Dict[str, Any]] = model_config.get(
            "init_config",
            None,
        )
        self.gene_encoder_config: Optional[Dict[str, Any]] = model_config.get(
            "gene_encoder",
            None,
        )
        if self.init_config is None:
            self.init_config = init_config_defaults
        if self.gene_encoder_config is None:
            self.gene_encoder_config = gene_encoder_defaults
        if self.cell_emb_style not in ["cls", "avg-pool", "w-pool"]:
            raise ValueError(f"Unknown cell_emb_style: {self.cell_emb_style}")

        self.gene_encoder: GeneEncoder = GeneEncoder(
            self.vocab_size,
            self.d_model,
            padding_idx=self.pad_token_id,
            use_norm=self.gene_encoder_config["use_norm"],
            gene_encoder_cfg=self.gene_encoder_config,
        )
        self.flag_encoder: nn.Embedding = nn.Embedding(2, self.d_model)

        expression_encoder_config: DictConfig = model_config.expression_encoder
        self.input_emb_style: str = expression_encoder_config.get(
            "input_emb_style",
            "continuous",
        )
        if self.input_emb_style not in ["category", "continuous"]:
            raise ValueError(
                f"input_emb_style should be one of category or continuous"
                f"got {self.input_emb_style}",
            )
        if self.input_emb_style == "continuous":
            self.expression_encoder: ContinuousValueEncoder = ContinuousValueEncoder(
                d_model=self.d_model,
                dropout=expression_encoder_config.get("dropout", 0.1),
                max_value=expression_encoder_config.get("max_value", 512),
                activation=expression_encoder_config.get("activation", "relu"),
                use_norm=expression_encoder_config.get("use_norm", False),
            )
        elif self.input_emb_style == "category":
            assert self.n_input_bins > 0
            self.expression_encoder: CategoryValueEncoder = CategoryValueEncoder(
                self.n_input_bins,
                self.d_model,
                padding_idx=self.pad_value,
                use_norm=False,
            )
        else:
            raise ValueError(f"Unknown input_emb_style: {self.input_emb_style}")

        if self.use_chem_token:
            chem_encoder_config: DictConfig = model_config.chemical_encoder
            self.chem_encoder: ChemEncoder = ChemEncoder(
                drug_fps_path=chem_encoder_config.get("drug_fps_path"),
                d_out=self.d_model,
                padding_idx=chem_encoder_config.get("padding_idx", 0),
                activation=chem_encoder_config.get("activation", "leaky_relu"),
                freeze=chem_encoder_config.get("freeze", False),
            )

        encoder_layers: SCGPTBlock = SCGPTBlock(
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
        self.transformer_encoder: SCGPTEncoder = SCGPTEncoder(
            encoder_layers,
            self.n_layers,
            use_norm=self.norm_scheme == "pre",
            norm_config=self.norm_config,
            attn_config=self.attn_config,
        )

        expression_decoder_config: DictConfig = model_config.expression_decoder
        self.expression_decoder: ExprDecoder = ExprDecoder(
            d_model=self.d_model,
            n_outputs=expression_decoder_config.get("n_outputs", 1),
            n_layers=expression_decoder_config.get("n_layers", 2),
            activation=expression_decoder_config.get("activation", "leaky_relu"),
        )

        if model_config.mvc is not None:
            mvc_config: DictConfig = model_config.mvc
            self.mvc_decoder: MVCDecoder = MVCDecoder(
                d_model=self.d_model,
                arch_style=mvc_config.arch_style,
                query_activation=mvc_config.query_activation,
                scaled_dot_product=mvc_config.get("scaled_dot_product", False),
            )

        # Used for storing current gene token embeddings during forward passes
        self.cur_gene_token_embs: Optional[Tensor] = None

        if self.init_device != "meta":
            log.info(
                'MosaicML recommends using config.init_device="meta" with Composer + FSDP for faster initialization.',
            )
            self.apply(self.param_init_fn)

    def param_init_fn(self, module: nn.Module) -> None:
        """Initialize parameters for a given module using the configured
        initialization scheme.

        This method applies parameter initialization to modules that don't have the skip_init
        attribute set to True. The initialization scheme is specified in self.init_config
        and uses the llm-foundry parameter initialization functions.

        Args:
            module: The PyTorch module to initialize parameters for.
        """
        # skip initialization for modules that has skip_init=True
        if hasattr(module, "skip_init") and module.skip_init:
            log.info(f"Skipping re-initializing for {module._get_name()}")
            return
        init_fn_name: str = self.init_config["name"]
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
        """Encode gene tokens and expression values through the transformer
        encoder.

        This method processes gene identities and their corresponding expression values
        through the encoder components and transformer layers. It combines gene embeddings
        with expression embeddings and passes them through the transformer encoder in
        perceptual mode (no generative component).

        Args:
            src: Gene token IDs with shape (batch_size, seq_len). Contains integer
                indices corresponding to genes in the vocabulary.
            values: Gene expression values with shape (batch_size, seq_len). Can be
                either continuous values or binned categorical indices depending on
                the configured input_emb_style.
            src_key_padding_mask: Boolean mask with shape (batch_size, seq_len) where
                True indicates valid positions and False indicates padding positions.

        Returns:
            Encoded representations with shape (batch_size, seq_len, d_model).
            These are contextualized embeddings that capture both gene identity
            and expression information processed through transformer layers.
        """
        src = self.gene_encoder(src)  # (batch, seq_len, embsize)
        self.cur_gene_token_embs = src
        values = self.expression_encoder(values)  # (batch, seq_len, embsize)
        total_embs = src + values
        output = self.transformer_encoder(
            pcpt_total_embs=total_embs,
            gen_total_embs=None,
            pcpt_key_padding_mask=src_key_padding_mask,
            gen_key_padding_mask=None,
        )
        return output  # (batch, seq_len, embsize)

    def transformer_generate(
        self,
        pcpt_genes: Tensor,
        pcpt_values: Tensor,
        pcpt_key_padding_mask: Tensor,
        gen_genes: Tensor,
        gen_key_padding_mask: Tensor,
        input_cell_emb: Optional[Tensor] = None,
        drug_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Generate transformer outputs in perception-generation mode.

        This method implements the core generative functionality where the model uses
        a subset of genes (perception) to generate predictions for another subset of
        genes (generation). This is the key mechanism for masked gene expression
        modeling and generative pre-training.

        The method processes both perceptual and generative gene sequences through
        separate embedding paths, optionally incorporates chemical perturbation
        information, and passes them through the transformer encoder to produce
        contextualized representations for both sequences.

        Args:
            pcpt_genes: Perceptual gene token IDs with shape (batch_size, pcpt_len).
                These are the genes used as context for generation.
            pcpt_values: Perceptual gene expression values with shape (batch_size, pcpt_len).
                Expression levels corresponding to the perceptual genes.
            pcpt_key_padding_mask: Boolean mask for perceptual sequence with shape
                (batch_size, pcpt_len). True indicates valid positions.
            gen_genes: Generative gene token IDs with shape (batch_size, gen_len).
                These are the genes to be predicted/generated.
            gen_key_padding_mask: Boolean mask for generative sequence with shape
                (batch_size, gen_len). True indicates valid positions.
            input_cell_emb: Optional pre-computed cell embeddings with shape
                (batch_size, d_model). If provided, replaces the CLS token embedding
                in the perceptual sequence.
            drug_ids: Optional drug/chemical compound IDs with shape (batch_size,).
                Used only when use_chem_token is True for modeling chemical perturbations.

        Returns:
            A tuple containing:
            - pcpt_output: Contextualized embeddings for perceptual genes with shape
              (batch_size, pcpt_len, d_model)
            - gen_output: Contextualized embeddings for generative genes with shape
              (batch_size, gen_len, d_model). Can be None if gen_genes is None.
        """

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

        pcpt_output, gen_output = self.transformer_encoder(
            pcpt_total_embs=pcpt_total_embs,
            gen_total_embs=gen_total_embs,
            pcpt_key_padding_mask=pcpt_key_padding_mask,
            gen_key_padding_mask=gen_key_padding_mask,
        )

        return pcpt_output, gen_output

    def _get_cell_emb_from_layer(
        self,
        layer_output: Tensor,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """Extract cell-level embeddings from transformer layer outputs.

        This method aggregates sequence-level representations into a single cell-level
        embedding using one of three strategies: CLS token, average pooling, or
        weighted pooling. This is crucial for downstream tasks that require a single
        representation per cell (e.g., cell type classification, cell similarity).

        Args:
            layer_output: Transformer output with shape (batch_size, seq_len, d_model).
                Contains contextualized embeddings for each gene in the sequence.
            weights: Optional attention weights with shape (batch_size, seq_len).
                Only used when cell_emb_style is "w-pool". Higher weights indicate
                more important genes for the cell representation.

        Returns:
            Cell embeddings with shape (batch_size, d_model). Each row represents
            the aggregated cell-level representation derived from the gene sequence.

        Raises:
            ValueError: If weights is None when cell_emb_style is "w-pool", or if
                weights has incorrect dimensionality.
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
        """Extend model output with additional decoder predictions.

        This method augments the base model output dictionary with additional
        predictions from auxiliary decoders. It always adds cell embeddings and
        optionally includes predictions from classification (CLS) and masked
        value prediction (MVC) decoders.

        Args:
            output: Existing output dictionary containing model predictions.
                Will be modified in-place to include additional outputs.
            transformer_output: Transformer encoder outputs with shape
                (batch_size, seq_len, d_model). Used for generating cell embeddings
                and as input to auxiliary decoders.
            CLS: Whether to include cell type classification predictions.
                Requires that self.cls_decoder is defined.
            MVC: Whether to include masked value prediction outputs.
                Requires that self.mvc_decoder is defined.

        Returns:
            Extended output dictionary containing the original predictions plus:
            - 'cell_emb': Cell-level embeddings (always included)
            - 'cls_output': Classification logits (if CLS=True)
            - 'mvc_output': Masked value predictions (if MVC=True)
        """
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
        """Forward pass router that delegates to generative or perceptual mode.

        This method serves as the main entry point for model forward passes.
        It routes the call to either generative_forward (for perception-generation
        training) or perceptual_forward (for discriminative tasks or inference)
        based on the 'generative_training' keyword argument.

        Args:
            *args: Positional arguments passed through to the specific forward method.
            **kwargs: Keyword arguments including 'generative_training' (required)
                which determines the forward mode, plus other method-specific arguments.

        Returns:
            Dictionary containing model predictions with keys depending on the mode:
            - For generative mode: 'pcpt_preds', 'gen_preds', 'cell_emb', etc.
            - For perceptual mode: 'pcpt_preds', 'cell_emb', etc.

        Raises:
            ValueError: If 'generative_training' is not specified in kwargs.
        """
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
        """Forward pass for generative training mode (perception-generation).

        This method implements the core generative training paradigm where the model
        uses a subset of genes (perception) to predict expression values for another
        subset of genes (generation). This is the primary training mode for foundation
        model pre-training and supports various auxiliary objectives.

        The method processes both perceptual and generative gene sequences, passes them
        through the transformer encoder, and uses the expression decoder to predict
        expression values for all genes. It can optionally include auxiliary predictions
        from classification and MVC decoders.

        Args:
            pcpt_genes: Perceptual gene token IDs with shape (batch_size, pcpt_len).
                These genes provide context for predicting the generative genes.
            pcpt_values: Perceptual gene expression values with shape (batch_size, pcpt_len).
                Known expression levels for the perceptual genes.
            pcpt_key_padding_mask: Boolean mask for perceptual genes with shape
                (batch_size, pcpt_len). True indicates valid positions.
            gen_genes: Generative gene token IDs with shape (batch_size, gen_len).
                These are the genes whose expressions will be predicted.
            gen_key_padding_mask: Boolean mask for generative genes with shape
                (batch_size, gen_len). True indicates valid positions.
            CLS: Whether to include cell type classification predictions.
            MVC: Whether to include masked value prediction (gene expression reconstruction).
            input_cell_emb: Optional pre-computed cell embeddings with shape
                (batch_size, d_model). Replaces CLS token if provided.
            drug_ids: Optional drug IDs with shape (batch_size,) for chemical perturbation
                modeling. Only used when use_chem_token is True.

        Returns:
            Dictionary containing:
            - 'pcpt_preds': Predictions for perceptual genes with shape (batch_size, pcpt_len)
            - 'gen_preds': Predictions for generative genes with shape (batch_size, gen_len)
            - 'cell_emb': Cell-level embeddings with shape (batch_size, d_model)
            - 'cls_output': Classification logits if CLS=True
            - 'mvc_output': MVC predictions if MVC=True
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
        """Forward pass for perceptual (discriminative) mode.

        This method processes complete gene expression profiles without any generative
        component. It's used for discriminative tasks like cell type classification,
        embedding extraction, and inference. The entire gene sequence is processed
        through the encoder to produce contextualized representations.

        Unlike generative mode, this method treats all input genes as observed and
        produces predictions for the same genes (useful for denoising) or extracts
        cell-level representations for downstream tasks.

        Args:
            src: Gene token IDs with shape (batch_size, seq_len). Contains all genes
                to be processed without masking or generation splits.
            values: Gene expression values with shape (batch_size, seq_len).
                Observed expression levels for all genes.
            src_key_padding_mask: Boolean mask with shape (batch_size, seq_len).
                True indicates valid gene positions, False indicates padding.
            CLS: Whether to include cell type classification predictions using
                the cls_decoder.
            MVC: Whether to include masked value prediction outputs using the
                mvc_decoder for gene expression reconstruction.

        Returns:
            Dictionary containing:
            - 'pcpt_preds': Expression predictions for all genes with shape
              (batch_size, seq_len)
            - 'cell_emb': Cell-level embeddings with shape (batch_size, d_model)
            - 'cls_output': Classification logits if CLS=True
            - 'mvc_output': MVC gene predictions if MVC=True
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

    def fsdp_wrap_fn(self, module: nn.Module) -> bool:
        """Determine if a module should be wrapped with FSDP (Fully Sharded Data
        Parallel).

        This function is used by PyTorch's FSDP to determine which modules should be
        individually wrapped for memory-efficient distributed training. Wrapping
        transformer blocks allows for fine-grained sharding of model parameters.

        Args:
            module: PyTorch module to evaluate for FSDP wrapping.

        Returns:
            True if the module should be wrapped with FSDP, False otherwise.
            Currently returns True only for SCGPTBlock instances.
        """
        return isinstance(module, SCGPTBlock)

    def activation_checkpointing_fn(self, module: nn.Module) -> bool:
        """Determine if a module should use activation checkpointing.

        Activation checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This is particularly useful
        for large transformer models where memory is a constraint.

        Args:
            module: PyTorch module to evaluate for activation checkpointing.

        Returns:
            True if the module should use activation checkpointing, False otherwise.
            Currently returns True only for SCGPTBlock instances.
        """
        return isinstance(module, SCGPTBlock)


class ComposerSCGPTModel(ComposerModel):
    """Composer wrapper for SCGPTModel enabling distributed training and
    evaluation.

    This class wraps the SCGPTModel with MosaicML Composer's ComposerModel interface,
    providing standardized training loops, metrics computation, loss calculation, and
    distributed training support. It handles the training and validation logic for
    the foundation model pre-training with support for multiple objectives.

    The model supports multiple training objectives:
    - MSE (Mean Squared Error): Primary reconstruction loss for generative genes
    - MVC (Masked Value Prediction): Auxiliary loss for gene expression reconstruction
    - GEN (Cell-Conditioned Generation): Optional conditioned generation loss
    - Spearman correlation: Validation metric for expression correlation

    Key features:
    - Automatic loss computation combining multiple objectives
    - Built-in metrics tracking for training and validation
    - Support for cell-conditioned generation training
    - FLOP counting for performance monitoring
    - Output scaling for normalized training

    Args:
        model_config: Model architecture configuration passed to SCGPTModel.
        collator_config: Data collation configuration for padding and tokenization.
        device: Device for model parameters ('cpu', 'cuda', etc.).

    Attributes:
        model: The underlying SCGPTModel instance
        criterion: Loss function (masked_mse_loss)
        pad_token_id: Token ID used for padding genes
        use_cell_conditioned_generation: Whether to use cell-conditioned generation
        n_active_params: Number of trainable parameters
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        standard_scale_outputs: Whether to apply standard scaling to outputs
        collator_config: Stored collator configuration
        model_config: Stored model configuration
    """

    def __init__(
        self,
        model_config: DictConfig,
        collator_config: DictConfig,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.criterion = masked_mse_loss
        self.pad_token_id: int = collator_config.pad_token_id
        self.use_cell_conditioned_generation: bool = model_config.get(
            "use_cell_conditioned_generation",
            False,
        )
        self.model: SCGPTModel = SCGPTModel(
            model_config=model_config,
            collator_config=collator_config,
            device=device,
        )
        self.n_active_params: int = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.train_metrics: Dict[str, MaskedMseMetric] = {
            "MSE": MaskedMseMetric(name="MSE"),
            "MVC": MaskedMseMetric(name="MVC"),
        }
        self.standard_scale_outputs: bool = model_config.get(
            "standard_scale_outputs",
            False,
        )
        self.collator_config: DictConfig = collator_config
        self.model_config: DictConfig = model_config

        self.val_metrics: Dict[str, Union[MaskedMseMetric, MaskedSpearmanMetric]] = {
            "MSE": MaskedMseMetric(name="MSE"),
            "MVC": MaskedMseMetric(name="MVC"),
            "Spearman": MaskedSpearmanMetric(name="Spearman"),
        }
        if self.use_cell_conditioned_generation:
            self.train_gen: MaskedMseMetric = MaskedMseMetric(name="GEN")
            self.train_metrics.update({"GEN": self.train_gen})
        if self.use_cell_conditioned_generation:
            self.val_gen: MaskedMseMetric = MaskedMseMetric(name="GEN")
            self.val_metrics.update({"GEN": self.val_gen})

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass for training with automatic batch processing.

        This method processes a training batch through the SCGPT model in generative
        mode, handling the perception-generation split and optionally performing
        cell-conditioned generation for enhanced training.

        Args:
            batch: Dictionary containing batched training data with keys:
                - 'pcpt_gene': Perceptual gene token IDs (batch_size, pcpt_len)
                - 'pcpt_expr': Perceptual gene expressions (batch_size, pcpt_len)
                - 'gen_gene': Generative gene token IDs (batch_size, gen_len)
                - 'drug_ids': Optional drug IDs for chemical perturbation (batch_size,)

        Returns:
            Dictionary containing model outputs:
            - 'pcpt_preds': Perceptual gene predictions
            - 'gen_preds': Generative gene predictions
            - 'mvc_output': MVC predictions
            - 'cell_emb': Cell embeddings
            - 'cell_conditioned_gen_preds': Conditioned generation predictions (optional)
        """
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

    def eval_forward(
        self,
        batch: Dict[str, Tensor],
        outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Optional[Dict[str, Tensor]]:
        """Evaluation forward pass with optional output caching.

        This method handles forward passes during evaluation/validation. It can
        either use pre-computed outputs (for efficiency) or compute new outputs
        by calling the standard forward method.

        Args:
            batch: Dictionary containing evaluation batch data with same format as forward().
            outputs: Optional pre-computed model outputs. If provided, these are
                returned directly without additional computation.

        Returns:
            Model outputs dictionary or None. If outputs parameter is provided,
            returns it directly. Otherwise computes and returns new outputs.
        """
        if outputs:
            return outputs

        self.model.zero_grad(set_to_none=True)

        return outputs if outputs is not None else self.forward(batch)

    def loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        """Compute combined loss from multiple training objectives.

        This method calculates the total training loss by combining multiple objectives:
        1. MSE loss: Primary reconstruction loss for generative gene predictions
        2. MVC loss: Auxiliary masked value prediction loss
        3. GEN loss: Optional cell-conditioned generation loss (if enabled)

        The losses are averaged to provide a balanced training signal across objectives.

        Args:
            outputs: Dictionary of model predictions containing:
                - 'gen_preds': Generative gene predictions
                - 'mvc_output': MVC decoder predictions
                - 'cell_conditioned_gen_preds': Cell-conditioned predictions (optional)
            batch: Dictionary of batch data containing:
                - 'pcpt_gene': Perceptual gene tokens for sequence length reference
                - 'gen_gene': Generative gene tokens for masking
                - 'gen_expr_target': Target expression values for generative genes

        Returns:
            Combined scalar loss tensor averaged across enabled objectives.
        """
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

    def update_metric(
        self,
        batch: Dict[str, Tensor],
        outputs: Dict[str, Tensor],
        metric: Union[MaskedMseMetric, MaskedSpearmanMetric],
    ) -> None:
        """Update a specific metric with batch predictions and targets.

        This method extracts the appropriate predictions and targets based on the
        metric name and updates the metric's internal state. Different metrics
        use different prediction sources and may apply different preprocessing.

        Args:
            batch: Dictionary containing batch data with target values and masks.
            outputs: Dictionary containing model predictions from forward pass.
            metric: The metric object to update (MSE, MVC, GEN, or Spearman).

        Raises:
            ValueError: If metric name is not recognized or if cell-conditioned
                generation is not enabled for GEN metrics.
        """
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

    def get_metrics(
        self,
        is_train: bool = False,
    ) -> Dict[str, Union[MaskedMseMetric, MaskedSpearmanMetric]]:
        """Get the appropriate metrics dictionary for training or validation.

        Args:
            is_train: Whether to return training metrics (True) or validation metrics (False).

        Returns:
            Dictionary of metrics appropriate for the current phase.
        """
        # defines which metrics to use in each phase of training
        metric_dict = self.train_metrics if is_train else self.val_metrics
        return metric_dict

    def flops_per_batch(self, batch: Dict[str, Tensor]) -> int:
        """Calculate the number of floating-point operations (FLOPs) for a
        batch.

        This method estimates the computational cost of processing a batch by
        calculating parameter FLOPs and attention FLOPs. This is used for
        performance monitoring and computational budgeting.

        Args:
            batch: Dictionary containing batch data with gene sequences.

        Returns:
            Estimated number of FLOPs for processing this batch.
        """
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

    def scale_outputs(self, x: Tensor) -> Tensor:
        """Apply standard scaling to model outputs.

        This method normalizes outputs from the discrete bin range [1, num_bins-1]
        to the continuous range [-1, 1]. This is useful for stabilizing training
        when using binned expression values.

        Args:
            x: Input tensor with values in the range [1, num_bins-1].

        Returns:
            Scaled tensor with values in the range [-1, 1].
        """
        min_value = 1
        max_value = self.collator_config.num_bins - 1
        normalized_value = (x - min_value) / (max_value - min_value)
        # Scale to -1..1
        return 2 * normalized_value - 1


class ComposerSCGPTPerturbationModel(ComposerModel):
    """Composer wrapper for SCGPT perturbation prediction model.

    This class specializes the SCGPTModel for perturbation prediction tasks, where
    the model predicts gene expression changes in response to genetic or chemical
    perturbations. It extends the base model with perturbation-specific encoders
    and decoders for modeling intervention effects.

    The model processes control gene expressions and perturbation flags to predict
    post-perturbation gene expression profiles. This is particularly useful for
    drug discovery, genetic screening, and understanding cellular responses to
    interventions.

    Key components:
    - Base SCGPTModel: Provides the transformer backbone
    - Perturbation encoder: Embeds perturbation type flags (0: control, 1: perturbed, 2: masked)
    - Affine expression decoder: Predicts expression changes relative to control

    Args:
        model_config: Model architecture configuration for the underlying SCGPTModel.
        collator_config: Data collation configuration for tokenization and padding.
        device: Device for model parameters ('cuda', 'cpu', etc.). Defaults to 'cuda'.

    Attributes:
        device: Device where model parameters are located
        criterion: Loss function (masked_mse_loss)
        pad_token_id: Token ID for padding genes
        use_cell_conditioned_generation: Whether cell-conditioned generation is enabled
        model: The underlying SCGPTModel instance
        pert_encoder: Embedding layer for perturbation flags
        pert_decoder: Affine decoder for predicting expression changes
    """

    def __init__(
        self,
        model_config: DictConfig,
        collator_config: DictConfig,
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.device: str = device
        self.criterion = masked_mse_loss
        self.pad_token_id: int = collator_config.pad_token_id
        self.use_cell_conditioned_generation: bool = model_config.get(
            "use_cell_conditioned_generation",
            False,
        )
        self.model: SCGPTModel = SCGPTModel(
            model_config=model_config,
            collator_config=collator_config,
            device=device,
        )
        self.pert_encoder: nn.Embedding = nn.Embedding(
            3,
            self.model.d_model,
            padding_idx=0,
        )
        self.pert_decoder: AffineExprDecoder = AffineExprDecoder(self.model.d_model)

    def forward(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass for perturbation prediction.

        This method processes control gene expressions and perturbation flags to
        predict post-perturbation gene expression profiles. It combines gene
        identity embeddings, control expression embeddings, and perturbation
        flag embeddings, then processes them through the transformer to predict
        expression changes.

        Args:
            batch: Dictionary containing:
                - 'genes': Gene token IDs with shape (batch_size, seq_len)
                - 'expressions_ctrl': Control expression values (batch_size, seq_len)
                - 'perturb_flags': Perturbation flags (batch_size, seq_len) where
                  0=control, 1=perturbed, 2=masked

        Returns:
            Dictionary containing:
            - 'predicted_expr_perturbed': Predicted post-perturbation expressions
              with shape (batch_size, seq_len)
        """
        gene_ids = batch["genes"]
        ctrl_expr = batch["expressions_ctrl"]
        perturbation_flags = batch["perturb_flags"]

        gene_token_emb = self.model.gene_encoder(gene_ids.to(self.device))
        gene_expr_emb = self.model.expression_encoder(ctrl_expr.to(self.device))
        pert_flag_emb = self.pert_encoder(perturbation_flags.to(self.device))
        combined_input_embs = gene_token_emb + gene_expr_emb + pert_flag_emb

        transformer_encoding = self.model.transformer_encoder(
            pcpt_total_embs=combined_input_embs,
            gen_total_embs=None,
            pcpt_key_padding_mask=None,
            gen_key_padding_mask=None,
        )
        predicted_post_expr = self.pert_decoder(transformer_encoding, ctrl_expr)
        output = {
            "predicted_expr_perturbed": predicted_post_expr["pred"],
        }
        return output

    def loss(self, outputs: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Tensor:
        """Compute the perturbation prediction loss.

        This method calculates the mean squared error between predicted and actual
        post-perturbation gene expressions. The loss is computed over all genes
        using a uniform mask (no padding-based masking in this implementation).

        Args:
            outputs: Dictionary containing model predictions with key:
                - 'predicted_expr_perturbed': Predicted post-perturbation expressions
            batch: Dictionary containing batch data with keys:
                - 'expressions_perturbed': Target post-perturbation expressions
                - 'genes': Gene token IDs (used for mask shape)

        Returns:
            Scalar loss tensor representing the MSE between predictions and targets.
        """
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
