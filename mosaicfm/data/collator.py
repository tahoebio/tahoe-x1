# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import json
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch
from composer.utils import dist
from transformers import DefaultDataCollator

from mosaicfm.tokenizer import GeneVocab
from mosaicfm.utils import download_file_from_s3_url


class DataCollator(DefaultDataCollator):
    """Data collator for the mask value learning task. It pads the sequences to
    the maximum length in the batch and masks the gene expression values.

    Args:
        vocab (:obj: GeneVocab): The vocabulary that includes the gene ids, name, special tokens, etc.
        use_chem_token (:obj:`bool`): whether to create and use the chemical token in the sequence.
        drug_to_id_path (:obj:`dict`): path to the drug to id .json file.
        do_padding (:obj:`bool`): whether to pad the sequences to the max length.
        unexp_padding (:obj:`bool`): whether to pad the sequences with unexpressed genes. If False it pads with pad token.
        pad_token_id (:obj:`int`, optional): the token id to use for padding.
            This is required if do_padding is True.
        pad_value (:obj:`int`): the value to use for padding the expression
            values to the max length.
        do_mlm (:obj:`bool`): whether to do masking with MLM.
        do_binning (:obj:`bool`): whether to bin the expression values.
        log_transform (:obj:`bool`): whether to transform the gene expression values.
        target_sum (:obj:`int`): The target sum of the normalized counts before log1p transformation.
        mlm_probability (:obj:`float`): the probability of masking with MLM.
        mask_value (:obj:`int`): the value to fill at the expression postions
            that are masked.
        max_length (:obj:`int`, optional): the maximum length of the sequences.
            This is required if do_padding is True.
        sampling (:obj:`bool`): whether to do sampling instead of truncation if
            length > max_length.
        reserve_keys (:obj:`List[str]`, optional): a list of keys in the examples
            to reserve in the output dictionary. Default to []. These fields
            will be kept unchanged in the output.
        keep_first_n_tokens (:obj:`int`): the number of tokens in the beginning
            of the sequence to keep unchanged from sampling. This is useful when
            special tokens have been added to the beginning of the sequence.
            Default to 1.
        data_style (:obj:`str`): the style of the data. If "pcpt", the data is
            masked and padded for perception training. If "gen", only the gene
            tokens are provided, but not the expression values, for pure generative
            training setting. If "both", the output will contain both fields above.
            Choices: "pcpt", "gen", "both". Default to "pcpt".
        num_bins (:obj:`int`): the number of bins to use for binning the expression
        right_binning (:obj:`bool`): whether to use right sided-binning. Torch default is False
    """

    def __init__(
        self,
        vocab: GeneVocab,
        drug_to_id_path: Optional[dict] = None,
        use_chem_token: bool = False,
        do_padding: bool = True,
        unexp_padding: bool = False,
        pad_token_id: Optional[int] = None,
        pad_value: int = 0,
        do_mlm: bool = True,
        do_binning: bool = True,
        log_transform: bool = False,
        target_sum: int = 10000,
        mlm_probability: Union[float, List[float]] = 0.15,
        mask_value: int = -1,
        max_length: Optional[int] = None,
        sampling: bool = True,
        reserve_keys: Optional[List[str]] = None,
        keep_first_n_tokens: int = 1,
        data_style: str = "pcpt",
        num_bins: int = 51,
        right_binning: bool = False,
        return_tensors: str = "pt",
    ):
        super().__init__(return_tensors=return_tensors)
        self.do_padding = do_padding
        self.unexp_padding = unexp_padding
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.do_mlm = do_mlm
        self.do_binning = do_binning
        self.log_transform = log_transform
        self.target_sum = target_sum
        self.mlm_probability = mlm_probability
        self.mask_value = mask_value
        self.max_length = max_length
        self.sampling = sampling
        self.reserve_keys = reserve_keys if reserve_keys is not None else []
        self.keep_first_n_tokens = keep_first_n_tokens
        self.data_style = data_style
        self.num_bins = num_bins
        self.right_binning = right_binning
        # filter non_special gene_ids
        gene_to_id = vocab.get_stoi()
        self.non_special_gene_ids = torch.tensor(
            [
                gene_id
                for gene_name, gene_id in gene_to_id.items()
                if not gene_name.startswith("<")
            ],
        )
        self.vocab = vocab
        self.use_chem_token = use_chem_token
        if self.use_chem_token:
            assert "<drug>" in vocab, "<drug> token must be in the vocabulary."
            self.drug_token_id = vocab["<drug>"]
        else:
            self.drug_token_id = None
        assert not self.use_chem_token or drug_to_id_path is not None, (
            "If `use_chem_token` is True, `drug_to_id_path` must be provided.",
        )
        assert drug_to_id_path is None or self.use_chem_token, (
            "If `drug_to_id_path` is provided, `use_chem_token` must be True.",
        )
        assert not self.use_chem_token or self.keep_first_n_tokens > 1, (
            "If `use_chem_token` is True, we need to keep <cls> and <drug> token in the beggining of pcpt_genes. So `keep_first_n_tokens` must be >=2!",
        )
        # load drug_to_id mapping if present
        if self.use_chem_token and drug_to_id_path is not None:
            if dist.get_local_rank() == 0:
                download_file_from_s3_url(
                    s3_url=drug_to_id_path["remote"],
                    local_file_path=drug_to_id_path["local"],
                )
            with dist.local_rank_zero_download_and_wait(drug_to_id_path["local"]):
                dist.barrier()

            with open(drug_to_id_path["local"]) as f:
                self.drug_to_id = json.load(f)

    def __post_init__(self):
        if self.do_padding:
            if self.pad_token_id is None:
                raise ValueError("`pad_token_id` is required if `do_padding`.")
            if self.max_length is None:
                raise ValueError("`max_length` is required if `do_padding`.")
        if self.do_binning and self.log_transform:
            raise ValueError(
                "Only one of `do_binning` and `log_transform` can be True.",
            )
        if self.unexp_padding and not self.do_padding:
            raise ValueError("`do_padding` should be be True if `unexp_padding`.")
        if isinstance(self.mlm_probability, float):
            if self.mlm_probability <= 0 or self.mlm_probability >= 1:
                raise ValueError("`mlm_probability` must be between 0 and 1.")
        elif isinstance(self.mlm_probability, (list, tuple)):
            if min(self.mlm_probability) <= 0 or max(self.mlm_probability) >= 1:
                raise ValueError("`mlm_probability` must be between 0 and 1.")
        else:
            raise ValueError("`mlm_probability` must be a float or iterable of floats.")

        if isinstance(self.reserve_keys, str):
            self.reserve_keys = [self.reserve_keys]

        if self.max_length is not None and (
            self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length
        ):
            raise ValueError(
                f"`keep_first_n_tokens` must be between 0 and `max_length` ({self.max_length}).",
            )

        if self.data_style not in ["pcpt", "gen", "both"]:
            raise ValueError("`data_style` must be one of 'pcpt', 'gen', 'both'.")

    def __call__(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            examples (:obj:`List[Dict[str, torch.Tensor]]`): a list of data dicts.
                Each dict is for one cell. It contains multiple 1 dimensional tensors
                like the following exmaple:
                    {'id': tensor(184117),
                    'genes': tensor([36572, 17868, ..., 17072]),
                    'expressions': tensor([ 0.,  2., ..., 18.])}

        Returns:
            :obj:`Dict[str, torch.Tensor]`: a dict of tensors.
        """
        for example in examples:
            if self.use_chem_token:
                drug = (
                    example["drug"]
                    if "drug" in example and example["drug"] in self.drug_to_id
                    else "<pad>"
                )
                example["drug_id"] = torch.as_tensor(
                    self.drug_to_id[drug],
                    dtype=torch.int,
                )
            if isinstance(example["genes"], list):
                example["genes"] = torch.as_tensor(example["genes"])
            example["genes"] = torch.squeeze(example["genes"])
            if isinstance(example["expressions"], list):
                example["expressions"] = torch.as_tensor(example["expressions"])
            example["expressions"] = torch.squeeze(example["expressions"])
        if len(self.reserve_keys) > 0:
            assert all(key in examples[0] for key in self.reserve_keys), (
                f"reserve_keys must be a subset of the keys in the examples. "
                f"Got {self.reserve_keys} but expected keys in {list(examples[0].keys())}."
            )

        if self.data_style == "pcpt":
            data_dict = self._call_pcpt(examples)
        elif self.data_style == "gen":
            data_dict = self._call_gen(examples)
        elif self.data_style == "both":
            data_dict = self._call_both(examples)
        else:
            raise ValueError(f"Unknown data_style: {self.data_style}")

        # add reserved keys
        device = examples[0]["genes"].device
        for key in self.reserve_keys:
            data_ = [example[key] for example in examples]
            if isinstance(data_[0], torch.Tensor):
                # if the reserved key is a tensor, stack them
                data_dict[key] = torch.stack(data_, dim=0).to(device)
            else:
                data_dict[key] = data_  # if not tensor, just keep the list

        return data_dict

    def _call_pcpt(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Each example is like:

            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])}

        Args:
            examples (:obj:`List[Dict[str, torch.Tensor]]`): a list of examples.
                Each example is a dictionary of tensors.
        Returns:
            :obj:`Dict[str, torch.Tensor]`: a dictionary of tensors.
        """
        if not isinstance(examples[0], Mapping):
            raise NotImplementedError

        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = (
            self.max_length
            if self.max_length is not None and max_ori_len >= self.max_length
            else max_ori_len
        )

        # pad and truncate
        padded_genes = []
        padded_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :],
                    n_bins=self.num_bins,
                    right=self.right_binning,
                )
            elif self.log_transform:
                assert not (
                    self.do_binning
                ), "Only one of `do_binning` and `log_transform` can be True."
                expressions[self.keep_first_n_tokens :] = log_transform(
                    row=expressions[self.keep_first_n_tokens :],
                    target_sum=self.target_sum,
                )
            genes, expressions = self._sample_or_truncate_plus_pad(
                genes,
                expressions,
                max_length=_max_length,
            )  # torch tensors of length _max_length
            padded_genes.append(genes)
            padded_expressions.append(expressions)

        padded_genes = torch.stack(padded_genes, dim=0).to(device)
        padded_expressions = torch.stack(padded_expressions, dim=0).to(device)

        data_dict = {
            "gene": padded_genes,
            "expr": padded_expressions,
        }

        # mask
        if self.do_mlm:
            masked_expressions = self._mask(
                padded_expressions,
                self.keep_first_n_tokens,
            )
        else:
            masked_expressions = padded_expressions
        data_dict["masked_expr"] = masked_expressions

        return data_dict

    def _call_gen(
        self,
        examples: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """This method will simply return the gene ids, with needed padding.
        There is no masking for pure generative training, and no input of expr
        values.

        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072])}

        Returns:
            Dict[str, torch.Tensor]: a dict of tensors.
            Example:
                {'pcpt_gene': tensor([[36572, 17868, ..., 17072],
                                        [36572, 17868, ..., 17072],
                                        ...,
                                        [36572, 17868, ..., 17072]]),
                'pcpt_expr': tensor([[ 0.,  2., ..., 18.],
                                        [ 0.,  2., ..., 18.],
                                        ...,
                                        [ 0.,  2., ..., 18.]])}
        """

        if not isinstance(examples[0], Mapping):
            raise NotImplementedError

        device = examples[0]["genes"].device

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = (
            self.max_length
            if self.max_length is not None and max_ori_len >= self.max_length
            else max_ori_len
        )

        # pad and truncate
        padded_pcpt_genes = []
        padded_pcpt_expressions = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :],
                    n_bins=self.num_bins,
                    right=self.right_binning,
                )
            elif self.log_transform:
                assert not (
                    self.do_binning
                ), "Only one of `do_binning` and `log_transform` can be True."
                expressions[self.keep_first_n_tokens :] = log_transform(
                    row=expressions[self.keep_first_n_tokens :],
                    target_sum=self.target_sum,
                )
            genes, expressions = self._sample_or_truncate_plus_pad(
                genes,
                expressions,
                max_length=_max_length,
            )
            padded_pcpt_genes.append(genes)
            padded_pcpt_expressions.append(expressions)

        padded_pcpt_genes = torch.stack(padded_pcpt_genes, dim=0).to(device)
        padded_pcpt_expressions = torch.stack(padded_pcpt_expressions, dim=0).to(device)

        data_dict = {
            "pcpt_gene": padded_pcpt_genes,
            "pcpt_expr": padded_pcpt_expressions,
        }
        return data_dict

    def _call_both(
        self,
        examples: List[Dict[str, torch.Tensor]],
        gen_prob: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """This method will split the input into the peception part and the
        generation part. The perception part will be processed into gene ids and
        expr values, and the generation part will be processed into gene ids
        only.

        By default, the mlm_probability will be used to select the genese assigned to
        the generation part.

        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])},
            'drug_id': Optinal = tensor(256), id 0 refers to <pad> token and indicates that drug is not available

        Args:
            gen_prob (float, optional): the probability of a gene being assigned to
                the generation part. If not provided, the mlm_probability will be used.

        Returns:
            Dict[str, torch.Tensor]: a dict of tensors.
            Example:
                {'pcpt_gene': tensor([[36572, 17868, ..., 17072],
                                        [36572, 17868, ..., 17072],
                                        ...,
                                        [36572, 17868, ..., 17072]]),
                'pcpt_expr': tensor([[ 0.,  2., ..., 18.],
                                        [ 0.,  2., ..., 18.],
                                        ...,
                                        [ 0.,  2., ..., 18.]]),
                'gen_gene': tensor([[36573, 17869, ..., 17073],
                                        [36573, 17869, ..., 17073],
                                        ...,
                                        [36573, 17869, ..., 17073]]),
                'gen_expr_target': tensor([[ 1.,  3., ..., 19.],
                                        [ 1.,  3., ..., 19.],
                                        ...,
                                        [ 1.,  3., ..., 19.]])}
        """
        if not isinstance(examples[0], Mapping):
            raise NotImplementedError

        if not self.do_mlm:
            # if not doing mlm, then the perceptrual part is the whole input
            return self._call_gen(examples)

        if gen_prob is None:
            gen_prob = self.get_mlm_probability()

        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = (
            self.max_length
            if self.max_length is not None and max_ori_len >= self.max_length
            else max_ori_len
        )

        gen_length = int((_max_length - self.keep_first_n_tokens) * gen_prob)
        pcpt_length = _max_length - gen_length  # perception part length

        # pad and truncate
        padded_pcpt_genes = []
        padded_pcpt_expressions = []
        padded_pcpt_original_exp = []
        padded_gen_genes = []
        padded_gen_expressions = []
        padded_gen_original_exp = []
        drug_ids = []

        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]

            if self.use_chem_token:
                # add drug token <drug>, and pad_value=-2 expression at location 1  (after <cls>) of genes and expressions
                genes = torch.cat(
                    (
                        genes[:1],
                        torch.tensor(
                            [self.drug_token_id],
                            device=genes.device,
                            dtype=genes.dtype,
                        ),
                        genes[1:],
                    ),
                )
                expressions = torch.cat(
                    (
                        expressions[:1],
                        torch.tensor(
                            [self.pad_value],
                            device=expressions.device,
                            dtype=expressions.dtype,
                        ),
                        expressions[1:],
                    ),
                )

            original_expressions = expressions.detach().clone()

            if self.do_binning:
                expressions[self.keep_first_n_tokens :] = binning(
                    row=expressions[self.keep_first_n_tokens :],
                    n_bins=self.num_bins,
                    right=self.right_binning,
                )
            elif self.log_transform:
                assert not (
                    self.do_binning
                ), "Only one of `do_binning` and `log_transform` can be True."
                expressions[self.keep_first_n_tokens :] = log_transform(
                    row=expressions[self.keep_first_n_tokens :],
                    target_sum=self.target_sum,
                )

            (
                gen_genes,
                gen_expressions,
                gen_original_exp,
                pcpt_genes,
                pcpt_expressions,
                pcpt_original_exp,
            ) = self._random_split(
                genes[self.keep_first_n_tokens :],
                expressions[self.keep_first_n_tokens :],
                original_expressions[self.keep_first_n_tokens :],
                ratio=gen_prob,
            )
            pcpt_genes = torch.cat(
                (genes[: self.keep_first_n_tokens], pcpt_genes),
                dim=0,
            )
            pcpt_expressions = torch.cat(
                (expressions[: self.keep_first_n_tokens], pcpt_expressions),
                dim=0,
            )

            pcpt_original_exp = torch.cat(
                (original_expressions[: self.keep_first_n_tokens], pcpt_original_exp),
                dim=0,
            )

            pcpt_genes, pcpt_expressions, pcpt_original_exp = (
                self._sample_or_truncate_plus_pad(
                    pcpt_genes,
                    pcpt_expressions,
                    pcpt_original_exp,
                    max_length=pcpt_length,
                )
            )  # torch tensors of length pcpt_length
            padded_pcpt_genes.append(pcpt_genes)
            padded_pcpt_expressions.append(pcpt_expressions)
            padded_pcpt_original_exp.append(pcpt_original_exp)

            gen_genes, gen_expressions, gen_original_exp = (
                self._sample_or_truncate_plus_pad(
                    gen_genes,
                    gen_expressions,
                    gen_original_exp,
                    max_length=gen_length,
                )
            )  # torch tensors of length gen_length
            padded_gen_genes.append(gen_genes)
            padded_gen_expressions.append(gen_expressions)
            padded_gen_original_exp.append(gen_original_exp)

            if self.use_chem_token:
                # add drug id, id=0 corresponds to <pad> which indicates that drug is not available
                drug = examples[i]["drug_id"]
                drug_ids.append(drug)

        padded_pcpt_genes = torch.stack(padded_pcpt_genes, dim=0)
        padded_pcpt_expressions = torch.stack(padded_pcpt_expressions, dim=0)
        padded_pcpt_original_exp = torch.stack(padded_pcpt_original_exp, dim=0)
        padded_gen_genes = torch.stack(padded_gen_genes, dim=0)
        padded_gen_expressions = torch.stack(padded_gen_expressions, dim=0)
        padded_gen_original_exp = torch.stack(padded_gen_original_exp, dim=0)

        if self.use_chem_token:
            drug_ids = torch.stack(drug_ids)

            data_dict = {
                "pcpt_gene": padded_pcpt_genes,
                "pcpt_expr": padded_pcpt_expressions,
                "pcpt_expr_raw": padded_pcpt_original_exp,  # "raw" means "not binned"
                "gen_gene": padded_gen_genes,
                "gen_expr_target": padded_gen_expressions,
                "gen_expr_raw": padded_gen_original_exp,  # "raw" means "not binned"
                "drug_ids": drug_ids,
            }
        else:
            data_dict = {
                "pcpt_gene": padded_pcpt_genes,
                "pcpt_expr": padded_pcpt_expressions,
                "pcpt_expr_raw": padded_pcpt_original_exp,  # "raw" means "not binned"
                "gen_gene": padded_gen_genes,
                "gen_expr_target": padded_gen_expressions,
                "gen_expr_raw": padded_gen_original_exp,  # "raw" means "not binned"
            }
        return data_dict

    def _random_split(
        self,
        *arrays: torch.Tensor,
        ratio: float,
    ) -> Tuple[torch.Tensor, ...]:
        """Randomly split the arrays into two parts. The first part will have
        the.

        length of `ratio * length`, and the second part will have the length of
        `(1 - ratio) * length`. When multiple arrays are provided, they are supposed
        to have the same length.

        This method reflects the behavior of `sklearn.model_selection.train_test_split`

        Args:
            *arrays (torch.Tensor): the arrays to be split.
            ratio (float): the ratio of the first part.

        Returns:
            Tuple[torch.Tensor, ...]: the split arrays.
        """
        assert len(arrays) > 0
        assert 0 < ratio < 1
        if len(arrays) > 1:
            assert all(
                array.shape[0] == arrays[0].shape[0] for array in arrays
            ), "The arrays must have the same length."

        length = arrays[0].shape[0]
        split_index = int(length * ratio)

        indices = torch.randperm(length, device=arrays[0].device)
        first_part_indices = indices[:split_index]
        second_part_indices = indices[split_index:]

        first_parts = tuple(array[first_part_indices] for array in arrays)
        second_parts = tuple(array[second_part_indices] for array in arrays)

        return first_parts + second_parts

    def get_mlm_probability(self) -> float:
        """Get the mlm probability for the current step."""
        if isinstance(self.mlm_probability, float):
            return self.mlm_probability
        elif isinstance(self.mlm_probability, list):
            # random choose a probability
            return np.random.choice(self.mlm_probability)
        else:
            raise ValueError(
                f"mlm_probability must be a float or a list of floats, but got {type(self.mlm_probability)} instead.",
            )

    def _mask(
        self,
        expressions: torch.Tensor,
        keep_first_n_tokens: int = 0,
    ) -> torch.Tensor:
        """Mask the expression values with MLM."""
        if keep_first_n_tokens > 0:
            result_ = self._mask(
                expressions[:, keep_first_n_tokens:],
                keep_first_n_tokens=0,
            )
            return torch.cat([expressions[:, :keep_first_n_tokens], result_], dim=1)

        device = expressions.device
        shape = expressions.shape

        probability_matrix = torch.full(shape, self.get_mlm_probability())
        # set padded postion probability to 0
        probability_matrix[expressions.eq(self.pad_value)] = 0
        if self.keep_first_n_tokens > 0:
            probability_matrix[:, : self.keep_first_n_tokens] = 0

        mask = torch.bernoulli(probability_matrix).bool()
        mask = mask.to(device)

        masked_expressions = expressions.masked_fill(mask, self.mask_value)
        return masked_expressions

    def _sample_or_truncate_plus_pad(
        self,
        *arrays: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, ...]:
        assert len(arrays) > 0
        assert max_length > 0
        if len(arrays) > 1:
            assert all(
                array.shape[0] == arrays[0].shape[0] for array in arrays
            ), "The arrays must have the same length."

        if len(arrays[0]) == max_length:
            return tuple(array for array in arrays)
        if len(arrays[0]) > max_length:  # sample or truncate
            if self.sampling:
                return self._sample(*arrays, max_length=max_length)
            else:
                return tuple(array[:max_length] for array in arrays)
        # We either pad by pad_token or pad by unexpressed genes
        elif self.unexp_padding:
            return self._pad_unexp_genes(*arrays, max_length=max_length)
        else:  # pad with pad token
            return self._pad(*arrays, max_length=max_length)

    def _sample(
        self,
        *arrays: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, ...]:
        # NOTE: the fastest way to sample in torch has been benchmarked here
        # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
        # it shows the randperm on gpu is the fastest.
        # NOTE: also, the current implementation permute the orders of the genes
        # and expressions, although it is probably a nice argmentation.
        assert len(arrays) > 0
        if len(arrays) > 1:
            assert all(
                array.shape[0] == arrays[0].shape[0] for array in arrays
            ), "The arrays must have the same length."

        device = arrays[0].device
        if self.keep_first_n_tokens == 0:
            indices = torch.randperm(len(arrays[0]), device=device)[:max_length]
        else:
            # keep the first n tokens unchanged
            _n = self.keep_first_n_tokens
            indices = torch.randperm(len(arrays[0]) - _n, device=device)[
                : max_length - _n
            ]
            indices = torch.cat([torch.arange(_n), indices + _n], dim=0)
        return tuple(array[indices] for array in arrays)

    def _pad(
        self,
        *arrays: torch.Tensor,  # First tensor is genes, rest are  expressions
        max_length: int,
    ):
        device = arrays[0].device
        return tuple(
            torch.cat(
                [
                    array,
                    torch.full(
                        (max_length - len(array),),
                        self.pad_token_id if i == 0 else self.pad_value,
                        dtype=array.dtype,
                        device=device,
                    ),
                ],
            )
            for i, array in enumerate(arrays)
        )

    def _pad_unexp_genes(
        self,
        *arrays: torch.Tensor,  # First tensor is genes, rest are  expressions respectively processed expressions, raw expressions (optional)
        max_length: int,
    ):
        device = arrays[0].device

        num_to_pad = max_length - len(arrays[0])

        # get list of all valid gene ids
        non_special_gene_ids = self.non_special_gene_ids.to(device)

        # filter out the expressed gene ids
        mask = ~torch.isin(non_special_gene_ids, arrays[0])
        unexp_genes = non_special_gene_ids[mask]

        # randomly sample from unexpressed gene ids
        idx = torch.randperm(unexp_genes.shape[0])[:num_to_pad]
        random_unexp_genes = unexp_genes[idx]

        # Pad the first tensor(gene_ids) with random unexpressed gene ids and the rest (expressions) with zeros.
        return tuple(
            torch.cat(
                [
                    array,
                    (
                        random_unexp_genes
                        if i == 0
                        else torch.zeros(num_to_pad, dtype=array.dtype, device=device)
                    ),
                ],
            )
            for i, array in enumerate(arrays)
        )


@torch.no_grad()
def log_transform(
    row: Union[np.ndarray, torch.Tensor],
    target_sum: int,
    eps: float = 1e-9,
) -> Union[np.ndarray, torch.Tensor]:
    """Log transform the row.

    Args:
        row (Union[np.ndarray, torch.Tensor]):
            The row to be log-1p-transformed.
        target_sum (int, optional):
            The target sum of the normalized row before log-1p transformation. Default to 10000.
        eps (float, optional):
            The epsilon value used for normalization.
    Returns:
        Union[np.ndarray, torch.Tensor]:
            The log-1p-transformed row.
    """
    dtype = row.dtype
    is_tensor = isinstance(row, torch.Tensor)
    if not is_tensor:
        row = torch.as_tensor(row)
    row = (row / (row.sum(axis=-1, keepdims=True) + eps)) * target_sum
    row = torch.log1p(row)
    if not is_tensor:
        return row.numpy().astype(dtype)
    return row


@torch.no_grad()
def binning(
    row: Union[np.ndarray, torch.Tensor],
    n_bins: int,
    right: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """Binning the row into n_bins.

    Args:
        row (Union[np.ndarray, torch.Tensor]):
            The row to be binned.
        n_bins (int):
            The number of bins.
        right (bool, optional):
            Argument passed to `torch.bucketize`. if False, return the first suitable location that is found.
            If True, return the last such index.
            If no suitable index found, return 0 for non-numerical value (eg. nan, inf) or the size of boundaries
            (one pass the last index). In other words, if False, gets the lower bound index for each value
            in input from boundaries. If True, gets the upper bound index instead. Default value is False.
    .
    Returns:
        Union[np.ndarray, torch.Tensor]:
            The binned row.
    """
    dtype = row.dtype
    return_np = not (isinstance(row, torch.Tensor))
    if not isinstance(row, torch.Tensor):
        row = torch.as_tensor(row)
    GRADES = torch.linspace(0, 1, n_bins - 1, dtype=torch.float32, requires_grad=False)
    if row.min() <= 0:
        non_zero_ids = row.nonzero()
        non_zero_row = row[non_zero_ids]
        bins = torch.quantile(non_zero_row, GRADES)
        non_zero_digits = torch.bucketize(non_zero_row, bins, right=right)
        binned_row = torch.zeros_like(row, dtype=non_zero_digits.dtype)
        binned_row[non_zero_ids] = non_zero_digits
    else:
        bins = torch.quantile(row, GRADES)
        binned_row = torch.bucketize(row, bins, right=right)
    if return_np:
        binned_row = binned_row.astype(dtype)
    if not (right):
        # Left sided binning satisfies the condition: bins[i - 1] < row < bins[i]
        # For right=False, the smallest binned values is 0
        # To avoid inconsistency: always make output in be in the range 1...n_bins-1
        binned_row = binned_row + 1
    # For right=True, the output will be in the range 1...n_bins-1
    return binned_row
