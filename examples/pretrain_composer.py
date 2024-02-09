import datasets
import torch
from composer import Trainer
from composer.models import ComposerModel
from torch.utils.data import DataLoader

import scgpt as scg
from scgpt import logger
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from composer.utils import dist


# import composer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


raw_dataset = datasets.load_from_disk("/vevo/cellxgene/cellxgene_primary_2023-12-15_0_cls_appended.dataset")
raw_dataset = raw_dataset.with_format("torch")
raw_dataset = raw_dataset.with_format("torch")
raw_dataset = raw_dataset.train_test_split(
    test_size=0.03, shuffle=True
)
train_dataset = raw_dataset["train"]
valid_dataset = raw_dataset["test"]
logger.info(f"train set number of samples: {len(train_dataset)}, ")
logger.info(f"valid set number of samples: {len(valid_dataset)}, ")

vocab = GeneVocab.from_file("/vevo/cellxgene/cellxgene_primary_2023-12-15_vocab.json")
special_tokens = ["<pad>", "<cls>", "<eoc>"]
for s in special_tokens:
    if s not in vocab:
        vocab.append_token(s)
collator = scg.DataCollator(
    do_padding=True,
    pad_token_id=vocab["<pad>"],
    pad_value=-2,
    do_mlm=True,
    do_binning=True,
    mlm_probability=0.40,
    mask_value=-1,
    max_length=1024,
    sampling=True,
    data_style="both",
)
train_sampler = dist.get_sampler(train_dataset, shuffle=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=2048,
    collate_fn=collator,
    drop_last=False,
    num_workers=16,
    pin_memory=True,
    prefetch_factor=4,
    sampler=train_sampler,
)
valid_sampler = dist.get_sampler(valid_dataset, shuffle=False)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=2048,
    collate_fn=collator,
    drop_last=False,
    num_workers=16,
    pin_memory=True,
    sampler=valid_sampler,
)


class scGPTComposer(ComposerModel):
    def __init__(self, vocab):
        super().__init__()
        self.criterion = masked_mse_loss
        self.vocab = vocab
        ntokens = len(vocab)
        self.model = TransformerModel(
            ntokens,
            d_model=512,
            nhead=8,
            d_hid=512,
            nlayers=12,
            nlayers_cls=3,
            n_cls=1,
            vocab=vocab,
            dropout=0.1,
            pad_token="<pad>",
            pad_value=-2,
            do_mvc=True,
            do_dab=False,
            use_batch_labels=False,  # TODO: try using batch labels, may help MVC
            input_emb_style="continuous",
            n_input_bins=51,
            use_generative_training=True,
            use_fast_transformer=True,
            fast_transformer_backend="flash",
        )
        self.pad_token = "<pad>"

    def forward(self, batch):  # batch is the output of the dataloader
        # specify how batches are passed through the model
        data_dict = batch
        pcpt_gene = data_dict["pcpt_gene"]
        pcpt_expr = data_dict["pcpt_expr"]
        pcpt_key_padding_mask = pcpt_gene.eq(self.vocab[self.pad_token])
        gen_gene = data_dict["gen_gene"]
        gen_expr_target = target_values = data_dict["gen_expr_target"]
        gen_key_padding_mask = gen_gene.eq(self.vocab[self.pad_token])
        output_dict = self.model(
            pcpt_gene,
            pcpt_expr,
            pcpt_key_padding_mask,
            gen_gene,
            gen_key_padding_mask,
            MVC=True,
            generative_training=True,
        )
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
        data_dict = batch
        pcpt_gene = data_dict["pcpt_gene"]
        # pcpt_expr = data_dict["pcpt_expr"]
        # pcpt_key_padding_mask = pcpt_gene.eq(self.model.vocab[self.model.pad_token])
        gen_gene = data_dict["gen_gene"]
        gen_expr_target = data_dict["gen_expr_target"]
        gen_key_padding_mask = gen_gene.eq(self.vocab[self.pad_token])
        positions_to_match = ~gen_key_padding_mask

        gen_expr_preds = outputs["gen_preds"]

        loss_mse = self.criterion(
            gen_expr_preds, gen_expr_target, positions_to_match
        )
        loss_mvc = self.criterion(
            outputs["mvc_output"][:, pcpt_gene.shape[1]:],
            gen_expr_target,
            positions_to_match,
        )

        loss_gen = self.criterion(outputs["GEPC"], gen_expr_target, positions_to_match)

        loss = loss_mse + loss_mvc + loss_gen

        return loss


model = scGPTComposer(vocab)
logger.info(f"Total Model parameters: {count_parameters(model.model) / (10**6)} M parameters")
for name, sub_model in model.model.named_children():
    logger.info(f"{name}: {count_parameters(sub_model) / (10**6)} M parameters")

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
trainer = Trainer(
    model=model,
    optimizers=optimizer,
    train_dataloader=train_loader,
    max_duration='10ep',
    device="gpu",
    device_train_microbatch_size="auto",
    precision='amp_fp16'
)
trainer.fit()
