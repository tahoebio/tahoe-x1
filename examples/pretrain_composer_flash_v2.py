import composer
import datasets
import torch
from composer import Trainer
from composer.models import ComposerModel
from composer.utils import dist
from torch.utils.data import DataLoader
from torchmetrics import Metric

from scgpt import DataCollator
from scgpt import logger
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
import logging

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.basicConfig(level=logging.INFO)
class MaskedMseMetric(Metric):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.add_state(
            "sum_mse",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sum_mask",
            default=torch.tensor(0.0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )

    def update(
        self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> None:
        if preds.shape != target.shape:
            raise ValueError("preds and target must have the same shape")
        mask = mask.float()
        self.sum_mse += torch.nn.functional.mse_loss(
            preds * mask, target * mask, reduction="sum"
        )
        self.sum_mask += mask.sum()

    def compute(self) -> torch.Tensor:
        return self.sum_mse / self.sum_mask


raw_dataset = datasets.load_from_disk(
    "/vevo/cellxgene/cellxgene_primary_2023-12-15_merged.dataset"
)
raw_dataset = raw_dataset.with_format("torch")
raw_dataset = raw_dataset.train_test_split(
    test_size=0.03,
    shuffle=True,
    seed=44,
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
collator = DataCollator(
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
    batch_size=256*6,
    collate_fn=collator,
    drop_last=False,
    num_workers=16,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True,
    sampler=train_sampler,
)
valid_sampler = dist.get_sampler(valid_dataset, shuffle=False)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    collate_fn=collator,
    num_workers=1,
    drop_last=False,
    pin_memory=True,
    persistent_workers=True,
    sampler=valid_sampler,
    prefetch_factor=4,
)


class ScgptComposer(ComposerModel):
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
            pre_norm=True,
        )
        self.pad_token = "<pad>"
        self.train_mse = MaskedMseMetric(name="MSE")
        self.train_mvc = MaskedMseMetric(name="MVC")
        self.train_gen = MaskedMseMetric(name="GEN")
        self.val_mse = MaskedMseMetric(name="MSE")
        self.val_mvc = MaskedMseMetric(name="MVC")
        self.val_gen = MaskedMseMetric(name="GEN")

    def forward(self, batch):  # batch is the output of the dataloader
        # specify how batches are passed through the model
        pcpt_gene = batch["pcpt_gene"]
        pcpt_expr = batch["pcpt_expr"]
        pcpt_key_padding_mask = pcpt_gene.eq(self.vocab[self.pad_token])
        gen_gene = batch["gen_gene"]
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
        # previous_cell_embs = output_dict["cell_emb"].detach()
        # preds = self.model(
        #     pcpt_gene,
        #     pcpt_expr,
        #     pcpt_key_padding_mask,
        #     gen_gene,
        #     gen_key_padding_mask,
        #     MVC=False,
        #     input_cell_emb=previous_cell_embs,
        #     generative_training=True,
        # )["gen_preds"]
        # output_dict["GEPC"] = preds
        output_dict["GEPC"] = output_dict["gen_preds"]
        return output_dict

    def loss(self, outputs, batch):
        # pass batches and `forward` outputs to the loss
        pcpt_gene = batch["pcpt_gene"]
        gen_gene = batch["gen_gene"]
        gen_expr_target = batch["gen_expr_target"]
        gen_key_padding_mask = gen_gene.eq(self.vocab[self.pad_token])
        positions_to_match = ~gen_key_padding_mask

        gen_expr_preds = outputs["gen_preds"]

        loss_mse = self.criterion(gen_expr_preds, gen_expr_target, positions_to_match)
        loss_mvc = self.criterion(
            outputs["mvc_output"][:, pcpt_gene.shape[1] :],
            gen_expr_target,
            positions_to_match,
        )

        loss_gen = self.criterion(outputs["GEPC"], gen_expr_target, positions_to_match)

        loss = loss_mse + loss_mvc + loss_gen
        return loss

    def update_metric(self, batch, outputs, metric):
        pcpt_gene = batch["pcpt_gene"]
        gen_gene = batch["gen_gene"]
        mask = ~gen_gene.eq(self.vocab[self.pad_token])
        target = batch["gen_expr_target"]
        if metric.name == "MSE":
            preds = outputs["gen_preds"]
        elif metric.name == "MVC":
            preds = outputs["mvc_output"][:, pcpt_gene.shape[1] :]
        elif metric.name == "GEN":
            preds = outputs["GEPC"]
        else:
            raise ValueError(f"metric {metric.name} not recognized")
        metric.update(preds=preds, target=target, mask=mask)

    def get_metrics(self, is_train=False):
        # defines which metrics to use in each phase of training
        if is_train:
            metric_dict = {
                "MSE": self.train_mse,
                "MVC": self.train_mvc,
                "GEN": self.train_gen,
            }
        else:
            metric_dict = {
                "MSE": self.val_mse,
                "MVC": self.val_mvc,
                "GEN": self.val_gen,
            }
        return metric_dict


model = ScgptComposer(vocab)
logger.info(
    f"Total Model parameters: {count_parameters(model.model) / (10 ** 6)} M parameters"
)
for name, sub_model in model.model.named_children():
    logger.info(f"{name}: {count_parameters(sub_model) / (10 ** 6)} M parameters")

optimizer = composer.optim.DecoupledAdamW(model.parameters(), lr=0.0001)
scheduler = composer.optim.scheduler.CosineAnnealingWithWarmupScheduler(
    t_warmup="0.05dur", t_max="1dur", alpha_f=0.0
)

fsdp_config = {
    "sharding_strategy": "SHARD_GRAD_OP",
    "cpu_offload": False,  # Not supported yet
    "mixed_precision": "DEFAULT",
    "backward_prefetch": "BACKWARD_POST",
    "activation_checkpointing": False,
    "activation_cpu_offload": False,
    "verbose": True,
    "limit_all_gathers": True,
    "state_dict_type": "sharded",
    "sharded_ckpt_prefix_dir": "ba{batch}-shards",
}



# Instantiate the trainer
trainer = Trainer(
    model=model,
    optimizers=optimizer,
    train_dataloader=train_loader,
    max_duration="6ep",
    device="gpu",
    eval_dataloader=valid_loader,
    eval_interval="500ba",
    schedulers=scheduler,
    # fsdp_config=fsdp_config,
    device_train_microbatch_size=256,
    precision="amp_bf16",
    callbacks=[
        composer.callbacks.SpeedMonitor(
            window_size=20
        ),
        composer.callbacks.SystemMetricsMonitor(gpu_available=True)
    ],
    loggers=[composer.loggers.WandBLogger(project="vevo-scgpt", log_artifacts=False)],
    save_folder="/vevo/scgpt/checkpoints/{run_name}",
    save_interval="250ba",
    save_latest_filename="latest",
)
trainer.fit()
