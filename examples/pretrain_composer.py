import composer
import torch
from composer import Trainer

from composer.models import ComposerModel
from torchmetrics import Metric
from streaming import StreamingDataset, StreamingDataLoader
from composer.utils import dist, get_device

from scgpt import DataCollator
from scgpt import logger
from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import download_file_from_s3_url


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


class ScgptComposer(ComposerModel):
    def __init__(self, vocab):
        super().__init__()
        self.criterion = masked_mse_loss
        self.vocab = vocab
        ntokens = len(vocab)
        self.model = TransformerModel(
            ntokens,
            d_model=2048,
            nhead=16,
            d_hid=2048 * 4,
            nlayers=24,
            nlayers_cls=2,
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


def main():
    # streaming.base.util.clean_stale_shared_memory()
    dist_timeout = 600.0
    dist.initialize_dist(get_device(None), timeout=dist_timeout)
    train_dataset = StreamingDataset(
        remote="s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2023-12-15_MDS/train",
        local="mds-data-folder/train",
        predownload=48 * 32 * 6,
        download_timeout=300,
        # split="train",
        allow_unsafe_types=True,
        shuffle=True,
    )
    composer.utils.dist.barrier()
    valid_dataset = StreamingDataset(
        remote="s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2023-12-15_MDS/val",
        local="mds-data-folder/val",
        predownload=48 * 32 * 6,
        download_timeout=300,
        # split="val",
        allow_unsafe_types=True,
        shuffle=False,
    )
    composer.utils.dist.barrier()
    logger.info(f"train set number of samples: {(train_dataset.size)}, ")
    logger.info(f"valid set number of samples: {(valid_dataset.size)}, ")

    download_file_from_s3_url(
        "s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2023-12-15_MDS/cellxgene_primary_2023-12-15_vocab.json",
        "vocab.json",
    )
    vocab = GeneVocab.from_file("vocab.json")
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
    train_loader = StreamingDataLoader(
        train_dataset,
        batch_size=32 * 8,
        collate_fn=collator,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=48,
        persistent_workers=True,
    )
    valid_loader = StreamingDataLoader(
        valid_dataset,
        batch_size=32 * 8,
        collate_fn=collator,
        num_workers=8,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=48,
    )

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
    trainer = Trainer(
        model=model,
        optimizers=optimizer,
        train_dataloader=train_loader,
        max_duration="6ep",
        device="gpu",
        eval_dataloader=valid_loader,
        eval_interval="500ba",
        schedulers=scheduler,
        device_train_microbatch_size=32,
        precision="amp_bf16",
        deepspeed_config={
            "zero_optimization": {
                "stage": 2,
                "overlap_comm": "true",
                "reduce_scatter": "true",
            },
            "precision": "amp_bf16",
            "bf16": {"enabled": "true"},
        },
        callbacks=[composer.callbacks.SpeedMonitor(window_size=20)],
        loggers=[
            composer.loggers.WandBLogger(project="vevo-scgpt", log_artifacts=False),
            # composer.callbacks.RuntimeEstimator(skip_batches=10, time_unit="hours"),
        ],
        save_folder="s3://vevo-ml-datasets/vevo-scgpt/models/{run_name}",
        save_interval="250ba",
        save_latest_filename="latest",
    )
    trainer.fit()


if __name__ == "__main__":
    main()
