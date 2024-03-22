import composer
from composer import Trainer
from streaming import StreamingDataset, StreamingDataLoader
from composer.utils import dist, get_device

from scgpt import DataCollator
from scgpt import logger
from scgpt.model import ComposerSCGPTModel
from scgpt.tokenizer import GeneVocab
from scgpt.utils import download_file_from_s3_url


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    # streaming.base.util.clean_stale_shared_memory()
    dist_timeout = 600.0
    dist.initialize_dist(get_device(None), timeout=dist_timeout)
    train_dataset = StreamingDataset(
        remote="s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2023-12-15_MDS/train",
        local="mds-data-folder/train",
        # predownload=48 * 32 * 6,
        download_timeout=300,
        # split="train",
        allow_unsafe_types=True,
        shuffle=True,
    )
    composer.utils.dist.barrier()
    valid_dataset = StreamingDataset(
        remote="s3://vevo-ml-datasets/vevo-scgpt/datasets/cellxgene_primary_2023-12-15_MDS/val",
        local="mds-data-folder/val",
        # predownload=48 * 32 * 6,
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
        batch_size=8 * 256,
        collate_fn=collator,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=48,
        persistent_workers=True,
    )
    valid_loader = StreamingDataLoader(
        valid_dataset,
        batch_size=8 * 256,
        collate_fn=collator,
        num_workers=8,
        drop_last=False,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=48,
    )

    model = ComposerSCGPTModel(vocab)
    logger.info(
        f"Total Model parameters: {count_parameters(model.model) / (10 ** 6)} M parameters"
    )
    for name, sub_model in model.model.named_children():
        logger.info(f"{name}: {count_parameters(sub_model) / (10 ** 6)} M parameters")

    optimizer = composer.optim.DecoupledAdamW(model.parameters(), lr=2e-4,
                                              betas=(0.9, 0.95), eps=1e-8)
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
        device_train_microbatch_size=256,
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
