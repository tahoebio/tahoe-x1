# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import copy
import gc
import logging
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Union

import composer
import torch
from composer.core.callback import Callback
from composer.utils import dist, get_device, reproducibility
from llmfoundry.utils.builders import (
    build_algorithm,
    build_callback,
    build_logger,
    build_optimizer,
    build_scheduler,
)
from llmfoundry.utils.config_utils import (
    log_config,
    pop_config,
    process_init_device,
    update_batch_size_info,
)
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from rich.traceback import install
from streaming.base.util import clean_stale_shared_memory

install()

from mosaicfm.data import build_dataloader
from mosaicfm.model import ComposerSCGPTModel
from mosaicfm.tokenizer import GeneVocab
from mosaicfm.utils import download_file_from_s3_url

log = logging.getLogger(__name__)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(cfg: DictConfig) -> composer.Trainer:
    # Resolve all interpolation variables as early as possible
    om.resolve(cfg)

    # Create copy of config for logging
    logged_cfg: DictConfig = copy.deepcopy(cfg)

    # Set seed first
    seed: int = pop_config(cfg, "seed", must_exist=True)
    reproducibility.seed_all(seed)

    # Initialize pytorch distributed training process groups
    dist_timeout: Union[int, float] = pop_config(
        cfg,
        "dist_timeout",
        must_exist=False,
        default_value=600.0,
    )
    dist.initialize_dist(get_device(None), timeout=dist_timeout)

    # Get global and device batch size information from distributed/single node setting
    cfg = update_batch_size_info(cfg)
    logged_cfg.update(cfg, merge=True)

    # Mandatory model training configs
    model_config: DictConfig = pop_config(cfg, "model", must_exist=True)
    optimizer_config: Dict[str, Any] = pop_config(
        cfg,
        "optimizer",
        must_exist=True,
        convert=True,
    )
    scheduler_config: Dict[str, Any] = pop_config(
        cfg,
        "scheduler",
        must_exist=True,
        convert=True,
    )
    train_loader_config: DictConfig = pop_config(cfg, "train_loader", must_exist=True)
    valid_loader_config: DictConfig = pop_config(cfg, "valid_loader", must_exist=True)
    collator_config: DictConfig = pop_config(cfg, "collator", must_exist=True)
    vocab_config: DictConfig = pop_config(cfg, "vocabulary", must_exist=True)
    # Optional deepspeed, FSDP, and torch-compile config
    deepspeed_config: Optional[Dict[str, Any]] = pop_config(
        cfg,
        "deepspeed_config",
        must_exist=False,
        default_value=None,
        convert=True,
    )
    compile_config: Optional[Dict[str, Any]] = pop_config(
        cfg,
        "compile_config",
        must_exist=False,
        default_value=None,
    )
    fsdp_config: Optional[Dict[str, Any]] = pop_config(
        cfg,
        "fsdp_config",
        must_exist=False,
        default_value=None,
        convert=True,
    )
    if (fsdp_config is not None) and (deepspeed_config is not None):
        raise ValueError("FSDP and DeepSpeed cannot be used together.")

    # Optional logging, evaluation and callback configs
    logger_configs: Optional[DictConfig] = pop_config(
        cfg,
        "loggers",
        must_exist=False,
        default_value=None,
        convert=True,
    )
    callback_configs: Optional[DictConfig] = pop_config(
        cfg,
        "callbacks",
        must_exist=False,
        default_value=None,
        convert=True,
    )
    algorithm_configs: Optional[DictConfig] = pop_config(
        cfg,
        "algorithms",
        must_exist=False,
        default_value=None,
    )

    # Mandatory hyperparameters for training
    device_train_batch_size: int = pop_config(
        cfg,
        "device_train_batch_size",
        must_exist=True,
    )
    device_eval_batch_size: int = pop_config(
        cfg,
        "device_eval_batch_size",
        must_exist=True,
    )
    max_duration: Union[int, str] = pop_config(cfg, "max_duration", must_exist=True)
    eval_interval: Union[int, str] = pop_config(
        cfg,
        "eval_interval",
        default_value="500ba",
        must_exist=False,
    )
    precision: str = pop_config(cfg, "precision", must_exist=True)

    # Optional parameters will be set to default values if not specified.
    default_run_name: str = os.environ.get("RUN_NAME")
    run_name: str = pop_config(
        cfg,
        "run_name",
        must_exist=False,
        default_value=default_run_name,
    )

    logged_cfg.update({"run_name": run_name})
    save_folder: Optional[str] = pop_config(
        cfg,
        "save_folder",
        must_exist=False,
        default_value=f"s3://vevo-ml-datasets/vevo-scgpt/models/{run_name}",
    )
    is_state_dict_sharded: bool = (
        (fsdp_config.get("state_dict_type", "full") == "sharded")
        if fsdp_config
        else False
    )

    save_latest_filename: str = pop_config(
        cfg,
        "save_latest_filename",
        must_exist=False,
        default_value=(
            "latest-sharded-rank{rank}"
            if is_state_dict_sharded
            else "latest-rank{rank}.pt"
        ),
    )

    save_overwrite: bool = pop_config(
        cfg,
        "save_overwrite",
        must_exist=False,
        default_value=False,
    )
    save_weights_only: bool = pop_config(
        cfg,
        "save_weights_only",
        must_exist=False,
        default_value=False,
    )
    save_filename: str = pop_config(
        cfg,
        "save_filename",
        must_exist=False,
        default_value="ep{epoch}-ba{batch}-rank{rank}.pt",
    )

    save_interval: Union[str, int] = pop_config(
        cfg,
        "save_interval",
        must_exist=False,
        default_value="250ba",
    )

    save_num_checkpoints_to_keep: int = pop_config(
        cfg,
        "save_num_checkpoints_to_keep",
        must_exist=False,
        default_value=-1,
    )

    progress_bar = pop_config(
        cfg,
        "progress_bar",
        must_exist=False,
        default_value=False,
    )
    log_to_console: bool = pop_config(
        cfg,
        "log_to_console",
        must_exist=False,
        default_value=True,
    )
    python_log_level: Optional[str] = pop_config(
        cfg,
        "python_log_level",
        must_exist=False,
        default_value="debug",
    )
    console_log_interval: Union[int, str] = pop_config(
        cfg,
        "console_log_interval",
        must_exist=False,
        default_value="1ba",
    )
    device_train_microbatch_size: Union[str, int] = pop_config(
        cfg,
        "device_train_microbatch_size",
        must_exist=False,
        default_value="auto",
    )

    load_path: str = pop_config(cfg, "load_path", must_exist=False, default_value=None)
    load_weights_only: bool = pop_config(
        cfg,
        "load_weights_only",
        must_exist=False,
        default_value=False,
    )
    load_strict_model_weights: bool = pop_config(
        cfg,
        "load_strict_model_weights",
        must_exist=False,
        default_value=True,
    )
    load_ignore_keys: Optional[List[str]] = pop_config(
        cfg,
        "load_ignore_keys",
        must_exist=False,
        default_value=None,
    )
    should_log_config: bool = pop_config(
        cfg,
        "log_config",
        must_exist=False,
        default_value=True,
    )
    # Enable autoresume from model checkpoints if possible
    autoresume_default: bool = False
    if (
        logged_cfg.get("run_name", None) is not None
        and save_folder is not None
        and not save_overwrite
        and not save_weights_only
    ):
        autoresume_default = True

    if cfg.get("autoresume") is None and autoresume_default:
        log.info(
            "As run_name, save_folder, and save_latest_filename are set, \
                    changing autoresume default to True...",
        )

    autoresume: bool = pop_config(
        cfg,
        "autoresume",
        must_exist=False,
        default_value=autoresume_default,
    )

    # Pop known unused parameters that are used as interpolation variables or
    # created by update_batch_size_info.
    pop_config(cfg, "data_local", must_exist=False)
    pop_config(cfg, "data_remote", must_exist=False)
    pop_config(cfg, "global_seed", must_exist=False)
    pop_config(cfg, "global_train_batch_size", must_exist=False)
    pop_config(cfg, "n_gpus", must_exist=False)
    pop_config(cfg, "device_train_grad_accum", must_exist=False)

    # Warn users for unused parameters
    for key in cfg:
        warnings.warn(
            f"Unused parameter {key} found in cfg. Please check your yaml to ensure this parameter is necessary.",
        )

    # Warn if fsdp is enabled but user only has 1 GPU
    if dist.get_world_size() == 1 and fsdp_config is not None:
        warnings.warn(
            "FSDP is not applicable for single-GPU training. Reverting to DDP.",
        )
        fsdp_config = None

    # Set logging level
    if python_log_level is not None:
        logging.basicConfig(
            # Example of format string
            # 2022-06-29 11:22:26,152: rank0[822018][MainThread]: INFO: Message here
            format=f"%(asctime)s: rank{dist.get_global_rank()}[%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
        )
        logging.getLogger("mosaicfm").setLevel(
            python_log_level.upper(),
        )  # vevo-scGPT module
        logging.getLogger(__name__).setLevel(python_log_level.upper())  # Train script

    # Initialize context
    init_context = process_init_device(model_config, fsdp_config)
    logged_cfg.update({"fsdp_config": fsdp_config}, merge=True)

    log.info("Downloading vocab...")
    if dist.get_local_rank() == 0:
        download_file_from_s3_url(
            s3_url=vocab_config["remote"],
            local_file_path=vocab_config["local"],
        )
    with dist.local_rank_zero_download_and_wait(vocab_config["local"]):
        dist.barrier()

    # Build vocab
    vocab = GeneVocab.from_file(vocab_config["local"])
    special_tokens = ["<pad>", "<cls>", "<eoc>"]

    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)
    if collator_config.get("use_junk_tokens", False):
        # Based on Karpathy's observation that 64 is a good number for performance
        # https://x.com/karpathy/status/1621578354024677377?s=20
        original_vocab_size = len(vocab)
        remainder = original_vocab_size % 64
        if remainder > 0:
            junk_tokens_needed = 64 - remainder
            for i in range(junk_tokens_needed):
                junk_token = f"<junk{i}>"
                vocab.append_token(junk_token)

    ## Update PAD token ID
    collator_config.pad_token_id = vocab["<pad>"]
    ## Update model config with Vocab Size
    model_config.vocab_size = len(vocab)
    log.info(f"Setting vocab size to: {len(vocab)}")
    logged_cfg.update({"vocab_size": len(vocab)})

    # Scheduler
    scheduler_name: str = scheduler_config.pop("name")
    scheduler = build_scheduler(scheduler_name, scheduler_config)

    # Loggers
    loggers = (
        [
            build_logger(str(name), logger_cfg)
            for name, logger_cfg in logger_configs.items()
        ]
        if logger_configs
        else []
    )

    # Callbacks
    callbacks: List[Callback] = (
        [
            build_callback(str(name), callback_cfg, om.to_container(logged_cfg))
            for name, callback_cfg in callback_configs.items()
        ]
        if callback_configs
        else []
    )

    # Algorithms
    algorithms = (
        [
            build_algorithm(str(name), algorithm_cfg)
            for name, algorithm_cfg in algorithm_configs.items()
        ]
        if algorithm_configs
        else None
    )

    # Build DataLoaders
    log.info("Building DataLoaders...")
    clean_stale_shared_memory()
    train_loader = build_dataloader(
        loader_cfg=train_loader_config,
        collator_cfg=collator_config,
        device_batch_size=device_train_batch_size,
    )
    log.info(f"train set number of samples: {(train_loader.dataloader.dataset.size)}")
    valid_loader = build_dataloader(
        loader_cfg=valid_loader_config,
        collator_cfg=collator_config,
        device_batch_size=device_eval_batch_size,
    )
    log.info(
        f"Validation set number of samples: {(valid_loader.dataloader.dataset.size)}",
    )
    logged_cfg.update(
        {
            "train_dataset_size": train_loader.num_samples,
            "valid_dataset_size": valid_loader.num_samples,
        },
    )
    with init_context:
        # Build Model
        model = ComposerSCGPTModel(
            model_config=model_config,
            collator_config=collator_config,
        )

    # Log number of parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logged_cfg.update(
        {
            "n_params": n_params,
            "n_trainable_params": n_trainable_params,
        },
    )
    log.info(f"Total parameters: {n_params / (10 ** 6)} M")
    log.info(f"Total trainable parameters: {n_trainable_params / (10 ** 6)} M ")
    for name, sub_model in model.model.named_children():
        log.info(f"{name}: {count_parameters(sub_model) / (10 ** 6)} M parameters")

    # Optimizer
    optimizer_name: str = optimizer_config.pop("name")
    optimizer = build_optimizer(model, optimizer_name, optimizer_config)

    # Build the Trainer
    log.info("Building Trainer...")
    trainer = composer.Trainer(
        run_name=run_name,
        seed=seed,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=valid_loader,
        optimizers=optimizer,
        schedulers=scheduler,
        max_duration=max_duration,
        eval_interval=eval_interval,
        progress_bar=progress_bar,
        log_to_console=log_to_console,
        console_log_interval=console_log_interval,
        loggers=loggers,
        callbacks=callbacks,
        precision=precision,
        algorithms=algorithms,
        device_train_microbatch_size=device_train_microbatch_size,
        fsdp_config=fsdp_config,
        deepspeed_config=deepspeed_config,
        save_folder=save_folder,
        save_filename=save_filename,
        save_latest_filename=save_latest_filename,
        save_interval=save_interval,
        save_num_checkpoints_to_keep=save_num_checkpoints_to_keep,
        save_overwrite=save_overwrite,
        save_weights_only=save_weights_only,
        load_path=load_path,
        load_weights_only=load_weights_only,
        load_strict_model_weights=load_strict_model_weights,
        load_ignore_keys=load_ignore_keys,
        autoresume=autoresume,
        dist_timeout=dist_timeout,
        compile_config=compile_config,
    )

    if should_log_config:
        log.info("Logging config")
        log_config(logged_cfg)
    torch.cuda.empty_cache()
    gc.collect()

    log.info("Starting training...")
    trainer.fit()
    log.info("Training finished.")
    return trainer


if __name__ == "__main__":
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    # Disable resolving environment variables through omegaconf.
    om.clear_resolver("oc.env")
    # Load yaml and cli arguments.
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    om.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    main(cfg)
