# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
# Copyright (C) Vevo Therapeutics 2024-2025. All rights reserved.
import types
from contextlib import nullcontext

import torch
from omegaconf import OmegaConf


def test_train_smoke(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "scanpy", types.ModuleType("scanpy"))
    monkeypatch.setitem(sys.modules, "anndata", types.SimpleNamespace(AnnData=object))
    from scripts import train

    vocab = train.GeneVocab(["A", "B", "C"], specials=["<pad>"])
    vocab.pad_token = "<pad>"
    monkeypatch.setattr(train.GeneVocab, "from_file", classmethod(lambda cls, path: vocab))
    monkeypatch.setattr(train, "download_file_from_s3_url", lambda *a, **k: None)

    def dummy_build_dataloader(vocab, loader_cfg, collator_cfg, device_batch_size):
        batch = {
            "pcpt_gene": torch.tensor([[1, 2, 0, 0]]),
            "pcpt_expr": torch.rand(1, 4),
            "pcpt_expr_raw": torch.rand(1, 4),
            "gen_gene": torch.tensor([[3, 0]]),
            "gen_expr_target": torch.rand(1, 2),
            "gen_expr_raw": torch.rand(1, 2),
        }
        class DummyLoader(list):
            def __init__(self, b):
                super().__init__([b])
                self.dataset = types.SimpleNamespace(size=1)

        class DummyDataSpec:
            def __init__(self, b):
                self.dataloader = DummyLoader(b)
                self.num_samples = 1

            def __iter__(self):
                return iter(self.dataloader)

        return DummyDataSpec(batch)

    monkeypatch.setattr(train, "build_dataloader", dummy_build_dataloader)

    class DummyModel:
        def __init__(self, model_config, collator_config, device=None):
            self.pad_token_id = collator_config.pad_token_id
            self.model = types.SimpleNamespace(named_children=lambda: [])

        def forward(self, batch):
            pcpt_mask = ~batch["pcpt_gene"].eq(self.pad_token_id)
            gen_mask = ~batch["gen_gene"].eq(self.pad_token_id)
            assert pcpt_mask.shape == batch["pcpt_gene"].shape
            assert gen_mask.shape == batch["gen_gene"].shape
            return {
                "gen_preds": torch.zeros_like(batch["gen_expr_target"]),
                "mvc_output": torch.zeros(
                    batch["pcpt_gene"].shape[0],
                    batch["pcpt_gene"].shape[1] + batch["gen_gene"].shape[1],
                ),
            }

        def eval_forward(self, batch, outputs=None):
            return self.forward(batch)

        def loss(self, outputs, batch):
            return torch.tensor(0.0)

        def get_metrics(self, is_train=False):
            return {}

        def update_metric(self, batch, outputs, metric):
            pass

        def parameters(self):
            return []

    monkeypatch.setattr(train, "ComposerSCGPTModel", DummyModel)

    class DummyTrainer:
        def __init__(self, model, train_dataloader, eval_dataloader=None, **kwargs):
            self.model = model
            self.train_dataloader = train_dataloader
            self.state = types.SimpleNamespace(run_name="")

        def fit(self):
            for batch in self.train_dataloader:
                self.model.forward(batch)
                break

    monkeypatch.setattr(train.composer, "Trainer", DummyTrainer)
    monkeypatch.setattr(train, "build_scheduler", lambda *a, **k: [])
    monkeypatch.setattr(train, "build_optimizer", lambda *a, **k: types.SimpleNamespace())

    monkeypatch.setattr(train.dist, "initialize_dist", lambda *a, **k: None)
    monkeypatch.setattr(train, "update_batch_size_info", lambda cfg: cfg)
    monkeypatch.setattr(train, "process_init_device", lambda *a, **k: nullcontext())

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "dist_timeout": 0.1,
            "device_train_batch_size": 1,
            "device_eval_batch_size": 1,
            "max_duration": "1ba",
            "eval_interval": "1ba",
            "eval_subset_num_batches": 1,
            "precision": "fp32",
            "model": {"attn_config": {"attn_impl": "torch", "use_attn_mask": False}},
            "optimizer": {"name": "sgd", "lr": 0.001},
            "scheduler": {"name": "constant"},
            "train_loader": {"dataset": {}, "drop_last": False},
            "valid_loader": {"dataset": {}, "drop_last": False},
            "collator": {
                "pad_token_id": 0,
                "pad_value": 0,
                "mlm_probability": 0.15,
                "mask_value": -1,
                "do_padding": True,
                "max_length": 4,
                "sampling": False,
                "data_style": "both",
            },
            "vocabulary": {"path": "tests/vocab.json", "remote": "", "local": ""},
            "run_name": "test",
            "save_folder": str(tmp_path),
        }
    )

    train.main(cfg)


def test_inference_smoke(tmp_path, monkeypatch):
    from scripts.inference import save_embeddings as infer

    from datasets import Dataset

    sample = {
        "genes": [[1, 2, 0, 0]],
        "expressions": [[1.0, 2.0, 0.0, 0.0]],
        "drug": [0],
        "sample": [0],
        "cell_line": [0],
        "BARCODE_SUB_LIB_ID": ["0"],
        "cell_line_id": [0],
    }
    ds = Dataset.from_dict(sample)
    monkeypatch.setattr(infer, "load_dataset", lambda *a, **k: ds)

    class DummyModel:
        def __init__(self, model_config, collator_config):
            self.pad_token_id = collator_config.pad_token_id
            self.model = self

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state, strict=True):
            pass

        def _encode(self, ids, expr, src_key_padding_mask):
            assert src_key_padding_mask.shape == ids.shape
            return torch.zeros(ids.shape[0], ids.shape[1], 4)

    monkeypatch.setattr(infer, "ComposerSCGPTModel", DummyModel)
    monkeypatch.setattr(infer.torch, "load", lambda *a, **k: {"state": {"model": {}}})

    class DummyWriter:
        def __init__(self, path, schema, use_dictionary=True):
            self.closed = False

        def write_table(self, table):
            pass

        def close(self):
            self.closed = True

    dummy_pa = types.SimpleNamespace(
        schema=lambda *a, **k: None,
        field=lambda *a, **k: None,
        dictionary=lambda *a, **k: None,
        int32=lambda *a, **k: None,
        string=lambda *a, **k: None,
        list_=lambda *a, **k: None,
        float32=lambda *a, **k: None,
        Table=types.SimpleNamespace(from_pydict=lambda *a, **k: None),
    )
    monkeypatch.setattr(infer, "pa", dummy_pa)
    monkeypatch.setattr(infer.pq, "ParquetWriter", DummyWriter)

    class DummyPbar:
        def __init__(self, total, desc):
            pass

        def update(self, x):
            pass

        def close(self):
            pass

    monkeypatch.setattr(infer, "tqdm", lambda **kw: DummyPbar(kw["total"], kw["desc"]))
    monkeypatch.setattr(
        infer.streaming,
        "StreamingDataLoader",
        lambda dataset, batch_size, collate_fn, **kw: [collate_fn([dataset[0]])],
    )

    coll_cfg = {
        "pad_token_id": 0,
        "pad_value": 0,
        "mlm_probability": 0.15,
        "mask_value": -1,
        "do_padding": True,
        "max_length": 4,
        "sampling": False,
        "data_style": "pcpt",
        "do_binning": False,
        "log_transform": False,
        "num_bins": 1,
    }
    model_cfg = {
        "vocab_size": 4,
        "n_layers": 1,
        "n_heads": 1,
        "d_model": 4,
        "expansion_ratio": 1,
        "use_generative_training": False,
        "precision": "fp32",
        "attn_config": {"attn_impl": "torch", "use_attn_mask": False},
        "gene_encoder": {"use_norm": False, "embeddings": {}},
        "expression_encoder": {"input_emb_style": "continuous", "use_norm": False},
        "expression_decoder": {"n_outputs": 1},
    }

    import yaml

    coll_cfg_path = tmp_path / "collator_config.yml"
    model_cfg_path = tmp_path / "model_config.yml"
    with open(coll_cfg_path, "w") as f:
        yaml.safe_dump(coll_cfg, f)
    with open(model_cfg_path, "w") as f:
        yaml.safe_dump(model_cfg, f)
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"dummy")

    cfg = OmegaConf.create(
        {
            "paths": {
                "vocab_file": "tests/vocab.json",
                "collator_config_path": str(coll_cfg_path),
                "model_config_path": str(model_cfg_path),
                "model_file": str(model_file),
                "output_dir": str(tmp_path),
            },
            "model": {"attn_impl": "torch", "use_attn_mask": False},
            "dataset": {"name": "dummy", "split": "train", "streaming": False},
            "data": {
                "max_length": 4,
                "batch_size": 1,
                "num_workers": 0,
                "prefetch_factor": 2,
                "reserve_keys": [
                    "drug",
                    "sample",
                    "cell_line",
                    "BARCODE_SUB_LIB_ID",
                    "cell_line_id",
                ],
            },
            "output": {"prefix": "emb"},
            "parquet": {"chunk_size": 1},
        }
    )

    infer.main(cfg)

