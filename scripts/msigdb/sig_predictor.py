# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GeneSigDataset(torch.utils.data.Dataset):
    def __init__(self, hit_matrix, embs):
        assert hit_matrix.index.equals(embs.index)
        self.embs = embs
        self.hit_matrix = hit_matrix

    def __len__(self):
        return len(self.embs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.embs.values[idx], dtype=torch.float32),
            torch.tensor(self.hit_matrix.values[idx], dtype=torch.float32),
            self.embs.index[idx],
        )

    def get_gene_names(self):
        return self.embs.index.values


class SigPredictor:
    def __init__(
        self,
        sig_names,
        emb_dim: int = 512,
        n_hidden_layers: int = 1,
        hidden_size: int = 25,
        dropout: float = 0.1,
    ):
        self.model = _SigPredictorModule(
            input_size=emb_dim,
            n_hidden_layers=n_hidden_layers,
            hidden_size=hidden_size,
            dropout=dropout,
            output_size=len(sig_names),
        )
        self.sig_names = sig_names
        self.trained = False

    def fit(
        self,
        train_dataset: GeneSigDataset,
        val_dataset: GeneSigDataset,
        batch_size=256,
        max_epochs=10000,
        enable_progress_bar: bool = True,
    ):
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=enable_progress_bar,
            log_every_n_steps=10,
        )
        trainer.fit(
            self.model,
            torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            ),
            torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
            ),
        )
        self.model.load_state_dict(self.model.best_state)
        self.trained = True

    @torch.no_grad()
    def predict(self, dataset: GeneSigDataset):
        """Predict on a dataset and return the predictions."""
        assert self.trained, "Model has not been trained"
        self.model.eval()
        preds = []
        gene_names = []
        for inputs, _, names in torch.utils.data.DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        ):
            preds.append(self.model(inputs).detach().numpy())
            gene_names += names
        return (
            pd.DataFrame(
                np.concatenate(preds),
                index=gene_names,
                columns=self.sig_names,
            )
            .melt(ignore_index=False, value_name="prediction", var_name="sig")
            .reset_index(names="gene")
        )


class _SigPredictorModule(pl.LightningModule):
    def __init__(
        self,
        input_size,
        output_size,
        n_hidden_layers: int = 1,
        hidden_size: int = 25,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
    ):
        super(_SigPredictorModule, self).__init__()
        self.save_hyperparameters()
        layers = [
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.LeakyReLU(),
                ],
            )
        layers.extend(
            [nn.Dropout(dropout), nn.Linear(hidden_size, output_size), nn.Sigmoid()],
        )

        self.net = nn.Sequential(*layers)
        self.learning_rate = learning_rate
        self.best_val_loss = np.inf
        self.best_state = None

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        loss = F.binary_cross_entropy(self(inputs), labels)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=inputs.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels, _ = batch
        loss = F.binary_cross_entropy(self(inputs), labels)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=inputs.size(0),
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        val_loss = self.trainer.callback_metrics["val_loss"].item()
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_state = self.state_dict()
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer=optimizer, patience=7)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "reduce_on_plateau": True,
            },
        ]

    def configure_callbacks(self):
        return [
            EarlyStopping(
                monitor="val_loss",
                patience=15,
                verbose=False,
                check_finite=False,
            ),
        ]
