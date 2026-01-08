import json
import os
import tempfile
from functools import partial
from pathlib import Path

import lightning as L
import optuna
import pandas as pd
import torch
import torch.nn as nn
import uncertainty_toolbox as uct
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import DeterministicRegression

from src.datamodules.UCIMLRepoDataModule import UCIMLDatasetId, UCIMLRepoDataModule


class MLPService:
    ACTIVATION_CANDIDATES = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    def __init__(
        self,
        n_input: int = 1,
        n_hidden: list[int] = [50, 50],
        n_output: int = 1,
        lr: float = 1e-3,
        activation_fn: nn.Module = nn.ReLU(),
        loss_fn: nn.Module = nn.MSELoss(),
        max_epochs: int = 500,
        patience: int = 20,
        batch_size: int = 32,
        store_hps: Path | None = None,
    ):
        seed_everything(42)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = lr
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.store_hps = store_hps
        self._build_network()
        self._build_model()

    def _build_network(self):
        self._network = MLP(
            n_inputs=self.n_input,
            n_hidden=self.n_hidden,
            n_outputs=self.n_output,
            activation_fn=self.activation_fn,
        )

    def _build_model(self):
        self._model = DeterministicRegression(
            model=self.network,
            loss_fn=self.loss_fn,
            optimizer=partial(torch.optim.Adam, lr=self.lr),  # type: ignore
        )

    @property
    def network(self):
        return self._network

    @property
    def model(self):
        return self._model

    def tune(self, temp_file, dm, n_trials: int = 50):
        def objective(trial: optuna.trial.Trial):
            self.lr = trial.suggest_float("lr", 1e-5, 1e-1)
            self.batch_size = trial.suggest_categorical("bs", [32, 64, 128, 256])
            # n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
            # n_units = []
            # for layer in range(n_hidden_layers):
            #    n_units.append(
            #        trial.suggest_categorical(f"unit_{layer}", [8, 16, 32, 64, 128])
            #    )
            # self.n_hidden = n_units

            # activation_candidate = trial.suggest_categorical(
            #    "activation_fn", choices=list(self.ACTIVATION_CANDIDATES)
            # )
            # self.activation_fn = self.ACTIVATION_CANDIDATES[activation_candidate]
            self._build_network()
            self._build_model()
            self.train(dm, temp_file)

            metrics_path = os.path.join(
                temp_file,
                "lightning_logs",
                sorted(
                    os.listdir(os.path.join(temp_file, "lightning_logs")),
                    key=lambda x: int(x.split("_")[1]),
                )[-1],
                "metrics.csv",
            )
            df = pd.read_csv(metrics_path)
            val_loss = df[df["val_loss"].notna()]["val_loss"].iloc[-1]
            return val_loss
            # preds = self.model.predict_step(dm.X_test)
            # return uct.metrics_accuracy.r2_score(
            #     dm.Y_test.flatten().numpy(), preds["pred"].flatten().numpy()
            # )

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1)

        best_params = study.best_params
        if self.store_hps:
            self.save_hps(best_params, filename=self.store_hps)
        return best_params

    def train(self, dm, temp_file):
        dm.batch_size = self.batch_size
        logger = CSVLogger(temp_file)
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            logger=logger,
            callbacks=[EarlyStopping(monitor="val_loss", patience=self.patience)],
            deterministic=True,
        )
        trainer.fit(self.model, datamodule=dm)

    def load_hps(self, filename: Path):
        """Loads hyperparameters from a JSON file."""
        with open(filename, "r") as f:
            hps = json.load(f)

        self.lr = hps["lr"]
        self.batch_size = hps["bs"]
        # self.n_hidden = [hps[f"unit_{i}"] for i in range(hps["n_hidden_layers"])]
        # self.activation_fn = self.ACTIVATION_CANDIDATES[hps["activation_fn"]]
        self._build_network()
        self._build_model()

    def save_hps(self, best_hps: dict, filename: Path):
        with open(filename, "w") as f:
            json.dump(best_hps, f, indent=4)


if __name__ == "__main__":
    temp_file = tempfile.mkdtemp()
    dm = UCIMLRepoDataModule(id=UCIMLDatasetId.AUTO_MPG, shift=False, batch_size=128)
    mlp_service = MLPService(
        n_input=dm.X_train.shape[1],
        n_hidden=[32, 16, 8],
        lr=0.001,
        activation_fn=nn.ReLU(),
        batch_size=128,
    )
    mlp_service.train(dm, temp_file=temp_file)
    preds = mlp_service.model.predict_step(dm.X_test)
    print(
        uct.metrics_accuracy.r2_score(
            dm.Y_test.flatten().numpy(), preds["pred"].flatten().numpy()
        )
    )
    print(
        uct.get_all_accuracy_metrics(
            preds["pred"].flatten().numpy(), dm.Y_test.flatten().numpy()
        )
    )
