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
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from lightning_uq_box.models import MLP
from lightning_uq_box.uq_methods import BNN_VI_Regression


class BNNVIService:
    ACTIVATION_CANDIDATES = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    def __init__(
        self,
        lr: float = 1e-2,
        max_epochs: int = 500,
        patience: int = 20,
        batch_size: int = 200,
        n_mc_samples_train: int = 10,
        n_mc_samples_test: int = 50,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0.0,
        prior_sigma: float = 1.0,
        posterior_mu_init: float = 0.0,
        posterior_rho_init: float = -6.0,
        alpha: float = 1e-3,
        bayesian_layer_type: str = "reparameterization",
        store_hps: Path | None = None,
        n_input: int = 1,
        n_hidden: list[int] = [50, 50],
        n_output: int = 1,
        activation_fn: nn.Module = nn.Tanh(),
    ):
        seed_everything(42)
        self.lr = lr
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.n_mc_samples_train = n_mc_samples_train
        self.n_mc_samples_test = n_mc_samples_test
        self.output_noise_scale = output_noise_scale
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.alpha = alpha
        self.bayesian_layer_type = bayesian_layer_type
        self.store_hps = store_hps
        self.n_hidden = n_hidden
        self.activation_fn = activation_fn
        self.n_input = n_input
        self.n_output = n_output
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
        self._model = BNN_VI_Regression(
            self.network,
            optimizer=partial(torch.optim.Adam, lr=self.lr),  # type: ignore
            n_mc_samples_train=self.n_mc_samples_train,
            n_mc_samples_test=self.n_mc_samples_test,
            output_noise_scale=self.output_noise_scale,
            prior_mu=self.prior_mu,
            prior_sigma=self.prior_sigma,
            posterior_mu_init=self.posterior_mu_init,
            posterior_rho_init=self.posterior_rho_init,
            alpha=self.alpha,
            bayesian_layer_type=self.bayesian_layer_type,
            stochastic_module_names=[-1],
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
            self.batch_size = trial.suggest_categorical("bs", [16, 32, 64, 128, 265])
            self.n_mc_samples_train = trial.suggest_int("n_mc_samples_train", 5, 20)
            self.n_mc_samples_test = trial.suggest_int("n_mc_samples_test", 20, 100)
            self.output_noise_scale = trial.suggest_float(
                "output_noise_scale", 0.5, 2.0
            )
            self.alpha = trial.suggest_float("alpha", 0.0, 1.0)

            # MLP stuff
            # n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 3)
            # units = []
            # for layer in range(n_hidden_layers):
            #    units.append(
            #        trial.suggest_categorical(f"unit_{layer}", [8, 16, 32, 64, 128])
            #    )
            # self.n_hidden = units
            # activation_candidate = trial.suggest_categorical(
            #    "activation_fn", list(self.ACTIVATION_CANDIDATES)
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
        self.n_mc_samples_train = hps["n_mc_samples_train"]
        self.n_mc_samples_test = hps["n_mc_samples_test"]
        self.output_noise_scale = hps["output_noise_scale"]
        self.alpha = hps["alpha"]

        # self.n_hidden = [
        #    hps[f"unit_{layer}"] for layer in range(hps["n_hidden_layers"])
        # ]
        # self.activation_fn = self.ACTIVATION_CANDIDATES[hps["activation_fn"]]
        self._build_network()
        self._build_model()

    def save_hps(self, best_hps: dict, filename: Path):
        with open(filename, "w") as f:
            json.dump(best_hps, f, indent=4)


if __name__ == "__main__":
    temp_file = tempfile.mkdtemp()
    dm = ToyHeteroscedasticDatamodule()

    bnnvi_service = BNNVIService(store_hps=Path("./bnnvi_best_hps.json"))
    best_hps = bnnvi_service.tune(temp_file, dm, n_trials=10)
    print(best_hps)
