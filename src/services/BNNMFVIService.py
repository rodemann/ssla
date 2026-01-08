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
from lightning_uq_box.uq_methods import NLL, BNN_VI_ELBO_Regression


class BNNMFVIService:
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
        n_output: int = 2,
        lr: float = 3e-3,
        activation_fn: nn.Module = nn.ReLU(),
        max_epochs: int = 500,
        patience: int = 20,
        batch_size: int = 32,
        burnin_epochs: int = 20,
        num_mc_samples_train: int = 10,
        num_mc_samples_test: int = 25,
        output_noise_scale: float = 1.3,
        prior_mu: float = 0,
        prior_sigma: float = 1,
        posterior_mu_init: float = 0,
        posterior_rho_init: float = -5,
        bayesian_layer_type: str = "reparameterization",
        store_hps: Path | None = None,
    ):
        seed_everything(42)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = lr
        self.activation_fn = activation_fn
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.burnin_epochs = burnin_epochs
        self.num_mc_samples_train = num_mc_samples_train
        self.num_mc_samples_test = num_mc_samples_test
        self.output_noise_scale = output_noise_scale
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.posterior_mu_init = posterior_mu_init
        self.posterior_rho_init = posterior_rho_init
        self.bayesian_layer_type = bayesian_layer_type
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
        weights = torch.cat([w.view(-1) for w in self.network.parameters()])
        beta = weights.flatten().shape[0]
        self._model = BNN_VI_ELBO_Regression(
            self.network,
            optimizer=partial(torch.optim.Adam, lr=self.lr),  # type:ignore
            criterion=NLL(),
            burnin_epochs=self.burnin_epochs,
            beta=beta,
            num_mc_samples_train=self.num_mc_samples_train,
            num_mc_samples_test=self.num_mc_samples_test,
            output_noise_scale=self.output_noise_scale,
            prior_mu=self.prior_mu,
            prior_sigma=self.prior_sigma,
            posterior_mu_init=self.posterior_mu_init,
            posterior_rho_init=self.posterior_rho_init,
            bayesian_layer_type="reparameterization",
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
            self.burnin_epochs = trial.suggest_int("burnin_epochs", 50, 200)
            self.num_mc_samples_train = trial.suggest_int("num_mc_samples_train", 5, 50)
            self.num_mc_samples_test = trial.suggest_int("num_mc_samples_test", 10, 100)
            self.output_noise_scale = trial.suggest_float(
                "output_noise_scale", 0.5, 2.0
            )

            # MLP
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
        self.burnin_epochs = hps["burnin_epochs"]
        self.num_mc_samples_train = hps["num_mc_samples_train"]
        self.num_mc_samples_test = hps["num_mc_samples_test"]
        self.output_noise_scale = hps["output_noise_scale"]

        # self.n_hidden = [hps[f"unit_{i}"] for i in range(hps["n_hidden_layers"])]
        # self.activation_fn = self.ACTIVATION_CANDIDATES[hps["activation_fn"]]

        self._build_network()
        self._build_model()

    def save_hps(self, best_hps: dict, filename: Path):
        with open(filename, "w") as f:
            json.dump(best_hps, f, indent=4)


if __name__ == "__main__":
    temp_file = tempfile.mkdtemp()
    dm = ToyHeteroscedasticDatamodule()

    bnnvi_service = BNNMFVIService(store_hps=Path("./bnnmfvi_hps.json"))

    best_hps = bnnvi_service.tune(temp_file, dm, n_trials=10)
    print(best_hps)
