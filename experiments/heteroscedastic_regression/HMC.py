import json
from itertools import repeat
from pathlib import Path
from typing import Any, Dict

import hamiltorch
import lightning as L
import optuna
import torch
import torch.nn as nn
from lightning_uq_box.models import MLP


class HMCModule:
    ACTIVATION_CANDIDATES = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }

    def __init__(
        self,
        model_loss: str = "regression",
        tau_in: float = 1.0,
        tau_out: float = 110.0,
        seed: int = 42,
        step_size: float = 5e-4,
        num_samples: int = 1000,
        num_steps_per_sample: int = 30,
        burn: int = -1,
        mass: float = 1.0,
        store_on_GPU: bool = False,
        n_input: int = 1,
        n_hidden: list[int] = [50, 50],
        n_output: int = 1,
        activation_fn: nn.Module = nn.Tanh(),
        store_hps: Path | None = None,
    ):
        super().__init__()
        self.model_loss = model_loss
        self.tau_in = tau_in
        self.tau_out = tau_out
        self.seed = seed
        self.step_size = step_size
        self.num_samples = num_samples
        self.num_steps_per_sample = num_steps_per_sample
        self.burn = burn
        self.mass = mass
        self.store_on_GPU = store_on_GPU
        self.params_hmc = None
        self.n_hidden = n_hidden
        self.activation_fn = activation_fn
        self.n_input = n_input
        self.n_output = n_output
        self.store_hps = store_hps
        self._build_network()
        hamiltorch.set_random_seed(self.seed)

    def _build_network(self):
        self._network = MLP(
            n_inputs=self.n_input,
            n_hidden=self.n_hidden,
            n_outputs=self.n_output,
            activation_fn=self.activation_fn,
        )

    @property
    def network(self):
        return self._network

    def train_hmc(self, dm: L.LightningDataModule):
        tau_list = torch.tensor(
            list(repeat(self.tau_in, len(list(self.network.parameters()))))
        )
        params_init = hamiltorch.util.flatten(self.network)
        inv_mass = torch.ones(params_init.shape) / self.mass

        self.params_hmc = hamiltorch.sample_model(
            self.network,
            dm.X_train,
            dm.Y_train,
            params_init=params_init,
            model_loss=self.model_loss,
            num_samples=self.num_samples,
            burn=self.burn,
            inv_mass=inv_mass,
            step_size=self.step_size,
            num_steps_per_sample=self.num_steps_per_sample,
            tau_out=self.tau_out,
            tau_list=tau_list,
            store_on_GPU=self.store_on_GPU,
            sampler=hamiltorch.Sampler.HMC,
            integrator=hamiltorch.Integrator.IMPLICIT,
            metric=hamiltorch.Metric.HESSIAN,
            desired_accept_rate=0.8,
            verbose=True,
        )
        return self.params_hmc

    def predict_hmc(self, X: torch.Tensor, y: torch.Tensor):
        tau_list = torch.tensor(
            list(repeat(self.tau_in, len(list(self.network.parameters()))))
        )

        pred_list, log_probs_f = hamiltorch.predict_model(
            self.network,
            samples=self.params_hmc,
            x=X,
            y=y,
            model_loss=self.model_loss,
            tau_out=self.tau_out,
            tau_list=tau_list,
            verbose=False,
        )
        return pred_list, log_probs_f

    def hmc_preds_to_lightning_uq_box_preds(
        self, pred_list: torch.Tensor, burn: int = 0
    ) -> Dict[str, torch.Tensor]:
        m = pred_list[burn:].mean(0)
        epistemic = pred_list[burn:].std(0)
        aleatoric = (pred_list[burn:].var(0) + self.tau_out**-1) ** 0.5
        return {
            "pred": m.flatten(),
            "pred_uct": torch.sqrt(epistemic**2 + aleatoric**2).flatten(),
            "epistemic_uct": epistemic.flatten(),
            "aleatoric_uct": aleatoric.flatten(),
        }

    def tune_hps(self, dm: L.LightningDataModule, n_trials: int = 50) -> Dict[str, Any]:
        def objective(trial: optuna.Trial):
            # Hyperparameters to optimize
            self.step_size = trial.suggest_float("step_size", 1e-5, 1e-3)
            self.num_samples = trial.suggest_int("num_samples", 500, 2000)
            self.num_steps_per_sample = trial.suggest_int(
                "num_steps_per_sample", 10, 50
            )
            self.tau_out = trial.suggest_float("tau_out", 1.0, 100.0)
            # self.tau_in = trial.suggest_float("tau_in", 0.1, 10.0)
            self.mass = trial.suggest_float("mass", 0.1, 10.0)
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

            self.train_hmc(dm)

            # pred_list, _ = self.predict_hmc(dm.X_val, dm.Y_val)
            # preds = self.hmc_preds_to_lightning_uq_box_preds(pred_list)

            # r2 = uct.metrics_accuracy.r2_score(
            #     dm.Y_val.flatten().numpy(), preds["pred"].flatten().numpy()
            # )

            # return r2
            _, log_prob = self.predict_hmc(dm.X_val, dm.Y_val)
            return -torch.mean(torch.tensor(log_prob)).item()

        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=42)
        )
        try:
            study.optimize(objective, n_trials=n_trials)
        except KeyboardInterrupt:
            print("Attemp graceful shutdown")
        finally:
            if self.store_hps:
                self.save_hps(study.best_params, self.store_hps)
        best_params = study.best_params
        return best_params

    def save_hps(self, best_hps: dict, filename: Path):
        """Saves the hyperparameters to a file."""
        with open(filename, "w") as f:
            json.dump(best_hps, f, indent=4)

    def load_hps(self, filename: Path):
        """Loads the hyperparameters from a file."""
        with open(filename, "r") as f:
            hps = json.load(f)

        self.step_size = hps["step_size"]
        self.num_samples = hps["num_samples"]
        self.num_steps_per_sample = hps["num_steps_per_sample"]
        self.tau_out = hps["tau_out"]
        # self.tau_in = hps["tau_in"]
        self.mass = hps["mass"]
        # self.n_hidden = [hps[f"unit_{i}"] for i in range(hps["n_hidden_layers"])]
        # self.activation_fn = self.ACTIVATION_CANDIDATES[hps["activation_fn"]]
        self._build_network()
