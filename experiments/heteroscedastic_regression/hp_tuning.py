import tempfile
from pathlib import Path

import torch.nn as nn
from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule

from experiments.heteroscedastic_regression.HMC import HMCModule
from src.services import BNNMFVIService, BNNVIService, MLPService

EXPERIMENT_PATH = Path("./experiments/heteroscedastic_regression")
HPS_PATH = EXPERIMENT_PATH / "hps"

if __name__ == "__main__":
    temp_file = tempfile.mkdtemp()
    dm = ToyHeteroscedasticDatamodule()

    # HMC
    MODEL_PATH = HPS_PATH / "hmc.json"
    hmc = HMCModule(
        burn=100,
        n_input=1,
        n_hidden=[50, 50],
        n_output=1,
        activation_fn=nn.ReLU(),
        store_hps=MODEL_PATH,
    )
    hmc.tune_hps(dm, 50)

    # MLP
    MODEL_PATH = HPS_PATH / "mlp_1.json"
    mlp_service = MLPService(
        n_input=1,
        n_hidden=[50, 50],
        n_output=1,
        loss_fn=nn.MSELoss(),
        activation_fn=nn.ReLU(),
        max_epochs=500,
        patience=20,
        store_hps=MODEL_PATH,
    )
    mlp_service.tune(temp_file, dm, n_trials=100)

    # BNN-VI
    MODEL_PATH = HPS_PATH / "bnn_vi.json"
    bnnvi_service = BNNVIService(
        n_input=1,
        n_hidden=[50, 50],
        n_output=1,
        activation_fn=nn.ReLU(),
        store_hps=MODEL_PATH,
    )
    bnnvi_service.tune(temp_file, dm, n_trials=100)

    # BNN-MFVI
    MODEL_PATH = HPS_PATH / "bnn_mfvi.json"
    bnnvi_service = BNNMFVIService(
        n_input=1,
        n_hidden=[50, 50],
        n_output=2,
        activation_fn=nn.ReLU(),
        store_hps=MODEL_PATH,
    )
    bnnvi_service.tune(temp_file, dm, n_trials=100)
