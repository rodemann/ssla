import tempfile
from pathlib import Path

import torch.nn as nn

from src.datamodules import UCIMLDatasetId, UCIMLRepoDataModule
from src.services import MLPService

EXPERIMENT_PATH = Path("./experiments/uciml")
HPS_PATH = EXPERIMENT_PATH / "hps"


if __name__ == "__main__":
    for dt in list(UCIMLDatasetId):
        temp_file = tempfile.mkdtemp()
        dataset_id = UCIMLDatasetId[dt.name]
        dm = UCIMLRepoDataModule(id=dataset_id, shift=False)

        mlp_service = MLPService(
            n_input=dm.X_train.shape[1],
            n_hidden=[50, 50],
            n_output=1,
            activation_fn=nn.ReLU(),
            store_hps=HPS_PATH / f"{dt.name.lower()}.json",
        )
        mlp_service.tune(temp_file, dm, n_trials=100)
