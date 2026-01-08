import tempfile
from pathlib import Path

import uncertainty_toolbox as uct
from laplace import HessianStructure, Likelihood, SubsetOfWeights

from experiments.uciml.utils import generate_table
from src.datamodules import UCIMLDatasetId, UCIMLRepoDataModule, subsample_datamodule
from src.services import MLPService
from src.utils import (
    CI,
    assla,
    compute_kde_from_ssla,
    coverage,
    la,
    ssla,
    ssla_dists_to_l_uq_box,
)

EXPERIMENT_PATH = Path("./experiments/uciml")
HPS_PATH = EXPERIMENT_PATH / "hps"


def metrics(dm, pred):
    cov_95 = coverage(pred, dm, CI.NINTY_FIVE)
    cov_75 = coverage(pred, dm, CI.SEVENTY_FIVE)
    cov_50 = coverage(pred, dm, CI.FIFTY)
    crps = uct.metrics_scoring_rule.crps_gaussian(
        y_pred=pred["pred"].flatten().numpy(),
        y_std=pred["pred_uct"].flatten().numpy(),
        y_true=dm.Y_test.flatten().numpy(),
    )
    nll = uct.metrics_scoring_rule.nll_gaussian(
        y_pred=pred["pred"].flatten().numpy(),
        y_std=pred["pred_uct"].flatten().numpy(),
        y_true=dm.Y_test.flatten().numpy(),
    )

    return {"95": cov_95, "75": cov_75, "50": cov_50, "nll": nll, "crps": crps}


def run_experiment():
    res = []
    for dt in UCIMLDatasetId:
        # if dt == UCIMLDatasetId.REALESTATE:
        #     continue
        temp_file = tempfile.mkdtemp()
        run_ssla = True
        dm = UCIMLRepoDataModule(id=dt, shift=False)
        if (
            dt.name == "CONCRETE"
            or dt.name == "WINE"
            or dt.name == "BIKESHARING"
            or dt.name == "AIRFOIL"
        ):
            subsample_datamodule(dm, 50)
            # run_ssla = False
            assert dm.X_test.shape[0] == 50
        mlp_service = MLPService(n_input=dm.X_train.shape[1])
        mlp_service.load_hps(HPS_PATH / f"{dt.name.lower()}.json")
        dm.batch_size = mlp_service.batch_size
        mlp_service.train(dm, temp_file)
        model = mlp_service.model
        assla_dict = {}
        assla_dists = assla(model, dm)
        assla_dict.update(
            metrics(dm, ssla_dists_to_l_uq_box(compute_kde_from_ssla(assla_dists)))
        )
        la_dict = {}
        la_pred = la(
            model,
            dm,
            temp_file,
            likelihood=Likelihood.REGRESSION,
            subset_of_weights=SubsetOfWeights.LAST_LAYER,
            hessian_structure=HessianStructure.KRON,
        )
        la_dict.update(metrics(dm, la_pred))
        ssla_dict = None
        if run_ssla:
            ssla_dict = {}
            ssla_dists = ssla(model, dm)
            ssla_dict.update(
                metrics(dm, ssla_dists_to_l_uq_box(compute_kde_from_ssla(ssla_dists)))
            )

        res.append(
            {
                "name": dt.name,
                "ASSLA": assla_dict,
                "LA": la_dict,
                "SSLA": ssla_dict,
            }
        )

    generate_table(res, target=Path("./experiments/uciml/latex/uciml_results.tex"))


if __name__ == "__main__":
    run_experiment()
