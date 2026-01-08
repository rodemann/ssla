import tempfile
from pathlib import Path

from laplace import HessianStructure, SubsetOfWeights
from lightning.pytorch import seed_everything
from lightning_uq_box.datamodules import ToyHeteroscedasticDatamodule
from nngeometry.object import PMatDense, PMatDiag, PMatKFAC

from experiments.heteroscedastic_regression.HMC import HMCModule
from experiments.heteroscedastic_regression.utils import (
    coverage_to_tex_table,
    visualize,
)
from src.services import BNNMFVIService, BNNVIService, MLPService
from src.utils import assla, compute_kde_from_ssla, la, ssla, ssla_dists_to_l_uq_box

EXPERIMENT_PATH = Path("./experiments/heteroscedastic_regression")
GRAPHICS_PATH = EXPERIMENT_PATH / "graphics"
LATEX_PATH = EXPERIMENT_PATH / "latex"
HPS_PATH = EXPERIMENT_PATH / "hps"


def hmc(dm):
    hmc_mod = HMCModule(
        tau_out=50, step_size=0.003, num_samples=3000, num_steps_per_sample=20, burn=300
    )
    hmc_mod.load_hps(HPS_PATH / "hmc.json")
    hmc_mod.train_hmc(dm)

    preds = hmc_mod.predict_hmc(dm.X_test, dm.Y_test)
    return hmc_mod.hmc_preds_to_lightning_uq_box_preds(preds[0])


def mfvi(
    temp_dir,
    dm,
):
    bbp_service = BNNMFVIService()
    bbp_service.load_hps(HPS_PATH / "bnn_mfvi.json")
    bbp_service.train(dm, temp_dir)

    preds = bbp_service.model.predict_step(dm.X_test)
    return preds


def vi(temp_dir, dm):
    bbp_service = BNNVIService()
    bbp_service.load_hps(HPS_PATH / "bnn_vi.json")
    bbp_service.train(dm, temp_dir)

    preds = bbp_service.model.predict_step(dm.X_test)
    return preds


def perform_experiment():
    seed_everything(42)

    temp_file = tempfile.mkdtemp()
    dm = ToyHeteroscedasticDatamodule(n_points=250, test_fraction=0.1)

    mlp_service = MLPService()
    mlp_service.load_hps(HPS_PATH / "mlp_1.json")
    mlp_service.train(dm, temp_file)
    model = mlp_service.model

    assla_kfac_dists = assla(model=model, dm=dm, fim_represetation=PMatKFAC)
    assla_dense_dists = assla(model=model, dm=dm, fim_represetation=PMatDense)
    assla_diag_dists = assla(model=model, dm=dm, fim_represetation=PMatDiag)

    ssla_kfac_dists = ssla(model=model, dm=dm, fim_represetation=PMatKFAC)
    ssla_dense_dists = ssla(model=model, dm=dm, fim_represetation=PMatDense)
    ssla_diag_dists = ssla(model=model, dm=dm, fim_represetation=PMatDiag)

    # LA ['last_layer_full', 'last_layer_diag', 'last_layer_kron']

    la_dense = la(
        model=model,
        dm=dm,
        temp_file=temp_file,
        subset_of_weights=SubsetOfWeights.LAST_LAYER,
        hessian_structure=HessianStructure.FULL,
    )
    la_diag = la(
        model=model,
        dm=dm,
        temp_file=temp_file,
        subset_of_weights=SubsetOfWeights.LAST_LAYER,
        hessian_structure=HessianStructure.DIAG,
    )
    la_kfac = la(
        model=model,
        dm=dm,
        temp_file=temp_file,
        subset_of_weights=SubsetOfWeights.LAST_LAYER,
        hessian_structure=HessianStructure.KRON,
    )

    # hmc
    hmc_pred = hmc(dm)
    # vi
    vi_pred = vi(temp_file, dm)

    # mfvi
    mfvi_pred = mfvi(temp_file, dm)

    ssla_kfac = ssla_dists_to_l_uq_box(compute_kde_from_ssla(ssla_kfac_dists))
    ssla_diag = ssla_dists_to_l_uq_box(compute_kde_from_ssla(ssla_diag_dists))
    ssla_dense = ssla_dists_to_l_uq_box(compute_kde_from_ssla(ssla_dense_dists))

    assla_kfac = ssla_dists_to_l_uq_box(compute_kde_from_ssla(assla_kfac_dists))
    assla_diag = ssla_dists_to_l_uq_box(compute_kde_from_ssla(assla_diag_dists))
    assla_dense = ssla_dists_to_l_uq_box(compute_kde_from_ssla(assla_dense_dists))

    visualize(
        dm,
        ssla_kfac,
        ssla_diag,
        ssla_dense,
        assla_kfac,
        assla_diag,
        assla_dense,
        la_kfac,
        la_diag,
        la_dense,
        vi_pred,
        mfvi_pred,
        hmc_pred,
        target=GRAPHICS_PATH / "heteroscedastic_plot.png",
    )

    coverage_to_tex_table(
        dm,
        ssla_kfac,
        ssla_diag,
        ssla_dense,
        assla_kfac,
        assla_diag,
        assla_dense,
        la_kfac,
        la_diag,
        la_dense,
        vi_pred,
        mfvi_pred,
        hmc_pred,
        target=LATEX_PATH / "heteroscedastic_coverage.tex",
    )


if __name__ == "__main__":
    perform_experiment()
