import logging
from typing import Literal, Type, Union

import torch
import tqdm
from lightning_uq_box.uq_methods.base import DeterministicModel
from nngeometry.object import PMatDense, PMatDiag, PMatKFAC, PMatLowRank

from src.ssl.ApproximationStrategy import (
    ApproximationStrategy,
    SSLAApproximationStrategy,
)
from src.ssl.SamplingStrategy import NeighborhoodSampling, SamplingStrategy


class SelfSupervisedLaplace:
    """
    Implementation of the Self-Supervised Laplace Approximation (SSLA)
    for posterior predictive distributions (PPD).

    This class leverages deterministic models to estimate predictive uncertainty
    by approximating the posterior predictive distribution. The Fisher Information
    Matrix (FIM) is used to efficiently capture model uncertainty.

    The code is based on Lightning-uq-box (https://lightning-uq-box.readthedocs.io/en/latest/index.html)
    """

    def __init__(
        self,
        model: DeterministicModel,
        fim_represetation: Union[
            Type[PMatKFAC], Type[PMatDense], Type[PMatLowRank], Type[PMatDiag]
        ] = PMatKFAC,
        task: Literal["regression", "classification"] = "regression",
        n_outputs: int = 1,
    ):
        """
        Initializes the Self-Supervised Laplace class.

        Args:
            model (DeterministicModel): The base deterministic model used for Laplace approximation.
            fim_represetation (Union[Type[PMatKFAC], Type[PMatDense], Type[PMatLowRank], Type[PMatDiag]]):
                Type of Fisher Information Matrix representation. Defaults to `PMatKFAC`.
            n_outputs (int): Number of outputs from the model. Defaults to 1.
        """
        self.fim_representation = fim_represetation
        self.task = task
        self.n_outputs = n_outputs
        self.base_model = model
        self.loss_fn = model.loss_fn

    def log_ppd(
        self,
        x_pred: torch.Tensor,
        y_pred: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        approximation_strategy: ApproximationStrategy,
    ) -> torch.Tensor:
        # x_pred of size [n, k] where n is the number of observations, k is the number of covariates
        batch_size = x_pred.size(0)
        if isinstance(approximation_strategy, SSLAApproximationStrategy):
            logging.info("SSLA does not support batches. Falling back to looping")
            return torch.tensor(
                list(
                    map(
                        lambda xy: approximation_strategy.log_ppd(
                            xy[0].unsqueeze(0),
                            xy[1].unsqueeze(0),
                            X_train,
                            y_train,
                            self.base_model,
                            self.loss_fn,
                            self.fim_representation,
                            self.n_outputs,
                        ),
                        tqdm.tqdm(zip(x_pred, y_pred), total=x_pred.size(0)),
                    )
                )
            )
        else:
            return approximation_strategy.log_ppd(
                x_pred,
                y_pred,
                X_train,
                y_train,
                self.base_model,
                self.loss_fn,
                self.fim_representation,
                self.n_outputs,
            )

    def sample_log_ppd(
        self,
        x_pred: torch.Tensor,
        y_pred: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        approximation_strategy: ApproximationStrategy,
        sampling_strategy: SamplingStrategy = NeighborhoodSampling(n=20, scale=1.96),
    ):
        y_new_values, x_flat, y_flat = sampling_strategy.sample(x_pred, y_pred, y_train)

        dist_flat = self.log_ppd(
            x_flat, y_flat, X_train, y_train, approximation_strategy
        )  # [obs * n]

        dist = dist_flat.view(x_pred.shape[0], sampling_strategy.n)  # [obs, n]

        dists = [
            {
                "x": x_pred[i].detach().cpu(),
                "y": y_pred[i].detach().cpu(),
                "dist": [
                    {"y": y_new_values[i, j, 0], "log_ppd": dist[i, j]}
                    for j in range(sampling_strategy.n)
                ],
            }
            for i in range(x_pred.shape[0])
        ]

        return dists
