import abc
import logging
from copy import deepcopy
from typing import Type, Union

import lightning as L
import torch
import torch.nn as nn
import torch.utils.data as data
from lightning.pytorch.callbacks import EarlyStopping
from lightning_uq_box.datamodules.utils import collate_fn_tensordataset
from lightning_uq_box.uq_methods import DeterministicModel
from nngeometry.metrics import FIM
from nngeometry.object import PMatDense, PMatDiag, PMatKFAC, PMatLowRank


class ApproximationStrategy(abc.ABC):
    """
    Abstract base class for various approximation strategies.
    Provides common utility methods for Fisher Information Matrix (FIM) computation,
    log-likelihood calculations, and weight extraction from neural networks.
    """

    def __init__(
        self,
        # track_states: bool = False, ignore the states stuff for right now
    ):
        """
        Initialize the ApproximationStrategy class.
        """
        self.llh = None
        self.log_prior = None
        self.fisher = None
        # self.track_states = track_states
        # self.states = []
        self._disable_pytorch_lightning_logs()

    def _disable_pytorch_lightning_logs(self):
        """
        Suppress PyTorch Lightning log messages containing specific keywords.
        """

        def filter(record):
            keywords = ["available:", "GPU", "HPU", "TPU", "reached."]
            return not any(keyword in record.getMessage() for keyword in keywords)

        logging.getLogger().addFilter(filter)
        logging.getLogger("lightning.pytorch.utilities.rank_zero").addFilter(filter)
        # logger = logging.getLogger("pytorch_lightning.utilities.rank_zero")
        # logger.setLevel(logging.ERROR)

    def _get_weights(self, network: nn.Module):
        """
        Retrieve flattened model weights.

        Args:
            network (nn.Module): The neural network model.

        Returns:
            torch.Tensor: Flattened weights of the model.
        """
        return torch.cat([w.detach().view(-1) for w in network.parameters()])

    def _get_fim(
        self,
        network: nn.Module,
        train_data: data.TensorDataset,
        fim_representation: Union[
            Type[PMatKFAC], Type[PMatDense], Type[PMatLowRank], Type[PMatDiag]
        ] = PMatKFAC,
        n_outputs: int = 1,
    ):
        """
        Compute the Fisher Information Matrix (FIM) for the given network and dataset. If the computation of the eigendecomposition for representations KFAC, Dense and LowRank fail, we fall back to Diag

        Args:
            network (nn.Module): The neural network model.
            train_data (data.TensorDataset): Training data for FIM computation.
            fim_representation (Union[Type[PMatKFAC], Type[PMatDense], Type[PMatLowRank], Type[PMatDiag]]):
                Type of Fisher Information Matrix representation. Defaults to PMatKFAC.
            n_outputs (int): Number of model outputs. Defaults to 1.

        Returns:
            torch.Tensor: Representation of the FIM.
        """
        fim_obj = FIM(
            model=network,
            loader=data.DataLoader(train_data),
            representation=fim_representation,
            n_output=n_outputs,
            variant="regression",
        )
        if not isinstance(fim_obj, PMatDiag):
            try:
                fim_obj.compute_eigendecomposition()
            except Exception as e:
                print("Failed to compute eigendecomposition. Falling back to DIAG")
                fim_obj = FIM(
                    model=network,
                    loader=data.DataLoader(train_data),
                    representation=PMatDiag,
                    n_output=n_outputs,
                    variant="regression",
                )
        return fim_obj.get_dense_tensor()

    def _get_llh(
        self, model: nn.Module, loss_fn: nn.Module, X: torch.Tensor, y: torch.Tensor
    ):
        """
        Compute the log-likelihood of the model given data.

        Args:
            model (nn.Module): The neural network model.
            loss_fn (nn.Module): Loss function to use.
            X (torch.Tensor): Input data.
            y (torch.Tensor): Target data.

        Returns:
            torch.Tensor: Log-likelihood of the data.
        """
        preds = model(X)
        return -loss_fn(preds.detach(), y)

    def _get_llh_per_observation(
        self, model: nn.Module, loss_fn: nn.Module, X: torch.Tensor, y: torch.Tensor
    ):
        """
        Compute the log-likelihood per observation of the model given data.

        Args:
            model (nn.Module): The neural network model.
            loss_fn (nn.Module): Loss function to use.
            X (torch.Tensor): Input data.
            y (torch.Tensor): Target data.

        Returns:
            torch.Tensor: Log-likelihood per observation of the data.
        """
        loss_reduction = loss_fn.reduction
        loss_fn.reduction = "none"  # type: ignore
        preds = model(X)
        loss = -loss_fn(preds.detach(), y)
        loss_fn.reduction = loss_reduction  # "mean"
        return loss

    @abc.abstractmethod
    def log_ppd(
        self,
        x_pred: torch.Tensor,
        y_pred: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        base_model: DeterministicModel,
        loss_fn: nn.Module,
        fim_representation: (
            type[PMatKFAC] | type[PMatDense] | type[PMatLowRank] | type[PMatDiag]
        ) = PMatKFAC,
        n_outputs: int = 1,
    ) -> torch.Tensor:
        """
        Compute the log posterior predictive density (log PPD).
        """
        raise NotImplementedError("Not implemented.")


class SSLAApproximationStrategy(ApproximationStrategy):
    """
    Self-Supervised Laplace Approximation.
    """

    def __init__(self, prior: torch.distributions.Distribution):
        """
        Initialize the SSLA approximation strategy.

        Args:
            prior (torch.distributions.Distribution): The prior distribution.
        """
        super().__init__()
        self.prior = prior

    def _get_trainer(self):
        """
        Get the PyTorch Lightning trainer for model training.

        Returns:
            L.Trainer: Lightning trainer object.
        """
        return L.Trainer(
            max_epochs=10,
            logger=False,
            callbacks=[EarlyStopping(monitor="train_loss")],
            enable_progress_bar=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            deterministic=True,
        )

    def _ssl_approx(
        self,
        llh: torch.Tensor,
        llh_np1: torch.Tensor,
        log_prior: torch.Tensor,
        log_prior_np1: torch.Tensor,
        fisher: torch.Tensor,
        fisher_np1: torch.Tensor,
    ):
        """
        Compute the SSLA approximation.

        Args:
            llh (torch.Tensor): Log-likelihood of the original training data.
            llh_np1 (torch.Tensor): Log-likelihood of the augmented training data.
            log_prior (torch.Tensor): Log-prior of the original training data.
            log_prior_np1 (torch.Tensor): Log-prior of the augmented training data.
            fisher (torch.Tensor): FIM of the original training data.
            fisher_np1 (torch.Tensor): FIM of the augmented training data.

        Returns:
            torch.Tensor: Computed SSL approximation.
        """

        delta_llh = llh_np1 - llh
        delta_prior = log_prior_np1 - log_prior
        delta_fisher = 0.5 * torch.log(torch.det(fisher) + 1e-6) - 0.5 * torch.log(
            torch.det(fisher_np1) + 1e-6
        )
        log_posteriorpredictive = delta_llh + delta_prior + delta_fisher
        return log_posteriorpredictive

    def log_ppd(
        self,
        x_pred: torch.Tensor,
        y_pred: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        base_model: DeterministicModel,
        loss_fn: nn.Module,
        fim_representation: (
            type[PMatKFAC] | type[PMatDense] | type[PMatLowRank] | type[PMatDiag]
        ) = PMatKFAC,
        n_outputs: int = 1,
    ):
        """Compute the log posterior predictive density for a single observation $(\hat y_{n+1}, x_{n+1})$

        Args:
            x_pred (torch.Tensor): The n+1-th observation (aka. the new observation)
            y_pred (torch.Tensor): The prediction of $f(x_{n+1})$
            X_train (torch.Tensor): Train features
            y_train (torch.Tensor): Train target
            base_model (DeterministicModel): the basemodel to use
            loss_fn (nn.Module): The loss function applied during training the base model_
            fim_representation (type[PMatKFAC]  |  type[PMatDense]  |  type[PMatLowRank]  |  type[PMatDiag], optional): The representation of the covariance. Defaults to PMatKFAC.
            n_outputs (int, optional): the number of outputs, i.e. targets. For regression this is typically 1. Defaults to 1.

        Returns:
            torch.Tensor: The log ppd of $\hat y_{n+1}$ given data
        """
        try:
            batch_size, covariates_size = x_pred.size(0), x_pred.size(1)
        except IndexError as e:
            raise ValueError(
                f"Expected input of shape [1, 1] where 1 is the number of observations, 1 the number of covariates, but got: {x_pred.shape}"
            ) from e
        try:
            batch_size, target_size = y_pred.size(0), y_pred.size(1)
            if target_size > 1:
                raise ValueError("Currently only supporting regression cases")
        except IndexError as e:
            raise ValueError(
                f"Expected input of shape [1, 1] where 1 is the number of observations, 1 the number of targets, but got: {y_pred.shape}"
            ) from e

        if self.llh is None:
            self.llh = self._get_llh(base_model, loss_fn, X_train, y_train)
        if self.fisher is None:
            self.fisher = self._get_fim(
                base_model.model,
                data.TensorDataset(X_train, y_train),
                fim_representation,
                n_outputs,
            )
        self.log_prior = self.prior.log_prob(self._get_weights(base_model.model))

        augmented_X = torch.cat((X_train, x_pred))
        augmented_y = torch.cat((y_train, y_pred))
        augmented_ds = data.TensorDataset(augmented_X, augmented_y)
        _base_model = deepcopy(base_model)
        trainer = self._get_trainer()
        trainer.fit(
            _base_model,
            train_dataloaders=data.DataLoader(
                augmented_ds,
                batch_size=32,
                shuffle=True,
                collate_fn=collate_fn_tensordataset,
            ),
        )
        llh_D = self._get_llh(_base_model, loss_fn, X_train, y_train)
        llh_np1 = self._get_llh(_base_model, loss_fn, x_pred, y_pred)
        tilde_llh = llh_np1 + llh_D
        log_tilde_prior = self.prior.log_prob(self._get_weights(_base_model.model))
        fisher_D = self._get_fim(
            _base_model.model,
            data.TensorDataset(X_train, y_train),
            fim_representation,
            n_outputs,
        )
        fisher_np1 = self._get_fim(
            _base_model.model,
            data.TensorDataset(x_pred, y_pred),
            fim_representation,
            n_outputs,
        )
        tilde_fisher = fisher_D + fisher_np1

        return self._ssl_approx(
            self.llh,
            tilde_llh,
            self.log_prior,
            log_tilde_prior,
            self.fisher,
            tilde_fisher,
        )


class ASSLAApproximationStrategy(ApproximationStrategy):
    """
    Approximate Self-Supervised Laplace Approximation
    """

    def log_ppd(
        self,
        x_pred: torch.Tensor,
        y_pred: torch.Tensor,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        base_model: DeterministicModel,
        loss_fn: nn.Module,
        fim_representation: (
            type[PMatKFAC] | type[PMatDense] | type[PMatLowRank] | type[PMatDiag]
        ) = PMatKFAC,
        n_outputs: int = 1,
    ):
        """
        Computes the log posterior predictive for a single observation using ASSLA.

        Args:
            x_pred (torch.Tensor): Features of the prediction point with shape [k].
            y_pred (torch.Tensor): Target value for the prediction point with shape [1].
            X_train (torch.Tensor): Training feature set with shape [m, k].
            y_train (torch.Tensor): Training target set with shape [m].
            base_model (DeterministicModel): The base model used for likelihood and Fisher computation.
            loss_fn (nn.Module): Loss function to compute the negative log-likelihood.
            fim_representation (type): Type of Fisher matrix representation. Defaults to `PMatKFAC`.
            n_outputs (int): Number of outputs from the model. Defaults to 1.

        Returns:
            torch.Tensor: Log posterior predictive for the given observation.
        """
        try:
            _, _ = x_pred.size(0), x_pred.size(1)
        except IndexError as e:
            raise ValueError(
                f"Expected input of shape [n, k] where n is the number of observations, k the number of covariates, but got: {x_pred.shape}"
            ) from e
        try:
            _, target_size = y_pred.size(0), y_pred.size(1)
            if target_size > 1:
                raise ValueError("Currently only supporting regression cases")
        except IndexError as e:
            raise ValueError(
                f"Expected input of shape [n, 1] where n is the number of observations, 1 the number of targets, but got: {y_pred.shape}"
            ) from e

        if self.llh is None:
            self.llh = self._get_llh(base_model, loss_fn, X_train, y_train)
        if self.fisher is None:
            self.fisher = self._get_fim(
                base_model.model,
                data.TensorDataset(X_train, y_train),
                fim_representation,
                n_outputs,
            )

        llh_np1 = self._get_llh_per_observation(
            base_model.model, loss_fn, x_pred, y_pred
        )

        fisher_np1 = self._get_fim(
            base_model.model,
            data.TensorDataset(x_pred, y_pred),
            fim_representation,
            n_outputs,
        )

        _, logdet_fisher = torch.slogdet(
            self.fisher
            + 1e-6 * torch.eye(self.fisher.shape[0], device=self.fisher.device)
        )
        _, logdet_combined = torch.slogdet(
            1 / x_pred.numel() * fisher_np1
            + self.fisher
            + 1e-6 * torch.eye(self.fisher.shape[0], device=self.fisher.device)
        )

        delta_fisher = 0.5 * (logdet_fisher - logdet_combined)

        return llh_np1 + delta_fisher
