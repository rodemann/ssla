import abc

import torch


class SamplingStrategy(abc.ABC):
    def __init__(self, n: int = 20):
        self.n = n

    @abc.abstractmethod
    def sample(
        self,
        x_pred: torch.Tensor,
        y_pred: torch.Tensor,
        y_train: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Method not implemented")


class NeighborhoodSampling(SamplingStrategy):
    def __init__(self, n: int = 20, scale: float = 1.96):
        super().__init__(n)
        self.scale = scale

    def sample(
        self,
        x_pred: torch.Tensor,
        y_pred: torch.Tensor,
        y_train: torch.Tensor,
    ):
        y_std = y_train.std(unbiased=False)

        scale = self.scale * y_std
        y_offsets = torch.linspace(-scale, scale, self.n, device=y_pred.device)  # [n]

        y_new_values = y_pred.unsqueeze(1) + y_offsets.unsqueeze(0).unsqueeze(
            -1
        )  # [obs, n, 1]
        x_expanded = x_pred.unsqueeze(1).expand(-1, self.n, -1)  # [obs, n, c]

        x_flat = x_expanded.reshape(-1, x_pred.shape[1])  # [obs * n, c]
        y_flat = y_new_values.reshape(-1, 1)  # [obs * n, 1]
        return y_new_values, x_flat, y_flat


class ClassificationSampling(SamplingStrategy): ...
