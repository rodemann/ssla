import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import TensorDataset


def subsample_datamodule(dm: LightningDataModule, subset_size: int = 50):
    """
    Dynamically create a test DataLoader with subsampling if subset_size is provided.

    Args:
        dm (LightningDataModule): The fully initialized DataModule containing X_test and Y_test.
        subset_size (Optional[int]): Number of samples to use from the test dataset.

    Returns:
        DataLoader for either the full or subsampled test dataset.
    """
    num_obs = dm.X_test.shape[0]

    n_samples = min(subset_size, num_obs)
    test_dataset = TensorDataset(dm.X_test, dm.Y_test)

    subset_indices = torch.randperm(len(test_dataset))[:n_samples]
    dm.X_test = dm.X_test[subset_indices]
    dm.Y_test = dm.Y_test[subset_indices]
