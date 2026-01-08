from enum import Enum

import torch
from lightning import LightningDataModule
from lightning_uq_box.datamodules.utils import collate_fn_tensordataset
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override
from ucimlrepo import fetch_ucirepo, list_available_datasets


class UCIMLDatasetId(int, Enum):
    AUTO_MPG = 9
    LIVER = 60
    CONCRETE = 165
    WINE = 186
    BIKESHARING = 275
    AIRFOIL = 291
    DAILY_DEMAND = 409
    REALESTATE = 477


class UCIMLRepoMixin:
    @staticmethod
    def list_available_datasets(
        filter: str | None = None, search: str | None = None, area: str | None = None
    ):
        list_available_datasets(filter, search, area)

    @staticmethod
    def _query_dataset(
        id: UCIMLDatasetId,
    ):
        return fetch_ucirepo(id=id.value)


class UCIMLRepoDataset(UCIMLRepoMixin):
    _id: UCIMLDatasetId

    def __init__(self, shift: bool = False):
        self.shift = shift

    def setup_shifted_dataset(
        self, dataset
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raise ValueError(
            f"{self.__class__.__name__} has no shifted dataset as of right now (You can follow https://proceedings.mlr.press/v51/chen16d.pdf)."
        )

    def setup_nonshifted_dataset(
        self, dataset
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.X_train = torch.tensor(dataset.data.features.values).float()
        self.Y_train = torch.tensor(dataset.data.targets.values).float()
        # Apply scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        self.X_train = torch.tensor(scaler_X.fit_transform(self.X_train)).float()
        self.Y_train = torch.tensor(scaler_y.fit_transform(self.Y_train)).float()
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_train, self.Y_train, test_size=0.2, random_state=42
        )
        return self.X_train, self.Y_train, self.X_test, self.Y_test

    def setup_dataset(self):
        dataset = self._query_dataset(self._id)
        if self.shift:
            return self.setup_shifted_dataset(dataset)
        else:
            return self.setup_nonshifted_dataset(dataset)


class UCIMLRepoWineDataset(UCIMLRepoDataset):
    _id = UCIMLDatasetId.WINE

    @override
    def setup_nonshifted_dataset(
        self, dataset
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wine_full = dataset.data.original
        wine_full.color = [0 if c == "red" else 1 for c in wine_full.color]
        target = wine_full.loc[:, "alcohol"]
        features = wine_full.drop(columns=["alcohol"])

        cat_features = features.loc[:, ["quality", "color"]]
        cont_features = features.drop(columns=["quality", "color"])

        encoder = OneHotEncoder(drop="first")
        cat_encoded = encoder.fit_transform(cat_features)

        # Apply scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        cont_features_scaled = scaler_X.fit_transform(cont_features)
        cont_features_tensor = torch.tensor(cont_features_scaled).float()
        cat_features_tensor = torch.tensor(cat_encoded.toarray()).float()

        Y_train = torch.tensor(target.values.reshape(-1, 1)).float()
        Y_train_scaled = torch.tensor(
            scaler_y.fit_transform(Y_train).reshape(-1, 1)
        ).float()

        X_train = torch.cat((cont_features_tensor, cat_features_tensor), dim=1)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X_train, Y_train_scaled, test_size=0.2, random_state=42
        )
        return self.X_train, self.Y_train, self.X_test, self.Y_test


class UCIMLRepoAirFoilDataset(UCIMLRepoDataset):
    _id = UCIMLDatasetId.AIRFOIL


class UCIMLRepoConcreteDataset(UCIMLRepoDataset):
    _id = UCIMLDatasetId.CONCRETE


class UCIMLRepoBikeSharingDataset(UCIMLRepoDataset):
    _id = UCIMLDatasetId.BIKESHARING

    @override
    def setup_nonshifted_dataset(
        self, dataset
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bike_orig = dataset.data.original
        bike_orig = bike_orig.drop(columns=["dteday", "instant"])

        target, features = bike_orig.loc[:, "temp"], bike_orig.drop(columns=["temp"])

        binary_features = features[["yr", "holiday", "workingday"]]
        cat_features = features[["season", "mnth", "hr", "weekday", "weathersit"]]
        discrete_features = features[["cnt", "registered", "casual"]]
        cont_features = features[["windspeed", "hum", "atemp"]]
        X_train, X_test, Y_train, Y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        encoder = OneHotEncoder(drop="first")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", scaler_X, cont_features.columns),
                ("cat", encoder, cat_features.columns),
            ]
        )

        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)

        Y_train_scaled = scaler_y.fit_transform(Y_train.values.reshape(-1, 1))
        Y_test_scaled = scaler_y.transform(Y_test.values.reshape(-1, 1))

        self.X_train = torch.tensor(X_train_processed.toarray()).float()
        self.Y_train = torch.tensor(Y_train_scaled).float()
        self.X_test = torch.tensor(X_test_processed.toarray()).float()
        self.Y_test = torch.tensor(Y_test_scaled).float()

        return self.X_train, self.Y_train, self.X_test, self.Y_test


class UCIMLRepoRealEstateDataset(UCIMLRepoDataset):
    _id = UCIMLDatasetId.REALESTATE


class UCIMLRepoAutoMPGDataset(UCIMLRepoDataset):
    _id = UCIMLDatasetId.AUTO_MPG

    @override
    def setup_nonshifted_dataset(
        self, dataset
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        auto_mpg = dataset.data.original.dropna()
        auto_mpg = auto_mpg.drop(columns=["car_name"])
        target = auto_mpg.loc[:, "mpg"]
        features = auto_mpg.drop(columns=["mpg"])

        cat_features = features.loc[:, ["origin"]]
        cont_features = features.drop(columns=["origin"])

        encoder = OneHotEncoder(drop="first")
        cat_encoded = encoder.fit_transform(cat_features)

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        cont_features_scaled = scaler_X.fit_transform(cont_features)

        cont_features_tensor = torch.tensor(cont_features_scaled, dtype=torch.float32)
        cat_features_tensor = torch.tensor(cat_encoded.toarray(), dtype=torch.float32)

        Y_train = torch.tensor(target.values.reshape(-1, 1), dtype=torch.float32)
        Y_train_scaled = torch.tensor(
            scaler_y.fit_transform(Y_train).reshape(-1, 1), dtype=torch.float32
        )

        X_train = torch.cat((cont_features_tensor, cat_features_tensor), dim=1)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X_train, Y_train_scaled, test_size=0.2, random_state=42
        )
        return self.X_train, self.Y_train, self.X_test, self.Y_test


class UCIMLRepoLiverDataset(UCIMLRepoDataset):
    _id = UCIMLDatasetId.LIVER

    @override
    def setup_nonshifted_dataset(
        self, dataset
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        liver_orig = dataset.data.original
        liver_orig = liver_orig.drop(columns=["selector"])
        target, features = liver_orig.loc[:, "alkphos"], liver_orig.drop(
            columns=["alkphos"]
        )

        disc_features, cont_features = features.loc[:, "drinks"], features.drop(
            columns=["drinks"]
        )

        cat_features = torch.tensor(disc_features.values).reshape(-1, 1).float()
        cont_features = torch.tensor(cont_features.values).float()

        # self.X_train = torch.tensor(features.values).float()
        self.Y_train = torch.tensor(target.values).reshape(-1, 1).float()

        # Apply scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        cont_features = torch.tensor(scaler_X.fit_transform(cont_features)).float()
        self.X_train = torch.cat((cont_features, cat_features), dim=1).float()
        self.Y_train = torch.tensor(scaler_y.fit_transform(self.Y_train)).float()

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            self.X_train, self.Y_train, test_size=0.2, random_state=42
        )
        return self.X_train, self.Y_train, self.X_test, self.Y_test


class UCIMLRepoDailyDemandDataset(UCIMLRepoDataset):
    _id: UCIMLDatasetId = UCIMLDatasetId.DAILY_DEMAND

    @override
    def setup_nonshifted_dataset(
        self, dataset
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        daily_demand = dataset.data.original
        target = daily_demand.loc[:, "Total orders"]
        features = daily_demand.drop(columns=["Total orders"])

        cat_features = ["Week of the month", "Day of the week"]
        cont_features = [
            "Non-urgent order",
            "Urgent order",
            "Order type A",
            "Order type B",
            "Order type C",
            "Fiscal sector orders",
            "Orders from the traffic controller sector",
            "Banking orders (1)",
            "Banking orders (2)",
            "Banking orders (3)",
        ]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(drop="first"), cat_features),
                ("cont", StandardScaler(), cont_features),
            ]
        )

        X = preprocessor.fit_transform(features)

        scaler_y = StandardScaler()
        Y = torch.tensor(
            scaler_y.fit_transform(target.values.reshape(-1, 1)), dtype=torch.float32
        )
        X = torch.tensor(X, dtype=torch.float32)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        return self.X_train, self.Y_train, self.X_test, self.Y_test


class UCIMLRepoDataModule(UCIMLRepoMixin, LightningDataModule):
    """Implement UCIMLRepo DataModule."""

    def __init__(
        self,
        id: UCIMLDatasetId = UCIMLDatasetId.WINE,
        shift: bool = False,
        batch_size: int = 200,
    ) -> None:
        """Create the respective datamodule

        Args:
            id: id of the uciml repo dataset. Use UCIMLRepoDataModule.list_available_datasets to get a list of datasets
        """
        super().__init__()
        torch.manual_seed(42)
        try:
            klass = next(
                subclass
                for subclass in UCIMLRepoDataset.__subclasses__()
                if subclass._id == id
            )
        except StopIteration as e:
            raise ValueError(
                f"Please provide a proper UCIMLDatasetId. You provided: {id}"
            ) from e
        inst = klass(shift=shift)

        self.X_train, self.Y_train, self.X_test, self.Y_test = inst.setup_dataset()
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(
            self.X_train, self.Y_train, test_size=0.2, random_state=42
        )
        self.X_val, self.X_cal, self.Y_val, self.Y_cal = train_test_split(
            self.X_val, self.Y_val, test_size=0.4, random_state=42
        )

        self.batch_size = batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""
        return DataLoader(
            TensorDataset(self.X_train, self.Y_train),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn_tensordataset,
        )

    def val_dataloader(self) -> DataLoader:
        """Return val dataloader."""
        return DataLoader(
            TensorDataset(self.X_val, self.Y_val),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )

    def calibration_dataloader(self) -> DataLoader:
        """Return calibration dataloader."""
        return DataLoader(
            TensorDataset(self.X_cal, self.Y_cal),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            TensorDataset(self.X_test, self.Y_test),
            batch_size=self.batch_size,
            collate_fn=collate_fn_tensordataset,
        )
