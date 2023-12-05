import sys

sys.path.append("/home/ray/workspace/proj_fcst/")
from glob import glob

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import boxcox
import lightgbm as lgb
from sklearn.linear_model import Lasso

data = pd.read_parquet("/dat/yuwang/project/data_rank.parquet")
df_train = data.query('date < "2021-01-01"')
df_test = data.query('date >= "2021-01-01"')
train_date = df_train["datetime"]
test_date = df_test["datetime"]


target_col = "y_1"


class ModelMetrics:
    def __init__(self, y_hat, y, timestamps):
        if hasattr(y_hat, "values"):
            y_hat = y_hat.values
        if hasattr(y, "values"):
            y = y.values
        if hasattr(timestamps, "values"):
            timestamps = timestamps.values
        self.y_hat = y_hat
        self.y = y
        self.timestamps = timestamps
        self.mean_ic = 0
        self.mean_rt9 = 0
        self.mean_ac9 = 0
        self.ic_values = []
        self.rt9s = []
        self.ac9s = []
        self.metrics = {}

    def calculate_ic(self):
        ic_values = []

        for timestamp in np.unique(self.timestamps):
            mask = self.timestamps == timestamp
            y_hat_t = self.y_hat[mask]
            y_t = self.y[mask]

            ic = np.corrcoef(y_hat_t, y_t)[0, 1]
            ic_values.append(ic)
        self.ic_values = ic_values
        self.mean_ic = np.mean(ic_values)
        return self.mean_ic

    def calculate_top_10_percent_returns(self):
        top_10_percent_returns = []

        for timestamp in np.unique(self.timestamps):
            mask = self.timestamps == timestamp
            y_hat_t = self.y_hat[mask]
            y_t = self.y[mask]

            top_10_percent = int(len(y_hat_t) * 0.1)
            top_10_percent_indices = np.argsort(y_hat_t)[-top_10_percent:]
            top_10_percent_returns.append(np.mean(y_t[top_10_percent_indices]))
        self.rt9s = top_10_percent_returns
        self.mean_rt9 = np.mean(top_10_percent_returns)
        return self.mean_rt9

    def calculate_top_10_percent_accuracy(self):
        top_10_percent_accuracies = []

        for timestamp in np.unique(self.timestamps):
            mask = self.timestamps == timestamp
            y_hat_t = self.y_hat[mask]
            y_t = self.y[mask]

            top_10_percent = int(len(y_hat_t) * 0.1)
            top_10_percent_y_hat_indices = np.argsort(y_hat_t)[-top_10_percent:]
            top_10_percent_y_indices = np.argsort(y_t)[-top_10_percent:]

            accuracy = (
                len(set(top_10_percent_y_hat_indices) & set(top_10_percent_y_indices))
                / top_10_percent
            )
            top_10_percent_accuracies.append(accuracy)
        self.ac9s = top_10_percent_accuracies
        self.mean_ac9 = np.mean(top_10_percent_accuracies)
        return self.mean_ac9

    def make(self):
        self.mean_ic = self.calculate_ic()
        self.mean_rt9 = self.calculate_top_10_percent_returns()
        self.mean_ac9 = self.calculate_top_10_percent_accuracy()

    def evaluate(self):
        self.make()
        print(f"Information Coefficient: {self.mean_ic:.5f}")
        print(f"Average Returns of Top 10%: {self.mean_rt9:.5f}")
        print(f"Accuracy of Top 10%: {self.mean_ac9:.5f}")
        self.metrics = pd.DataFrame(
            {
                "ic": [self.mean_ic],
                "rt9": [self.mean_rt9],
                "ac9": [self.mean_ac9],
            }
        )


# @jit(nopython=True)
# def calculate_ic(y_hat:np.ndarray, y, timestamps:np.ndarray, unique_timestamps:np.ndarray):
#     ic_values = np.empty(unique_timestamps.shape[0], np.float64)

#     for i, timestamp in enumerate(unique_timestamps):
#         mask = timestamps == timestamp
#         y_hat_t = y_hat[mask]
#         y_t = y[mask]

#         ic = np.corrcoef(y_hat_t, y_t)
#         ic_values[i] = ic

#     return ic_values


# @jit(nopython=True)
# def calculate_top_10_percent_returns(y_hat, y, timestamps, unique_timestamps):
#     top_10_percent_returns = np.empty(unique_timestamps.shape[0], np.float64)

#     for i, timestamp in enumerate(unique_timestamps):
#         mask = timestamps == timestamp
#         y_hat_t = y_hat[mask]
#         y_t = y[mask]

#         top_10_percent = int(len(y_hat_t) * 0.1)
#         top_10_percent_indices = np.argsort(y_hat_t)[-top_10_percent:]
#         top_10_percent_returns[i] = np.mean(y_t[top_10_percent_indices])

#     return top_10_percent_returns


# @jit(nopython=True)
# def calculate_top_10_percent_accuracy(y_hat, y, timestamps, unique_timestamps):
#     top_10_percent_accuracies = np.empty(unique_timestamps.shape[0], np.float64)

#     for i, timestamp in enumerate(unique_timestamps):
#         mask = timestamps == timestamp
#         y_hat_t = y_hat[mask]
#         y_t = y[mask]

#         top_10_percent = int(len(y_hat_t) * 0.1)
#         top_10_percent_y_hat_indices = np.argsort(y_hat_t)[-top_10_percent:]
#         top_10_percent_y_indices = np.argsort(y_t)[-top_10_percent:]

#         accuracy = (
#             len(set(top_10_percent_y_hat_indices) & set(top_10_percent_y_indices))
#             / top_10_percent
#         )
#         top_10_percent_accuracies[i] = accuracy

#     return top_10_percent_accuracies


# class ModelMetrics:
#     def __init__(self, y_hat, y, timestamps):
#         if hasattr(y_hat, "values"):
#             y_hat = y_hat.values
#         if hasattr(y, "values"):
#             y = y.values
#         if hasattr(timestamps, "values"):
#             timestamps = List(timestamps.values)
#         self.y_hat = y_hat
#         self.y = y
#         self.timestamps = timestamps
#         self.mean_ic = 0
#         self.mean_rt9 = 0
#         self.mean_ac9 = 0
#         self.ic_values = []
#         self.rt9s = []
#         self.ac9s = []
#         self.metrics = {}

#     def make(self):
#         unique_timestamps = np.unique(self.timestamps)
#         self.ic_values = calculate_ic(
#             self.y_hat, self.y, self.timestamps, unique_timestamps
#         )
#         self.mean_ic = np.mean(self.ic_values)

#         self.rt9s = calculate_top_10_percent_returns(
#             self.y_hat, self.y, self.timestamps, unique_timestamps
#         )
#         self.mean_rt9 = np.mean(self.rt9s)

#         self.ac9s = calculate_top_10_percent_accuracy(
#             self.y_hat, self.y, self.timestamps, unique_timestamps
#         )
#         self.mean_ac9 = np.mean(self.ac9s)


def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, "w") as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))


def read_yaml_to_dict(
    yaml_path: str,
):
    with open(yaml_path) as file:
        dict_value = yaml.load(file.read(), Loader=yaml.FullLoader)
        return dict_value


from sklearn.model_selection import train_test_split


class FinanceDataModule(pl.LightningDataModule):
    def __init__(self, X_train, X_test, y_train, y_test, batch_size=64):
        super().__init__()
        self.X_train = X_train.values if hasattr(X_train, "values") else X_train
        self.X_test = X_test.values if hasattr(X_test, "values") else X_test
        self.y_train = y_train.values if hasattr(y_train, "values") else y_train
        self.y_test = y_test.values if hasattr(y_test, "values") else y_test

        self.batch_size = batch_size

    def setup(self, stage=None):
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )
        self.train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.float32),
        )
        self.valid_dataset = TensorDataset(
            torch.tensor(self.X_valid, dtype=torch.float32),
            torch.tensor(self.y_valid, dtype=torch.float32),
        )
        self.test_dataset = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.float32),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )


def boxcox_inverse(transformed_data, lambda_value):
    if lambda_value == 0:
        return np.exp(transformed_data) - 1
    else:
        return (transformed_data * lambda_value + 1) ** (1 / lambda_value) - 1


class LgbFeatureSelector:
    def __init__(
        self, X_train, y_train, X_test, y_test, model=None, n_features=None, params=None
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.n_features = n_features
        self.params = (
            params
            if params
            else {
                "objective": "regression",
                "boosting_type": "gbdt",
                "metric": "rmse",
                "n_jobs": -1,
                "verbosity": -1,
                "bagging_fraction": 0.5,
                "bagging_freq": 3,
                "feature_fraction": 0.9,
                "lambda_l1": 0.09,
                "lambda_l2": 0.02,
                "learning_rate": 0.08,
                "max_depth": 15,
                "min_data_in_leaf": 2600,
                "min_gain_to_split": 0.02012547611999127,
            }
        )

    def train_model(self):
        train_data = lgb.Dataset(self.X_train, label=self.y_train)
        val_data = lgb.Dataset(self.X_test, label=self.y_test, reference=train_data)
        self.model = lgb.train(
            self.params, train_data, valid_sets=[val_data], early_stopping_rounds=100
        )

    def make(self):
        if self.model is None:
            self.train_model()

        feature_importance = pd.DataFrame(
            {
                "feature": self.X_train.columns,
                "Lgb_importance": self.model.feature_importance(),
            }
        )
        feature_importance.sort_values(
            by="Lgb_importance", ascending=False, inplace=True
        )

        if self.n_features:
            selected_features = feature_importance.head(self.n_features)
        else:
            selected_features = feature_importance

        return selected_features


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV


class LassoFeatureSelector:
    def __init__(
        self,
        X,
        y,
        n_features=None,
        model=None,
        random_state=42,
    ):
        self.X = X
        self.y = y
        self.model = model
        self.random_state = random_state
        self.n_features = n_features

    def make(self):
        if self.model is None:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            lasso_cv = LassoCV(cv=5, random_state=42)
            lasso_cv.fit(X_train, y_train)

            self.model = Lasso(alpha=lasso_cv.alpha_, random_state=self.random_state)

            self.model.fit(self.X, self.y)
            coef_ = np.abs(self.model.coef_)
        feature_importance = pd.DataFrame(
            {
                "feature": self.X.columns,
                "Lasso_importance": coef_,
            }
        )
        feature_importance.sort_values(
            by="Lasso_importance", ascending=False, inplace=True
        )

        if self.n_features:
            selected_features = feature_importance.head(self.n_features)
        else:
            selected_features = feature_importance

        return selected_features


class IcFeatureSelector:
    def __init__(self, X, y, n_features=80) -> None:
        self.X = X
        self.y = y
        self.n_features = n_features

    def make(self):
        cof = self.X.apply(lambda x: np.corrcoef(x, self.y)[0, 1])
        cof = pd.DataFrame({"feature": self.X.columns, "IC_importance": cof})
        cof.sort_values(by="IC_importance", ascending=False, inplace=True)
        return cof.head(self.n_features) if self.n_features else cof


from sklearn.decomposition import PCA


class PcaFeatureSqueezer:
    def __init__(self, X, n_components, random_state=42):
        self.X = X
        self.n_components = n_components
        self.random_state = random_state

    def make(self):
        pca = PCA(n_components=self.n_components, random_state=self.random_state)
        X_transformed = pca.fit_transform(self.X)
        return X_transformed


if __name__ == "__main__":
    # Example usage
    y_test = np.random.randn(10)
    y_pred = np.random.randn(10)
    evaluator = ModelMetrics(y_test, y_pred)
    evaluator.evaluate()
