import optuna
from sklearn.metrics import mean_squared_error
import utils
import lightgbm as lgb
import numpy as np
from scipy.stats import pearsonr
from lightgbm import early_stopping, log_evaluation
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
import datetime
import warnings
import sys
from wandb.lightgbm import wandb_callback, log_summary
import pandas as pd

warnings.filterwarnings("ignore")
time_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def resample_by_y_abs(df, n_samples_per_group=500):
    df = df.assign(y_abs=df["y_1"].abs())
    df_sorted = df.sort_values(by="y_abs", ascending=False)
    idx = df_sorted.index[n_samples_per_group:-n_samples_per_group]
    df_tail = df_sorted.drop(idx, axis=0)
    df_mid = df_sorted.iloc[n_samples_per_group:-n_samples_per_group].sample(
        n=500, random_state=42
    )
    return pd.concat([df_tail, df_mid], axis=0)


sampled_data = utils.df_train
# grouped_train = sampled_data.groupby("datetime")
# df_train_resampled = grouped_train.apply(resample_by_y_abs, 1000)

# sample_fraction = 1
# sampled_data = utils.df_train.sample(frac=sample_fraction, random_state=42)

# Split the sampled data into features (X) and target (y)
X_sampled = sampled_data.loc[:, "X_1":"X_260"].fillna(0)
y_sampled = sampled_data["y_1"].fillna(0)


def objective(trial):
    # Define hyperparameters to optimize
    params = {
        "objective": "regression",
        "boosting_type": "gbdt",
        "metric": "rmse",
        "num_leaves": trial.suggest_int("num_leaves", 31, 500, step=20),
        "max_depth": trial.suggest_int("max_depth", -1, 20),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "max_bin": trial.suggest_int("max_bin", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0, step=0.1),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0, step=0.1),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
        "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 20),
        "min_sum_hessian_in_leaf": trial.suggest_float(
            "min_sum_hessian_in_leaf", 0.001, 1.0
        ),
        "n_jobs": -1,
        "verbosity": -1,
    }

    # Perform cross-validation to evaluate the model
    n_folds = 5
    kf = TimeSeriesSplit(
        n_splits=n_folds,
    )
    mse_scores = []
    ic_scores = []

    for train_index, val_index in kf.split(X_sampled):
        X_train_cv, X_val_cv = X_sampled.iloc[train_index], X_sampled.iloc[val_index]
        y_train_cv, y_val_cv = y_sampled.iloc[train_index], y_sampled.iloc[val_index]

        train_data = lgb.Dataset(X_train_cv, label=y_train_cv)
        val_data = lgb.Dataset(X_val_cv, label=y_val_cv, reference=train_data)

        gbm = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=val_data,
            callbacks=[
                early_stopping(20),
                log_evaluation(100),
            ],
        )
        y_pred_val = gbm.predict(X_val_cv, num_iteration=gbm.best_iteration)
        rmse = mean_squared_error(y_val_cv, y_pred_val, squared=False)
        mse_scores.append(rmse)
        ic = np.corrcoef(y_val_cv, y_pred_val)[0, 1]
        # ic, _ = pearsonr(y_val_cv, y_pred_val)
        ic_scores.append(ic)
    # Calculate the mean rmse across all folds
    mean_rmse = np.mean(mse_scores)
    mean_ic = np.mean(ic)
    print("===============================================")
    print("Mean rmse: {:.5f}".format(mean_rmse))
    print("Mean IC: {:.5f}".format(mean_ic))
    print("===============================================")
    return mean_ic  # not apply


if __name__ == "__main__":
    # Create an Optuna study and run the optimization
    study_name = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    study = optuna.create_study(
        direction="maximize", storage=f"sqlite:///gbm/lgb_{study_name}.db"
    )
    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"Value: {trial.value:.5f}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    utils.save_dict_to_yaml(trial.params, f"configs/lgb_tuning_{time_now}.yaml")
