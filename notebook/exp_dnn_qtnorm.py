import utils
from models.common import Pipeline
from models.dnn import Net
import pytorch_lightning as pl
import wandb
import os
from datetime import datetime
from scipy.stats import boxcox
import torch
import numpy as np
from sklearn.preprocessing import QuantileTransformer


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
# wandb.init(entity="sfcap", project="ml")
RET_SCALER = 1
BATCH_SIZE = 512

X_train = utils.df_train.filter(regex="X_")
X_test = utils.df_test.filter(regex="X_")
y_train = utils.df_train["y_2"]
y_test = utils.df_test["y_1"]

qtformer = QuantileTransformer(n_quantiles=10000, output_distribution="normal",random_state=42)
y_train = qtformer.fit_transform(y_train.values.reshape(-1, 1))
y_train *= RET_SCALER
y_test = qtformer.transform(y_test.values.reshape(-1, 1))


model = Net(
    input_size=260,
    hidden_sizes=[200, 150, 50],
    output_size=1,
    lr=1e-4,
    act="SELU",
    loss_fn="mse",
)
datamodule = utils.FinanceDataModule(
    X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE
)

pipeline = Pipeline(early_stop_patient=5, experiment_name=timestamp + "_qtnorm")
pipeline.wandb_logger.experiment.config["label"] = "qtnorm"
pipeline.make(model, datamodule, 60)
pipeline.evaluate(Net, X_test, y_test.flatten(), RET_SCALER)
