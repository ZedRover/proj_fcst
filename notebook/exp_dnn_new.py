import utils
from models.common import *
from models.dnn import    Net,MultiTaskNet
import pytorch_lightning as pl
import wandb
import os
from datetime import datetime
import torch as th
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

RET_SCALER = 1
BATCH_SIZE = 10311
LR = 1e-4
LOSS_FUNC = RankingLoss()
LABEL = f"{LOSS_FUNC.__class__}" if type(LOSS_FUNC) is not str else LOSS_FUNC


print("====================================")
print(timestamp)
print(LABEL)
print("====================================")

X_train = utils.df_train.filter(regex="X_")
X_test = utils.df_test.filter(regex="X_")
y_train = utils.df_train.loc[:, ["y_2", "rank_label"]]
y_test = utils.df_test.loc[:, ["y_1", "rank_label"]]


model = MultiTaskNet(
    input_size=260,
    hidden_sizes=[200, 150, 100],
    output_size=1,
    loss_fn1="mse",
    num_classes=4,
    act="SELU",
)
datamodule = utils.FinanceDataModule(
    X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE
)

pipeline = Pipeline(early_stop_patient=3, experiment_name=timestamp + f"_{LABEL}")
pipeline.wandb_logger.experiment.config["label"] = LABEL
pipeline.make(model, datamodule, 50, num_sanity_val_steps=0)

# pipeline.evaluate(Net, X_test, y_test, RET_SCALER)
# print("====================================")
# print(timestamp)
# print(LABEL)
# print("====================================")
