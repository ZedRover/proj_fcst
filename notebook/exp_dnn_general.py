import utils
from models.common import *
from models.dnn import Net
import pytorch_lightning as pl
import wandb
import os
from datetime import datetime
import torch as th

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

RET_SCALER = 1
BATCH_SIZE = 10311
LR = 1e-4
# LOSS_FUNC = RankingLoss(alpha=2e-1,beta_1=0)
LOSS_FUNC = CCCLoss()
LABEL = f"{LOSS_FUNC.__class__.__name__}" if type(LOSS_FUNC) is not str else LOSS_FUNC


print("====================================")
print(timestamp)
print(LABEL)
print("====================================")

X_train = utils.df_train.filter(regex="X_")
X_test = utils.df_test.filter(regex="X_")
y_train = utils.df_train["y_2"] * RET_SCALER
y_test = utils.df_test["y_1"]


model = Net(
    input_size=260,
    hidden_sizes=[200, 150, 100, 50],
    output_size=1,
    loss_fn=LOSS_FUNC,
    act="SELU",
    lr=LR,
)
datamodule = utils.FinanceDataModule(
    X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE
)

pipeline = Pipeline(early_stop_patient=3, experiment_name=timestamp + f"_{LABEL}")
pipeline.wandb_logger.experiment.config["label"] = LABEL
pipeline.make(
    model=model,
    datamodule=datamodule,
    max_epochs=50,
    num_sanity_val_steps=0,

)

pipeline.evaluate(Net, X_test, y_test, RET_SCALER)
print("====================================")
print(timestamp)
print(LABEL)
print("====================================")
