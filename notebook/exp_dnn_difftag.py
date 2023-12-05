import utils
from models.common import Pipeline
from models.dnn import  Net
import pytorch_lightning as pl
import wandb
import os
from datetime import datetime
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

exp_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
os.mkdir(f"./checkpoints/tmspc/{exp_name}")
# wandb.init(entity="sfcap", project="ml")
RET_SCALER = 1e2
BATCH_SIZE = 4096 * 2

tags = utils.df_train.timestamp.unique()

X_trains = []
X_tests = []
y_trains = []
y_tests = []

for tag in tags:
    X_train = utils.df_train[utils.df_train.timestamp == tag].filter(regex="X_")
    X_test = utils.df_test[utils.df_test.timestamp == tag].filter(regex="X_")
    y_train = utils.df_train[utils.df_train.timestamp == tag]["y_1"] * RET_SCALER
    y_test = utils.df_test[utils.df_test.timestamp == tag]["y_1"] * RET_SCALER
    X_trains.append(X_train)
    X_tests.append(X_test)
    y_trains.append(y_train)
    y_tests.append(y_test)

models = [
    Net(
        input_size=260,
        hidden_sizes=[200, 150, 80, 20],
        output_size=1,
        lr=1e-3,
        act="SELU",
    )
    for _ in range(3)
]
datamodules = [
    utils.FinanceDataModule(
        X_trains[i], X_tests[i], y_trains[i], y_tests[i], batch_size=BATCH_SIZE
    )
    for i in range(3)
]

trainers = [
    Pipeline(early_stop_patient=5, experiment_name=exp_name + "_timespec")
    for _ in range(3)
]
for trainer in trainers:
    trainer.wandb_logger.experiment.config["label"] = "spec"


timestamps = [utils.df_test.query("timestamp==@tags[@i]").datetime for i in range(3)]
for i in range(3):
    trainers[i].make(models[i], datamodules[i], 100)
    print(f"Finish {tags[i]}".center(50, "="))
    trainers[i].evaluate(models[i], X_tests[i], timestamps[i], RET_SCALER)
    torch.save(
        models[i].state_dict(), f"checkpoints/tmspc/{exp_name}/dnn_tmspc_{tags[i]}.pth"
    )
