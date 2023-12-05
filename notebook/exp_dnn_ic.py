import utils
from models.common import Pipeline
from models.dnn import  Net
import pytorch_lightning as pl
import wandb
import os
from datetime import datetime
import torch as th

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
# wandb.init(entity="sfcap", project="ml")
RET_SCALER = 1
BATCH_SIZE = 512

X_train = utils.df_train.filter(regex="X_")
X_test = utils.df_test.filter(regex="X_")
y_train = utils.df_train["y_2"] * RET_SCALER
y_test = utils.df_test["y_1"]


model = Net(
    input_size=260,
    hidden_sizes=[200, 150, 80, 50],
    output_size=1,
    lr=1e-3,
    act="SELU",
    loss_fn="pearsonr",
)
datamodule = utils.FinanceDataModule(
    X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE
)

pipeline = Pipeline(early_stop_patient=3, experiment_name=timestamp + "_ic")
pipeline.wandb_logger.experiment.config["label"] = "ic"
pipeline.make(model, datamodule, 50)

pipeline.evaluate(Net, X_test, y_test, RET_SCALER)
