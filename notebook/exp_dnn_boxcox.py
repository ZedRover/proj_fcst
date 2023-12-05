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

os.environ["CUDA_VISIBLE_DEVICES"] =[ "1","3"]


timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
# wandb.init(entity="sfcap", project="ml")
RET_SCALER = 1e2
BATCH_SIZE = 4096 * 4

X_train = utils.df_train.filter(regex="X_")
X_test = utils.df_test.filter(regex="X_")
y_train, opt = boxcox(1 + utils.df_train["y_1"])
y_train *= RET_SCALER
y_test = utils.df_test["y_1"]


model = Net(
    input_size=260,
    hidden_sizes=[200, 100, 100, 50, 20],
    output_size=1,
    lr=1e-4,
    act="SiLU",
    loss_fn="mse",
)
datamodule = utils.FinanceDataModule(
    X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE
)

trainer = Pipeline(early_stop_patient=3, experiment_name=timestamp + "_boxcox")
trainer.wandb_logger.experiment.config["label"] = "boxcox"
trainer.make(model, datamodule, 50)


ypred_ = (
    model(torch.from_numpy(X_test.values).float()).detach().numpy().flatten()
    / RET_SCALER
)
y_pred = utils.boxcox_inverse(ypred_, opt)

tm = utils.ModelMetrics(
    y_pred,
    y_test,
    utils.test_date,
)
tm.evaluate()
trainer.wandb_logger.log_text(key="metrics", dataframe=tm.metrics)
