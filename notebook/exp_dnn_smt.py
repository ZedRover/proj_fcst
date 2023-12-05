import utils
from models.common import Pipeline
from models.dnn import    Net, SMtNet
import pytorch_lightning as pl
import wandb
import os
from datetime import datetime

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


timestamp = datetime.now().strftime("%Y-%m%d-%H%M")
# wandb.init(entity="sfcap", project="ml")
RET_SCALER = 1e2
BATCH_SIZE = 4096 * 4

X_train = utils.df_train.filter(regex="X_")
X_test = utils.df_test.filter(regex="X_")
y_train = utils.df_train[["y_1", "y_2"]] * RET_SCALER
y_test = utils.df_test[["y_1", "y_2"]] * RET_SCALER


model = SMtNet(
    input_size=260,
    hidden_sizes=[200, 100, 50],
    output_size=2,
    lr=1e-4,
    act="SELU",
    loss_fn="mse",
)
datamodule = utils.FinanceDataModule(
    X_train, X_test, y_train, y_test, batch_size=BATCH_SIZE
)

pipeline = Pipeline(early_stop_patient=5, experiment_name=timestamp + "_smultask")
pipeline.wandb_logger.experiment.config["label"] = "smultask"
pipeline.make(model, datamodule, 50)
pipeline.evaluate(SMtNet, X_test, y_test, RET_SCALER)
