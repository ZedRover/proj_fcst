import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    GradientAccumulationScheduler,
    ModelCheckpoint,
)
import sys
from typing import *
import os
import numpy as np
from audtorch.metrics.functional import pearsonr

sys.path.append("../")
from notebook import utils

torch.set_float32_matmul_precision("high")


class BaseModel(pl.LightningModule):
    def __init__(self, lr: float = 1e-5):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Please implement the 'forward' method.")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self._get_reconstruction_loss(y, y_hat)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        if y_hat.shape[1] == 1:
            loss = self._get_reconstruction_loss(y, y_hat)
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
            ic = pearsonr(y_hat.squeeze(), y.squeeze()).squeeze()
            self.log("val_ic", ic, sync_dist=True)
        else:
            loss = self._get_reconstruction_loss(y, y_hat)
            self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = self._get_reconstruction_loss(y, y_hat)
        self.log("test_loss", loss, sync_dist=True)
        ic = pearsonr(y_hat.squeeze(), y.squeeze()).squeeze()
        self.log("test_ic", ic, sync_dist=True)
        self.log({"y_hat": y_hat, "y": y}, sync_dist=True)
        return loss

    def _get_reconstruction_loss(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError(
            "Please implement the '_get_reconstruction_loss' method."
        )


class Pipeline:
    def __init__(
        self,
        devices: Any = -1,
        early_stop_patient: int = 10,
        experiment_name: str = "tmp",
    ):
        self.devices = devices
        self.wandb_logger: pl_loggers.WandbLogger = pl_loggers.WandbLogger(
            project="ml-final", name=experiment_name
        )
        self.accumulator = GradientAccumulationScheduler(scheduling={4: 2})
        ckp_dir = os.path.join("./checkpoints", experiment_name)
        if not os.path.exists(ckp_dir):
            os.makedirs(ckp_dir, exist_ok=True)
        self.checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
            monitor="val_loss",
            dirpath=ckp_dir,
            filename="{epoch}-{val_loss:.3f}-{train_loss:.3f}",
            save_top_k=1,
            mode="min",
        )
        self.early_stop: EarlyStopping = EarlyStopping(
            monitor="val_loss", patience=early_stop_patient, mode="min"
        )
        self.trainer: pl.Trainer = None

    def make(
        self,
        model: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        max_epochs: int = 100,
        **kwargs,
    ) -> None:
        trainer = pl.Trainer(
            logger=self.wandb_logger,
            callbacks=[self.checkpoint_callback, self.early_stop],
            max_epochs=max_epochs,
            devices=self.devices,
            log_every_n_steps=10,
            **kwargs,
        )
        trainer.fit(model, datamodule)
        self.trainer = trainer

    def evaluate(self, model_class, X_test, y_test, RET_SCALER):
        if hasattr(X_test, "values"):
            X_test = X_test.values

        model = model_class.load_from_checkpoint(
            self.trainer.checkpoint_callback.best_model_path
        )
        y_test_pred = (
            model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().flatten()
            / RET_SCALER
        )
        evaluator = utils.ModelMetrics(y_test_pred, y_test, utils.test_date)
        evaluator.evaluate()
        self.wandb_logger.log_text(key="metrics", dataframe=evaluator.metrics)
        self.wandb_logger.log_metrics(
            {
                "IC": evaluator.mean_ic,
                "rt9": evaluator.mean_rt9,
                "ac9": evaluator.mean_ac9,
            }
        )


class CorrLoss(nn.Module):
    def __init__(self):
        super(CorrLoss, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred_mean = torch.mean(y_pred)
        y_true_mean = torch.mean(y_true)

        cov = torch.mean((y_pred - y_pred_mean) * (y_true - y_true_mean))

        y_pred_std = torch.std(y_pred)
        y_true_std = torch.std(y_true)

        pearson_corr = cov / (y_pred_std * y_true_std)

        loss = 1 - pearson_corr
        return loss


class WeightedQuantileLoss(nn.Module):
    def __init__(self, quantile=0.9):
        super(WeightedQuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        positive_errors = torch.max(errors, torch.zeros_like(errors))
        negative_errors = torch.max(-errors, torch.zeros_like(errors))

        # Calculate the quantile-based weights
        positive_weights = (1 - self.quantile) * torch.ones_like(errors)
        negative_weights = self.quantile * torch.ones_like(errors)

        # Calculate the weighted quantile loss
        loss = torch.mean(
            positive_weights * positive_errors + negative_weights * negative_errors
        )
        return loss


class AvgReturnTop10Loss(nn.Module):
    def __init__(self, alpha=5):
        super(AvgReturnTop10Loss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        errors = torch.pow(y_true - y_pred, 2)

        top_10_percent = int(y_pred.size(0) * 0.1)
        _, indices = torch.topk(y_true, top_10_percent)

        penalty_factor = torch.ones(y_pred.size(0), dtype=torch.float32, device="cuda")
        penalty_factor[indices] = self.alpha

        weighted_errors = penalty_factor * errors

        loss = torch.mean(torch.abs(weighted_errors))
        return loss


import torchsort


class SpearmanCorrLoss(torch.nn.Module):
    def __init__(self, regularization_strength=1.0):
        super(SpearmanCorrLoss, self).__init__()
        self.regularization_strength = regularization_strength

    @staticmethod
    def corrcoef(target, pred):
        pred_n = pred - pred.mean()
        target_n = target - target.mean()
        pred_n = pred_n / pred_n.norm()
        target_n = target_n / target_n.norm()
        return (pred_n * target_n).sum()

    def forward(self, pred, target):
        target = target.unsqueeze(1)
        pred = pred.unsqueeze(1)
        pred_rank = torchsort.soft_rank(
            pred, regularization_strength=self.regularization_strength
        )
        target_rank = torchsort.soft_rank(
            target, regularization_strength=self.regularization_strength
        )
        corr = self.corrcoef(target_rank, pred_rank)
        return -corr  # 我们希望最小化损失，因此需要取负的Spearman相关系数


class PinballLoss(nn.Module):
    def __init__(self, quantile=0.9):
        super(PinballLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        errors = y_true - y_pred
        loss = torch.mean(
            torch.max(self.quantile * errors, (self.quantile - 1) * errors)
        )
        return loss


class DynamicWeightedLoss(nn.Module):
    def __init__(self, base_loss=nn.MSELoss()):
        super(DynamicWeightedLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, y_pred, y_true):
        errors = torch.abs(y_true - y_pred)
        weights = torch.exp(errors)
        weighted_errors = weights * self.base_loss(y_pred, y_true)
        loss = torch.mean(weighted_errors)
        return loss


class MarginLoss(nn.Module):
    def __init__(self, margin=0.1, base_loss=nn.MSELoss()):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.base_loss = base_loss

    def forward(self, y_pred, y_true):
        base_loss = self.base_loss(y_pred, y_true)
        top_10_percent = int(y_pred.shape[0] * 0.1)
        sorted_pred_diff = torch.sort(
            y_pred[:-top_10_percent] - y_pred[-top_10_percent:]
        )[0]
        margin_loss = torch.mean(torch.clamp(self.margin - sorted_pred_diff, min=0.0))
        total_loss = base_loss + margin_loss
        return total_loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        alpha=0.5,
        base_loss1=nn.MSELoss(),
        base_loss2=CorrLoss(),
        beta1=1,
        beta2=1,
    ):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.base_loss1 = base_loss1
        self.base_loss2 = base_loss2
        self.beta1 = beta1
        self.beta2 = beta2

    def forward(self, y_pred, y_true):
        loss1 = self.base_loss1(y_pred, y_true) * self.beta1
        loss2 = self.base_loss2(y_pred, y_true) * self.beta2
        combined_loss = self.alpha * loss1 + (1 - self.alpha) * loss2
        print(
            "loss1: {:.5f}, loss2: {:.5f}, combined_loss: {:.5f}".format(
                loss1, loss2, combined_loss
            )
        )
        return combined_loss


class SoftRankLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(SoftRankLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred, y_true):
        sorted_y_true, _ = torch.sort(y_true, descending=True)
        sorted_y_pred, _ = torch.sort(y_pred, descending=True)
        loss = torch.mean(
            torch.relu(
                self.margin
                - (sorted_y_true[:-1] - sorted_y_true[1:])
                + (sorted_y_pred[:-1] - sorted_y_pred[1:])
            )
        )
        return loss


# class RankingLoss(nn.Module):
#     def __init__(self, alpha=1):
#         super(RankingLoss, self).__init__()
#         self.alpha = alpha

#     def forward(self, y_pred, y_true):
#         # Pointwise regression loss
#         reg_loss = torch.norm(y_pred - y_true, 2) ** 2

#         # Pairwise ranking-aware loss
#         N = y_pred.size(0)
#         rank_loss = 0
#         for i in range(N):
#             for j in range(N):
#                 diff_pred = y_pred[i] - y_pred[j]
#                 diff_true = y_true[i] - y_true[j]
#                 rank_loss += torch.relu(-diff_pred * diff_true)


#         # Combine both losses with alpha hyperparameter
#         loss = reg_loss + self.alpha * rank_loss
#         return loss
class RankingLoss(nn.Module):
    def __init__(self, alpha=2e-5, beta_1=1, beta_2=1):
        super(RankingLoss, self).__init__()
        self.alpha = alpha
        self.loss_fn1 = CorrLoss()
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def forward(self, y_pred, y_true):
        # Pointwise regression loss
        # reg_loss = torch.norm(y_pred - y_true, 2) ** 2
        reg_loss = self.beta_1 * self.loss_fn1(y_pred, y_true)

        # Pairwise ranking-aware loss
        N = y_pred.size(0)
        y_pred_exp = y_pred.unsqueeze(1).expand(-1, N)
        y_true_exp = y_true.unsqueeze(1).expand(-1, N)

        diff_pred_matrix = y_pred_exp - y_pred_exp.t()
        diff_true_matrix = y_true_exp - y_true_exp.t()

        rank_loss_matrix = torch.relu(-diff_pred_matrix * diff_true_matrix)
        rank_loss = self.beta_2 * self.alpha * rank_loss_matrix.sum() / N

        # Combine both losses with alpha hyperparameter
        loss = reg_loss + rank_loss
        print(
            "loss: {:.5f}| reg_loss: {:.5f}| rank_loss: {:.5f}\n".format(
                loss, reg_loss, rank_loss
            )
        )
        return loss


from torchmetrics import ConcordanceCorrCoef


class CCCLoss(nn.Module):
    def __init__(self, num_outputs=1) -> None:
        super(CCCLoss, self).__init__()
        self.num_outputs = num_outputs
        self.loss_fn = ConcordanceCorrCoef(num_outputs=num_outputs)

    def forward(self, y_pred, y_true):
        return 1 - self.loss_fn(y_pred, y_true)
