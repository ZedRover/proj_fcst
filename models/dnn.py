import torch
import torch.nn as nn
from .common import *
from typing import *
import sys

sys.path.append("../")
from notebook import utils


# class DNNModel(BaseModel):
#     def __init__(
#         self,
#         input_size: int,
#         hidden_sizes: List[int],
#         output_size: int,
#         dropout_rate: float = 0.5,
#         activation: nn.Module = nn.ReLU,
#         lr: float = 1e-5,
#     ):
#         super().__init__(lr=lr)
#         layers = []
#         layers.append(nn.Linear(input_size, hidden_sizes[0]))
#         layers.append(activation())
#         layers.append(nn.BatchNorm1d(hidden_sizes[0]))
#         layers.append(nn.Dropout(dropout_rate))

#         for i in range(len(hidden_sizes) - 1):
#             layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
#             layers.append(activation())

#         layers.append(nn.Linear(hidden_sizes[-1], output_size))

#         self.layers = nn.Sequential(*layers)
#         self.layers = torch.compile(layers)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.layers(x)

#     def _get_reconstruction_loss(
#         self, y: torch.Tensor, y_hat: torch.Tensor
#     ) -> torch.Tensor:
#         criterion = nn.MSELoss()
#         return criterion(y.squeeze(), y_hat.squeeze())


class Net(BaseModel):
    def __init__(
        self,
        input_size,
        output_size=1,
        hidden_sizes=(256,),
        act="LeakyReLU",
        lr=1e-5,
        loss_fn="mse",
    ):
        super().__init__(lr=lr)
        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_fn == "huber":
            self.loss_fn = nn.HuberLoss()
        elif loss_fn == "pearsonr":
            self.loss_fn = CorrLoss()
        elif loss_fn == "qtile":
            self.loss_fn = WeightedQuantileLoss()
        elif loss_fn == "avgloss":
            self.loss_fn = AvgReturnTop10Loss()
        else:
            self.loss_fn = loss_fn

        hidden_sizes = [input_size] + list(hidden_sizes)
        dnn_layers = []
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        hidden_units = input_size
        for i, (_input_size, hidden_units) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            fc = nn.Linear(_input_size, hidden_units)
            if act == "LeakyReLU":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
                self._wt_init = "leaky_relu"
            elif act == "SiLU":
                activation = nn.SiLU()
            elif act == "SELU":
                activation = nn.SELU()
                self._wt_init = "selu"
            elif act == "HardTanh":
                activation = nn.Hardtanh()
            else:
                self._wt_init = "linear"
                raise NotImplementedError(f"This type of input is not supported")
            bn = nn.BatchNorm1d(hidden_units)
            seq = nn.Sequential(fc, bn, activation)
            dnn_layers.append(seq)
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        fc = nn.Linear(hidden_units, output_size)
        dnn_layers.append(fc)
        # optimizer  # pylint: disable=W0631
        self.dnn_layers = nn.ModuleList(dnn_layers)
        # self.dnn_layers = torch.compile(self.dnn_layers)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity=self._wt_init
                )

    def forward(self, x):
        cur_output = x
        for i, now_layer in enumerate(self.dnn_layers):
            cur_output = now_layer(cur_output)
        return cur_output

    def _get_reconstruction_loss(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        criterion = self.loss_fn

        return criterion(y.squeeze(), y_hat.squeeze())


class MtNet(Net):
    def __init__(
        self,
        input_size,
        output_size=3,
        hidden_sizes=(256,),
        act="LeakyReLU",
        lr=1e-5,
        loss_fn="mse",
    ):
        super().__init__(input_size, output_size, hidden_sizes, act, lr, loss_fn)

    def _get_reconstruction_loss(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        criterion = self.loss_fn

        # Assuming y and y_hat have three columns, one for each task
        task_losses = []
        for task_idx in range(y.size(1)):
            task_loss = criterion(y[:, task_idx], y_hat[:, task_idx])
            task_losses.append(task_loss)

        # Average the losses of all tasks
        total_loss = sum(task_losses) / len(task_losses)
        return total_loss


class SMtNet(BaseModel):
    def __init__(
        self,
        input_size,
        output_size=2,
        hidden_sizes=(256,),
        act="LeakyReLU",
        lr=1e-5,
        loss_fn="mse",
    ):
        super().__init__(lr=lr)
        hidden_sizes = [input_size] + list(hidden_sizes)

        if loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_fn == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_fn == "huber":
            self.loss_fn = nn.HuberLoss()
        # Common layers
        self.common_layers = nn.Sequential()
        hidden_units = input_size
        for i, (_input_size, hidden_units) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            fc = nn.Linear(_input_size, hidden_units)
            if act == "LeakyReLU":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            elif act == "SiLU":
                activation = nn.SiLU()
            elif act == "SELU":
                activation = nn.SELU()
            elif act == "HardTanh":
                activation = nn.Hardtanh()
            else:
                raise NotImplementedError(f"This type of input is not supported")
            bn = nn.BatchNorm1d(hidden_units)
            seq = nn.Sequential(fc, bn, activation)
            self.common_layers.add_module(f"common_layer_{i}", seq)

        # Task-specific towers
        self.tower1 = nn.Linear(hidden_units, 1)
        self.tower2 = nn.Linear(hidden_units, 1)

        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="selu"
                )

    def forward(self, x):
        common_output = self.common_layers(x)
        y1 = self.tower1(common_output)
        y2 = self.tower2(common_output)
        return y1, y2

    def _get_reconstruction_loss(
        self, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        criterion = self.loss_fn
        y1, y2 = y[:, 0], y[:, 1]
        y_hat1, y_hat2 = y_hat[0], y_hat[1]

        loss1 = criterion(y1.squeeze(), y_hat1.squeeze())
        loss2 = criterion(y2.squeeze(), y_hat2.squeeze())

        return (loss1 * 0.8 + loss2 * 0.2) / 2


class MultiTaskNet(Net):
    def __init__(
        self,
        input_size,
        output_size=1,
        hidden_sizes=(256,),
        act="LeakyReLU",
        lr=1e-5,
        loss_fn1="mse",
        loss_fn2="cross_entropy",
        num_classes=10,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=hidden_sizes,
            act=act,
            lr=lr,
            loss_fn=loss_fn1,
        )
        self.num_classes = num_classes
        self.loss_fn1 = self.loss_fn

        self.tower1 = nn.Linear(hidden_sizes[-1], output_size)
        self.tower2 = nn.Linear(hidden_sizes[-1], num_classes)

        if loss_fn2 == "cross_entropy":
            self.loss_fn2 = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError(f"Loss function {loss_fn2} is not supported")

    def forward(self, x):
        base_output = x
        for i, now_layer in enumerate(self.dnn_layers[:-1]):
            base_output = now_layer(base_output)

        output1 = self.tower1(base_output)
        output2 = self.tower2(base_output)

        return output1, output2

    def _get_reconstruction_loss(
        self, y: torch.Tensor, y_hat: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        y_hat1, y_hat2 = y_hat
        loss1 = self.loss_fn1(y.squeeze(), y_hat1.squeeze())
        loss2 = self.loss_fn2(y.squeeze(), y_hat2.squeeze())
        return loss1 + loss2


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(Net):
    def __init__(self, input_size, output_size=1, num_residual_blocks=3):
        super(ResNet, self).__init__(input_size=input_size, output_size=output_size)

        self.in_channels = 64
        self.prep_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64, 64) for _ in range(num_residual_blocks)]
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.prep_layers(x)
        x = self.residual_blocks(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
