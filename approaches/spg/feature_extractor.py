from typing import *

import numpy as np
from torch import Tensor, nn

from approaches.spg.spg import SPG
from utils import assert_type
import torch
"""
这段代码定义了一个基于 AlexNet 架构的深度学习模型 ModelAlexnet，它在 PyTorch 框架下实现，
并且利用了先前定义的 SPG 类来增强其卷积和全连接层。下面逐一解释这些方法和类的主要特点：

ModelAlexnet 类
ModelAlexnet 是一个继承自 nn.Module 的类，用于构建神经网络模型。
构造函数 __init__ 接收输入尺寸 inputsize、隐藏层大小 nhid 和两个 dropout 率 drop1 和 drop2。
网络由三个卷积层（c1、c2、c3）和两个全连接层（fc1、fc2）组成，每个层都被 SPG 类封装，这提供了额外的功能，如梯度和激活掩码的跟踪。

compute_conv_output_size 方法
这是一个静态方法，用于计算卷积操作后的输出尺寸。它根据输入尺寸、卷积核大小、步长、填充和膨胀系数来计算输出尺寸。

forward 方法
定义了模型的前向传播逻辑。
输入张量 x 通过卷积层和全连接层传播，其中包含了 ReLU 激活函数、最大池化和 dropout 操作。
返回最后一层的输出和一个名为 misc 的字典，其中可能包含
了其他与模型相关的信息，如正则化项。

modify_grads 方法
这个方法用于调整网络中的梯度。它特别针对多任务学习场景，其中不同任务可能需要不同的梯度调整策略。
在方法中，通过迭代模型的所有子模块，检查是否有 SPG 类型的模块。如果是，那么对这些模块调用 softmask 方法，以便进行梯度的软性调整。
这种梯度调整有助于在多任务学习中平衡新旧任务之间的影响，防止灾难性遗忘。
"""

class ModelAlexnet(nn.Module):
    def __init__(self, inputsize: Tuple[int, ...], nhid: int, drop1: float, drop2: float):
        super().__init__()

        nch, size = inputsize[0], inputsize[1]

        self.c1 = SPG(nn.Conv2d(nch, 64, kernel_size=size // 8))
        s = self.compute_conv_output_size(size, size // 8)
        s = s // 2

        self.c2 = SPG(nn.Conv2d(64, 128, kernel_size=size // 10))
        s = self.compute_conv_output_size(s, size // 10)
        s = s // 2

        self.c3 = SPG(nn.Conv2d(128, 256, kernel_size=2))
        s = self.compute_conv_output_size(s, 2)
        s = s // 2

        self.smid = s
        self.maxpool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout(drop1)
        self.drop2 = nn.Dropout(drop2)

        self.fc1 = SPG(nn.Linear(256 * self.smid ** 2, nhid))
        self.fc2 = SPG(nn.Linear(nhid, nhid))
    # endddef

    @staticmethod
    def compute_conv_output_size(Lin, kernel_size, stride=1, padding=0, dilation=1) -> int:
        return int(np.floor((Lin + 2 * padding - dilation * (kernel_size - 1) - 1) / float(stride) + 1))
    # enddef

    def forward(self, x: Tensor, args: Dict[str, Any]) -> Tuple[Tensor, Dict]:
        assert_type(x, Tensor)

        self.device = x.device

        h = self.maxpool(self.drop1(self.relu(self.c1(x))))
        h = self.maxpool(self.drop1(self.relu(self.c2(h))))
        h = self.maxpool(self.drop2(self.relu(self.c3(h))))

        h = h.view(h.shape[0], -1)
        h = self.drop2(self.relu(self.fc1(h)))
        h = self.drop2(self.relu(self.fc2(h)))

        misc = {
            'reg': self.calculate_gradient_norm()
            }

        return h, misc
    # enddef

    @torch.no_grad()
    def calculate_gradient_norm(self) -> Tensor:
        gradient_norm = 0.0
        for module in self.modules():
            if isinstance(module, SPG):
                tgt = module.target_module
                for p in tgt.parameters():
                    if p.grad is not None:
                        grad = p.grad.clone().detach()
                        gradient_norm += torch.norm(grad)

        return gradient_norm

    def modify_grads(self, args: Dict[str, Any]):
        idx_task = args['idx_task']

        if idx_task == 0:
            return
        # endif

        for name_module, module in self.named_modules():
            if isinstance(module, SPG):
                module.softmask(idx_task)


# if __name__ == "__main__":
#     inputsize = (3, 32)
#     a = ModelAlexnet(inputsize, nhid=100, drop1=0.5, drop2=0.5)
#
#     for name_module, module in a.named_modules():
#         if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#             print(module)