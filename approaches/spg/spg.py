from typing import *

import torch
from torch import Tensor, nn

from utils import assert_type, myprint as print

"""
代码定义了一个名为 SPG 的类，继承自 PyTorch 的 nn.Module。该类提供了一系列方法，用于在神经网络的训练过程中实现特定的功能，如梯度的跟踪、激活掩码的计算和应用，以及软性梯度掩蔽。
__init__ 方法
初始化 SPG 类的实例。它接收一个目标模块 target_module 作为参数，这个模块将被 SPG 类包装。此外，该方法初始化几个字典属性，用于存储历史掩码、最大激活掩码和梯度数据。

forward 方法
定义了模块的前向传播逻辑。这个方法简单地调用了 target_module 的前向传播方，并返回其输出。

standardize_pm1 和 standardize 方法
这些方法用于标准化数据。standardize 方法将输入张量标准化为均值为 0，标准差为 1 的分布。standardize_pm1 在标准化后进一步应用 tanh 函数，将数据缩放到 [-1, 1] 的范围内。

register_grad 方法
用于在不同任务和时间步骤下追踪每个参数的梯度累计情况。这个方法更新 dict__idx_task__t__h 字典，该字典按任务索引和时间步骤组织了梯度数据。

compute_mask 方法
根据历史梯度数据计算每个参数的激活掩码，并存储在 history_mask 字典中。这个方法有助于跟踪参数在不同任务中的重要性。

a_max 方法
计算并更新每个参数的最大激活掩码。这个方法比较当前任务和之前任务的激活掩码，选择较大的掩码作为最大

激活掩码，并将其存储在 dict_amax 字典中。这有助于跟踪在多任务学习过程中每个参数的最重要状态。

softmask 方法
这个方法应用软性掩蔽机制来调整网络参数的梯度。它首先使用 a_max 方法获取最大激活掩码，然后基于这个掩码计算一个减少因子（red），用于调整目标模块参数的梯度。这样做的目的是在训练过程中软性地调整网络的学习过程，减少对旧任务学习的干扰，同时为新任务腾出学习空间。
"""


class SPG(nn.Module):
    """
    SPG 类的设计使其充当了对另一个神经网络模块的包装器。它通过在自己的 forward 方法中调用 target_module 的处理逻辑来实现这一点，
    并提供了额外的属性来跟踪历史数据和执行其他可能的操作。这种模式在 PyTorch 中是构建复杂神经网络架构时的常见做法，
    特别是在需要添加额外功能或自定义行为到现有模块时。
    """

    def __init__(self, target_module: nn.Module):
        super().__init__()
        # 构造函数接收一个 nn.Module 对象作为参数。这个对象称为 target_module，代表将要被 SPG 类包装的目标模块。
        assert_type(target_module, nn.Module)

        self.target_module = target_module
        # 在当前任务t-1训练完成后更新，输入为self.dict__idx_task__t__h  最后仅保留任务i的掩码
        self.history_mask = dict()  # type: Dict[int, Dict[str, Tensor]]
        # 在学习新任务的时候被def a_max()使用到，它保留了在学习任务t时候，前t-1个任务的最大掩码 key=idx_task
        # 为了尽可能减少遗忘，dict_amax 字典每一个k-v 都对应任务的过去每个任务的最大掩码 即 \gmma^<=t-1_i
        self.dict_amax = {}
        # 用于存储当前任务(idx_task)对过去任务(t)步骤特定的梯度数据。
        # 在   def register_grad 有初始化
        # {idx_task 0:{}, idx_task 1:{}, ...}
        #                             |
        #                             {t 0: {}, t 1: {}, ... }
        #                                            |
        #                                         {name1: grad1, name2: grad1 , name3: grad1 , ... }
        self.dict__idx_task__t__h = {}  # type: Dict[int, Dict[int, Dict[str, Tensor]]]
        self.dict__idx_task__t__h__hessian = {}  # type: Dict[int, Dict[int, Dict[str, Tensor]]] # save for hvp

    # enddef

    def forward(self, x: Tensor) -> Tensor:
        assert_type(x, Tensor)

        out = self.target_module(x)

        return out

    # enddef

    # 论文 Eq. (1).(3)
    def standardize_pm1(self, x: Tensor) -> Tensor:
        if torch.all(x == 0):
            pass
        else:
            x = self.standardize(x)
        # endif
        ret = torch.tanh(x)
        # ret = 2*torch.sigmoid(x)-1

        return ret

    # enddef

    @classmethod
    def standardize(cls, x: Tensor) -> Tensor:
        sh = x.shape
        x = x.view(-1)

        ret = (x - x.mean()) / x.std()

        return ret.view(*sh)

    # enddef

    def register_grad_hessian(self, idx_task: int, t: int, grads: Dict[str, Tensor]):
        # 初始化字典结构:
        if idx_task not in self.dict__idx_task__t__h__hessian.keys():
            self.dict__idx_task__t__h__hessian[idx_task] = {}
        # endif

        if t not in self.dict__idx_task__t__h__hessian[idx_task].keys():
            self.dict__idx_task__t__h__hessian[idx_task][t] = {}
        # endif
        # 遍历传入的梯度:
        for name, grad in grads.items():
            # 累加梯度
            if name in self.dict__idx_task__t__h__hessian[idx_task][t].keys():
                grad_prev = self.dict__idx_task__t__h__hessian[idx_task][t][name]
            else:
                grad_prev = 0
            # endif

            grad_new = grad_prev + grad
            # 更新梯度信息:
            # 更新 self.dict__idx_task__t__h[idx_task][t][name] 的值为累加后的新梯度值 (grad_new)。
            self.dict__idx_task__t__h__hessian[idx_task][t][name] = grad_new

    # 方法的作用是追踪在不同任务(idx_task)和过去任务(t)步骤下，每个参数的梯度累计情况
    def register_grad(self, idx_task: int, t: int, grads: Dict[str, Tensor]):
        # 初始化字典结构:
        if idx_task not in self.dict__idx_task__t__h.keys():
            self.dict__idx_task__t__h[idx_task] = {}
        # endif

        if t not in self.dict__idx_task__t__h[idx_task].keys():
            self.dict__idx_task__t__h[idx_task][t] = {}
        # endif
        # 遍历传入的梯度:
        for name, grad in grads.items():
            # 累加梯度
            if name in self.dict__idx_task__t__h[idx_task][t].keys():
                grad_prev = self.dict__idx_task__t__h[idx_task][t][name]
            else:
                grad_prev = 0
            # endif

            grad_new = grad_prev + grad
            # 更新梯度信息:
            # 更新 self.dict__idx_task__t__h[idx_task][t][name] 的值为累加后的新梯度值 (grad_new)。
            self.dict__idx_task__t__h[idx_task][t][name] = grad_new
        # endfor

    # enddef

    def compute_mask_hessian(self, idx_task: int):
        if idx_task not in self.dict__idx_task__t__h__hessian.keys():
            # ablations can take this route.
            return
        # endif

        names = self.dict__idx_task__t__h__hessian[idx_task][idx_task].keys()
        history = {}  # type: Dict[str, Tensor]
        for t, dict__name__h in self.dict__idx_task__t__h__hessian[idx_task].items():
            assert names == dict__name__h.keys()
            # history 字典初始化
            for name, h in dict__name__h.items():
                if name not in history.keys():
                    history[name] = torch.zeros_like(h)
                # endif
                # 赋值
                history[name] = torch.max(history[name], self.standardize_pm1(h).abs())  # 论文Eq.(4) \gamma^t_i
            # endfor
        # endfor

        self.history_mask = {idx_task: history.copy()}  # 保存 \gamma^t_i

        self.dict__idx_task__t__h__hessian.clear()

    def compute_mask(self, idx_task: int):
        if idx_task not in self.dict__idx_task__t__h.keys():
            # ablations can take this route.
            return
        # endif

        names = self.dict__idx_task__t__h[idx_task][idx_task].keys()
        history = {}  # type: Dict[str, Tensor]
        for t, dict__name__h in self.dict__idx_task__t__h[idx_task].items():
            assert names == dict__name__h.keys()
            # history 字典初始化
            for name, h in dict__name__h.items():
                if name not in history.keys():
                    history[name] = torch.zeros_like(h)
                # endif
                # 赋值
                history[name] = torch.max(history[name], self.standardize_pm1(h).abs())  # 论文Eq.(4) \gamma^t_i
            # endfor
        # endfor

        self.history_mask = {idx_task: history.copy()}  # 保存 \gamma^t_i

        self.dict__idx_task__t__h.clear()

    # enddef
    # 在给定任务索引 idx_task 和神经网络模块 latest_module 的情况下，计算和更新每个参数的最大激活掩码。
    # 返回 dict_amax[idx_task]，即当前任务的最大激活掩码
    def a_max(self, idx_task: int, latest_module: nn.Module) -> Dict[str, Tensor]:
        if idx_task == 0:
            return None
        else:
            if idx_task not in self.dict_amax.keys():
                ret = dict()

                for name_param, param in latest_module.named_parameters():
                    curr = self.history_mask[idx_task - 1][name_param]  # \gmma^t_i
                    if idx_task - 1 in self.dict_amax.keys():
                        prev = self.dict_amax[idx_task - 1][name_param]  # \gmma^<=t-1_i
                    else:
                        prev = curr
                    # endif

                    v1 = torch.max(prev, curr)  # 论文Eq.(5)            max(\gmma^t_i, \gmma^<=t-1_i)
                    ret[name_param] = v1
                # endfor

                self.dict_amax[idx_task] = ret  # 更新最大的掩码值  # 论文Eq.(5) \gmma^<=t_i
            # endif

            return self.dict_amax[idx_task]


    def softmask(self, idx_task: int):
        tgt = self.target_module

        a_max = self.a_max(idx_task, tgt)

        for n, p in tgt.named_parameters():
            if p.grad is None:
                pass
            else:
                red = (1 - a_max[n]).to(p.device)  # 论文 Eq(6)
                p.grad.data *= red  # 注意，仅使用.data  数值计算，是不会被考虑到计算图中

                if False:
                    num_0 = red[red == 0].numel()
                    num_all = red.numel()

                    classname = self.target_module.__class__.__name__
                    msg = f'[{classname}.{n}]' \
                          f' dead: {num_0}/{num_all}({num_0 / num_all:.2%})'
                    print(msg)
                # endif
            # endif
        # endfor
    # enddef
