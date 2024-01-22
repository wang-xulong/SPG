from typing import *

import torch
from torch import Tensor, nn

from approaches.spg.spg import SPG
from utils import assert_type
from approaches.abst_appr import AbstractAppr


"""
代码定义了一个名为 SPGClassifier 的类，它继承自 PyTorch 的 nn.Module。SPGClassifier 专为处理多任务学习场景设计，
其中每个任务都有一个专用的分类器头。此外，该类还定义了一个内部类 _TaskHead，用于表示单个任务的分类器

SPGClassifier 类
构造函数 (__init__):
接收任务的类别数列表 list__ncls、特征维度 dim 和一系列 SPG 模块的列表 list__spg。
对于 list__ncls 中的每个类别数 ncls，创建一个 _TaskHead 实例，并将其添加到模块列表 self.list__classifier 中。

前向传播 (forward):
根据传入的任务索引 idx_task，选择相应的 _TaskHead 实例进行前向传播。
返回分类器的输出。

梯度修改 (modify_grads):
对模型参数应用梯度裁剪以防止梯度爆炸。
对每个 _TaskHead 实例调用 softmask 方法以软性调整梯度。

_TaskHead 类
构造函数 (__init__):
接收一个 nn.Linear 类型的分类器和 list__spg 列表。
初始化分类器和 SPG 模块列表。

前向传播 (forward):
直接通过内部的 nn.Linear 分类器进行前向传播。

软性掩蔽 (softmask):
计算调整因子（modification），这是根据 SPG 模块列表中的最大激活掩码计算得到的。调整因子用于软性调整分类器参数的梯度，
以便在新任务的学习中减少对旧任务知识的干扰。
如果是第一次为特定的任务 idx_task 调用 softmask，则计算并存储调整因子。否则，使用之前计算的调整因子。
通过遍历分类器的所有参数，并将它们的梯度乘以调整因子，来实现软性调整。
"""


class SPGClassifier(nn.Module):
    def __init__(self, list__ncls: List[int], dim: int, list__spg: List[SPG]):
        super().__init__()

        self.list__classifier = nn.ModuleList()
        for ncls in list__ncls:
            head = _TaskHead(nn.Linear(dim, ncls), list__spg=list__spg)
            self.list__classifier.append(head)
        # endfor

    # enddef

    def forward(self, x: Tensor, args: Dict[str, Any]) -> Tuple[Any, Dict[str, Tensor]]:
        assert_type(x, Tensor)

        idx_task = args['idx_task']

        clf = self.list__classifier[idx_task]
        x = x.view(x.shape[0], -1)
        out = clf(x)

        misc = {
            'reg': self.calculate_gradient_norm(args)
        }

        return out, misc

    # enddef

    def modify_grads(self, args: Dict[str, Any]) -> None:
        idx_task = args['idx_task']

        torch.nn.utils.clip_grad_norm_(self.parameters(), 10000)

        for _, module in self.list__classifier.named_modules():
            if isinstance(module, _TaskHead):
                module.softmask(idx_task=idx_task)

    @torch.no_grad()
    def calculate_gradient_norm(self, args: Dict[str, Any]) -> Tensor:
        gradient_norm = 0.0
        idx_task = args['idx_task']
        for module in self.modules():
            if isinstance(module, nn.ModuleList):
                clf = module[idx_task]
                assert_type(clf, _TaskHead)
                for p in clf.parameters():
                    if p.grad is not None:
                        grad = p.grad.clone().detach()
                        gradient_norm += torch.norm(grad)
        return gradient_norm

class _TaskHead(nn.Module):
    def __init__(self, classifier: nn.Linear, list__spg: List[SPG]):
        super().__init__()

        self.classifier = classifier
        self.list__spg = list__spg

        self.dict__idx_task__red = {}

        self.device = None

    # enddef

    def forward(self, x: Tensor) -> Tensor:
        if self.device is None:
            self.device = x.device
        # endif

        return self.classifier(x)

    # enddef

    def softmask(self, idx_task: int):
        # 检查当前任务 idx_task 是否已经有计算好的调整因子（modification）。如果没有，则进行以下步骤：
        if idx_task not in self.dict__idx_task__red.keys():
            list__amax = []
            # 对于每个 SPG 模块
            for spg in self.list__spg:
                # 调用 a_max 方法来获取最大激活掩码。
                dict__amax = spg.a_max(idx_task=idx_task, latest_module=spg.target_module)
                # 将这些掩码（如果存在）合并到一个列表 list__amax 中
                if dict__amax is not None:
                    for _, amax in dict__amax.items():
                        list__amax.append(amax.view(-1))
                    # endfor
                # endif
            # endfor
            # 如果 list__amax 不为空，计算所有掩码值的平均，并由此得出调整因子。
            # 调整因子是基于平均掩码值的补数（1 - mean_amax）计算得到的。
            if len(list__amax) > 0:
                amax = torch.cat(list__amax, dim=0)
                mean_amax = amax.mean()
                modification = (1 - mean_amax).cpu().item()
            else:
                # 如果 list__amax 为空，调整因子设为 1。
                modification = 1
            # endif
            self.dict__idx_task__red[idx_task] = modification
        else:
            modification = self.dict__idx_task__red[idx_task]
        # endif

        if False:
            print(f'[classifier] modification: {modification}')
        # endif
        # 遍历分类器（self.classifier）的所有参数。
        # 对于每个有梯度的参数，将其梯度乘以调整因子。
        for n, p in self.classifier.named_parameters():
            if p.grad is not None:
                p.grad.data *= modification
            # endif
        # endfor
    # enddef
