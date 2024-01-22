from typing import *

from torch import Tensor, nn
from torch.utils.data import DataLoader
import torch.autograd as autograd
from approaches.spg.classifier import SPGClassifier, _TaskHead
from approaches.spg.feature_extractor import ModelAlexnet
from approaches.spg.other_tasks_loss import OtherTasksLoss
from approaches.spg.spg import SPG
from utils import assert_type
import torch


"""
这段代码定义了一个名为 ModelSPG 的类，它是一个神经网络模型，
专门设计用于处理多任务学习场景。这个类继承自 PyTorch 的 nn.Module，并且结合了特征提取器和分类器两个关键部分。
总体来说，ModelSPG 类将特征提取和分类处理封装在一起，提供了一个统一的接口来处理前向传播和梯度调整。

__init__ 构造函数
构造函数初始化了模型的两个主要组件：feature_extractor 和 classifier。
feature_extractor 是基于 ModelAlexnet 的模型，用于提取输入数据的特征。
classifier 是 SPGClassifier 的实例，用于进行分类任务。它根据任务数量（list__ncls）和 feature_extractor 的输出维度（nhid）动态创建分类头。
SPGClassifier 的设计使得每个分类任务都有自己的专用分类头，而 feature_extractor 负责提供通用的特征表示。

compute_importance 方法
此方法用于计算网络中各个权重的重要性，特别适用于多任务学习场景。
该方法通过迭代数据集并计算损失函数的梯度来完成。
对于网络中的每个 SPG 模块，收集并累加梯度数据，并最终计算每个模块的激活掩码。

forward 方法
定义了模型的前向传播逻辑。
输入数据首先通过 feature_extractor 提取特征，然后传递给 classifier 进行分类。
返回分类结果以及可能的额外信息（在 misc 字典中）。

modify_grads 方法
modify_grads 方法用于在训练过程中调整模型的梯度。这在多任务学习或增量学习场景中特别重要，以确保模型在学习新任务时不会忘记之前学到的知识。
方法首先调用特征提
取器 (self.feature_extractor) 的 modify_grads 方法，然后调用分类器 (self.classifier) 的 modify_grads 方法，传递相同的参数 args。
"""


class ModelSPG(nn.Module):
    """
    类的内部创建和初始化两个重要的模型组件：特征提取器 (feature_extractor) 和分类器 (classifier)。
    """

    def __init__(self, device: str, list__ncls: List[int], inputsize: Tuple[int, ...], backbone: str, nhid: int,
                 **kwargs):
        """
        类的内部创建和初始化两个重要的模型组件：特征提取器 (feature_extractor) 和分类器 (classifier)。
        Args:
            device: device
            list__ncls:
            inputsize:
            backbone:
            nhid:
            **kwargs:
        """
        super().__init__()

        self.device = device

        if backbone == 'alexnet':
            drop1 = kwargs['drop1']
            drop2 = kwargs['drop2']
            # inputsize, nhid, drop1, drop2 是传递给 ModelAlexnet 构造函数的参数，
            # 分别代表输入大小、隐藏层大小、两个 dropout 层的比率。
            self.feature_extractor = ModelAlexnet(inputsize, nhid=nhid, drop1=drop1, drop2=drop2)
            # list__ncls 是一个列表，包含了每个类或任务的类别数。
            # dim=nhid 指定了分类器内部使用的维度，这里使用与特征提取器相同的隐藏层大小。
            # list__spg=[...] 是一个列表，包含了从特征提取器 (ModelAlexnet) 中提取的几个组件，如卷积层和全连接层。
            # 这可能是为了构建一个特殊的分类器，它在内部使用特征提取器的某些层。
            self.classifier = SPGClassifier(list__ncls, dim=nhid,
                                            list__spg=[self.feature_extractor.c1, self.feature_extractor.c2,
                                                       self.feature_extractor.c3,
                                                       self.feature_extractor.fc1, self.feature_extractor.fc2])
        else:
            raise NotImplementedError
        # endif

    # enddef

    def compute_importance_hessian(self, idx_task: int, dl: DataLoader):
        range_tasks = range(idx_task + 1)
        self.train()
        for t in range_tasks:
            # net = copy.deepcopy(self)
            self.zero_grad()
            # 保存模型的weights
            weights = []
            for module in self.modules():
                if isinstance(module, SPG):
                    weights.append(module.target_module.weight.to(self.device))
            grad_w = None
            grad_f = None
            for w in weights:
                w.requires_grad_(True)
            # 第一次计算梯度
            for x, y in dl:
                x = x.to(self.device)
                y = y.to(self.device)

                args = {
                    'idx_task': t,
                }
                out, _ = self.__call__(x, args=args)

                if t == idx_task:
                    lossfunc = nn.CrossEntropyLoss()
                else:
                    lossfunc = OtherTasksLoss()  # CHI机制
                loss = lossfunc(out, y)
                grad_w_p = autograd.grad(loss, weights, create_graph=False)
                if grad_w is None:  # 第一次没有，需要创建
                    grad_w = list(grad_w_p)
                else:
                    for idx in range(len(grad_w)):
                        grad_w[idx] += grad_w_p[idx]
            # 第二次计算梯度，并与第一次梯度相乘，并将乘积再次投入网络计算梯度
            for x, y in dl:
                x = x.to(self.device)
                y = y.to(self.device)

                args = {
                    'idx_task': t,
                }
                out, _ = self.__call__(x, args=args)
                if t == idx_task:
                    lossfunc = nn.CrossEntropyLoss()
                else:
                    lossfunc = OtherTasksLoss()  # CHI机制
                loss = lossfunc(out, y)
                grad_f = autograd.grad(loss, weights, create_graph=True)
                z = 0
                count = 0
                for module in self.modules():
                    if isinstance(module, SPG):
                        z += (grad_w[count] * grad_f[count]).sum()
                        count += 1
                z.backward()

            for name_module, module in self.named_modules():
                if isinstance(module, SPG):
                    grads = {}
                    for name_param, param in module.target_module.named_parameters():
                        if param.grad is not None:
                            grad = -(param.data * param.grad).clone().cpu()
                        else:
                            grad = 0
                        if name_param not in grads.keys():
                            grads[name_param] = 0
                        grads[name_param] += grad
                    module.register_grad_hessian(idx_task=idx_task, t=t, grads=grads)
        for name, module in self.named_modules():
            if isinstance(module, SPG):
                module.compute_mask_hessian(idx_task=idx_task)

    def compute_importance(self, idx_task: int, dl: DataLoader):
        range_tasks = range(idx_task + 1)

        self.train()
        for t in range_tasks:
            self.zero_grad()

            for x, y in dl:
                x = x.to(self.device)
                y = y.to(self.device)

                args = {
                    'idx_task': t,
                }
                # 调用模型  对每个数据批次调用模型的 __call__ 方法。
                out, _ = self.__call__(x, args=args)
                # 损失函数计算:
                # 根据任务 t 是否是当前任务 idx_task 来选择损失函数
                if t == idx_task:
                    lossfunc = nn.CrossEntropyLoss()
                else:
                    lossfunc = OtherTasksLoss()  # CHI机制
                # endif

                loss = lossfunc(out, y)
                # 对损失进行反向传播
                loss.backward()
            # endfor
            # 处理 SPG 模块
            # 遍历模型的所有命名模块。
            for name_module, module in self.named_modules():
                # 检查模块是否是 SPG 类型。
                # 在每个 SPG 模块上，收集和注册梯度信息，
                if isinstance(module, SPG):
                    grads = {}
                    # 梯度收集:
                    # 梯度被收集并根据任务和时间步骤累加
                    # named_parameters() 方法返回一个生成器，它产生参数的名字 (name_param) 和参数对象 (param) 的对。
                    for name_param, param in module.target_module.named_parameters():
                        # 检查参数是否有梯度（即是否在前向传播过程中被使用）==>拿梯度
                        if param.grad is not None:
                            # 如果有梯度，复制这个梯度值并将其转移到 CPU。
                            # 这样做是为了防止修改原始梯度数据，同时确保梯度数据在不同硬件环境下可用。
                            grad = param.grad.data.clone().cpu()
                        else:
                            grad = 0
                        # 检查 grads 字典中是否已经有当前参数的梯度累加器。==>存梯度
                        if name_param not in grads.keys():
                            # 如果没有，则为这个参数名初始化一个梯度累加器
                            grads[name_param] = 0
                        # 将当前参数的梯度累加到其对应的累加器中。
                        grads[name_param] += grad
                    # endfor

                    module.register_grad(idx_task=idx_task, t=t, grads=grads)
                # endif
            # endfor
        # endfor
        # 并最终计算掩码。
        for name, module in self.named_modules():
            if isinstance(module, SPG):
                module.compute_mask(idx_task=idx_task)

    def plot_mask(self, idx_task: int):
        import matplotlib.pyplot as plt
        for name, module in self.named_modules():
            if isinstance(module, SPG) and isinstance(module.target_module, nn.Conv2d):
                for idx, amax in module.dict_amax.items():
                    if idx == idx_task:
                        count = 1
                        for (k, v) in amax.items():
                            plt.subplot(1, 2, count)
                            plt.title('task ' + str(idx) + ' nn.Linear' + str(k))
                            plt.hist(v.view(-1).cpu(), bins=30)
                            count += 1
                        plt.savefig('plot' + str(idx) + '.png')
                        plt.show()

    def forward(self, x: Tensor, args: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
        assert_type(x, Tensor)

        out, misc1 = self.feature_extractor(x, args=args)
        out, misc2 = self.classifier(out, args=args)

        misc = {
            'reg': 0
        }
        if 'reg' in misc1.keys() and 'reg' in misc2.keys():
            misc['reg'] = misc1['reg'] + misc2['reg']
        return out, misc

    def modify_grads(self, args: Dict[str, Any]):
        self.feature_extractor.modify_grads(args=args)
        self.classifier.modify_grads(args=args)

    def gradient_norm__feature_extractor(self, args: Dict[str, Any], target: Tensor, x: Tensor):
        r = args['r']
        alpha = args['alpha']
        original_grads = []
        with torch.no_grad():
            # Get the original gradients
            for module in self.modules():
                if isinstance(module, SPG):
                    tgt = module.target_module
                    for p in tgt.parameters():
                        if p.grad is not None:
                            original_grads.append(p.grad.clone().detach())

        # Perturb model parameters
        with torch.no_grad():
            count = 0
            for module in self.modules():
                if isinstance(module, SPG):
                    tgt = module.target_module
                    for param in tgt.parameters():
                        grad = original_grads[count]
                        param += r * grad / (torch.norm(grad) + 1e-6)
                        count += 1

        # Forward and backward pass for the perturbed model
        self.zero_grad()
        perturbed_logits, misc = self.__call__(x, args=args)
        lossfunc = nn.CrossEntropyLoss()
        perturbed_loss = lossfunc(perturbed_logits, target)
        # perturbed_loss = self.compute_loss(output=perturbed_logits, target=target, misc=misc)
        perturbed_loss.backward()
        # Compute the final gradient using interpolation
        final_grads = []
        count = 0
        with torch.no_grad():
            for name, module in self.named_modules():
                if isinstance(module, SPG):
                    tgt = module.target_module
                    for param in tgt.parameters():
                        perturbed_grad = param.grad.clone().detach()
                        orig_grad = original_grads[count]
                        interpolated_grad = (1 - alpha) * orig_grad + alpha * perturbed_grad
                        final_grads.append(interpolated_grad)
                        # Reset the parameters to their original values
                        param -= r * orig_grad / (torch.norm(orig_grad) + 1e-6)
                        count += 1

        # Apply the computed gradients to the model
        count = 0
        for name, module in self.named_modules():
            if isinstance(module, SPG):
                tgt = module.target_module
                for param in tgt.parameters():
                    param.grad = final_grads[count]
                    count += 1
        return final_grads

    def gradient_norm__classifier(self, args: Dict[str, Any], target: Tensor, x: Tensor):
        r = args['r']
        alpha = args['alpha']
        original_grads = []
        idx_task = args['idx_task']

        with torch.no_grad():
            # Get the original gradients
            for module in self.modules():
                if isinstance(module, nn.ModuleList):
                    clf = module[idx_task]
                    assert_type(clf, _TaskHead)
                    for p in clf.parameters():
                        if p.grad is not None:
                            original_grads.append(p.grad.clone().detach())
            # Perturb model parameters
            count = 0
            for module in self.modules():
                if isinstance(module, nn.ModuleList):
                    clf = module[idx_task]
                    assert_type(clf, _TaskHead)
                    for param in clf.parameters():
                        grad = original_grads[count]
                        param += r * grad / (torch.norm(grad) + 1e-6)
                        count += 1

        # Forward and backward pass for the perturbed model
        self.zero_grad()
        perturbed_logits, misc = self.__call__(x, args=args)
        lossfunc = nn.CrossEntropyLoss()
        perturbed_loss = lossfunc(perturbed_logits, target)
        # perturbed_loss = self.compute_loss(output=perturbed_logits, target=target, misc=misc)
        perturbed_loss.backward()
        # Compute the final gradient using interpolation
        final_grads = []
        count = 0
        with torch.no_grad():
            for module in self.modules():
                if isinstance(module, nn.ModuleList):
                    clf = module[idx_task]
                    assert_type(clf, _TaskHead)
                    for param in clf.parameters():
                        perturbed_grad = param.grad.clone().detach()
                        orig_grad = original_grads[count]
                        interpolated_grad = (1 - alpha) * orig_grad + alpha * perturbed_grad
                        final_grads.append(interpolated_grad)
                        # Reset the parameters to their original values
                        param -= r * orig_grad / (torch.norm(orig_grad) + 1e-6)
                        count += 1

        # Apply the computed gradients to the model
        count = 0
        for module in self.modules():
            if isinstance(module, nn.ModuleList):
                clf = module[idx_task]
                assert_type(clf, _TaskHead)
                for param in clf.parameters():
                    param.grad = final_grads[count]
                    count += 1
        return final_grads

if __name__ == "__main__":
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    import torch

    device = 'cuda'

    list__ncls = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    inputsize = (3, 32)
    nhid = 100
    backbone = 'alexnet'
    kwargs = {}
    kwargs['drop1'] = 0.05
    kwargs['drop2'] = 0.05

    net = ModelSPG(device=device, inputsize=inputsize, list__ncls=list__ncls, nhid=nhid, backbone=backbone, **kwargs)

    train_set = datasets.CIFAR10(root='./data', train=True,
                                 transform=ToTensor())
    train_loader = DataLoader(train_set, batch_size=64,
                              shuffle=True)
    net.to(device)
    # net.compute_importance_hessian(2, dl=train_loader)
    net.compute_importance(2, dl=train_loader)
