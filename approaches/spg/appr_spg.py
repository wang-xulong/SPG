from typing import *
from torch import Tensor
from approaches.abst_appr import AbstractAppr
from approaches.spg.model_spg import ModelSPG

"""
这段代码定义了一个名为 Appr 的类，继承自 AbstractAppr。这个类似乎是为了处理特定的学习任务或训练过程而设计的

__init__ 方法
Appr 类的构造
函数接收一个名为 appr_args 的字典作为参数，这个字典包含了初始化所需的各种参数和设置。
使用 super().__init__(**appr_args) 调用父类 AbstractAppr 的构造函数，传递解包的 appr_args 字典。

complete_learning 方法
这个方法用于在完成一个学习任务后执行必要的后续步骤。具体来说，它涉及到计算模型参数的重要性。
方法接收任务索引 idx_task 和一个字典 kwargs，其中包含训练所需的附加信息，如数据加载器 dl_train。
方法内部从 kwargs 中提取数据加载器 dl，然后调用模型的 compute_importance 方法来计算参数的重要性。
这一步骤可能涉及到评估参数对模型性能的影响，以便在后续的训练或剪枝操作中作出合理的决策。

modify_grads 方法
这个方法用于在训练过程中调整模型的梯度。这在处理多任务学习时尤为重要，因为不同的任务可能需要不同的梯度调整策略。
方法接收一个字典 args，其中包含调整梯度所需的参数。
方法内部调用模型的 modify_grads 方法，传递 args 字典。这可能包括梯度裁剪、梯度缩放或其他形式的梯度调整，
以帮助模型在学习新任务的同时保留对之前任务的记忆。
"""


class Appr(AbstractAppr):
    def __init__(self, appr_args: Dict[str, Any]):
        """
        在 Python 中，在函数调用时使用 ** 有一个特别的含义。当你在调用函数或构造函数时使用 ** 加上一个字典，
        它会将这个字典的键值对展开为命名参数。这被称为 "关键字参数展开,**appr_args 会将 appr_args 字典中
        的键值对展开为关键字参数。例如，如果 appr_args 是 {'param1': value1, 'param2': value2}，那么
         **appr_args 相当于 param1=value1, param2=value2。这个语句是在调用 Appr 类的父类 AbstractAppr
         的构造函数。使用 **appr_args 表示将 appr_args 字典中的所有键值对作为命名参数传递给 AbstractAppr
         类的构造函数。这样做的好处是，你不需要明确知道父类构造函数需要哪些参数，只需传入一个包含所有可能参数的字典即可。
        Args:
            appr_args: 相当于 param1=value1, param2=value2
        """
        super().__init__(**appr_args)
        # ModelSPG(**appr_args) 继承到nn.Module，最后会返回self.feature_extractor = ModelAlexnet和
        # self.classifier = SPGClassifier两个模块
        self.model = ModelSPG(**appr_args).to(self.device)

    # enddef

    def before_learning(self, idx_task: int, **kwargs) -> None:
        dl = kwargs['dl_train']
        # self.model.compute_importance_hessian(idx_task=idx_task, dl=dl)

    def complete_learning(self, idx_task: int, **kwargs) -> None:
        """
        从 kwargs 字典中提取键为 'dl_train' 的值并将其赋给变量 dl。这里 dl 可能是一个数据加载器（DataLoader），用于加载训练数据。
        Args:
            idx_task:是一个参数，表示当前任务的索引，应该是一个整数
            **kwargs:是一个可变关键字参数字典，允许你传入任意数量的命名参数
        """
        # 从 kwargs 字典中提取键为 'dl_train' 的值并将其赋给变量 dl。
        # 这里 dl 可能是一个数据加载器（DataLoader），用于加载训练数据。
        dl = kwargs['dl_train']
        # 在论文中，当前任务收敛完后，开始计算重要性gamma
        self.model.compute_importance(idx_task=idx_task, dl=dl)
        # self.model.compute_importance_hessian(idx_task=idx_task, dl=dl)  # 改为hessian
        # self.model.plot_mask(idx_task=idx_task)   # 绘制掩码的分布

    # enddef

    def modify_grads(self, args: Dict[str, Any]):
        # 修改梯度
        self.model.modify_grads(args=args)

    # enddef
    def gradient_norm(self, args: Dict[str, Any], target: Tensor, x: Tensor):
        self.model.gradient_norm__feature_extractor(args=args, x=x, target=target)
        self.model.gradient_norm__classifier(args=args, x=x, target=target)
