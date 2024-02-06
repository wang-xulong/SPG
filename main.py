#!/usr/bin/env python3
import os
import pickle
import tempfile
from datetime import datetime
from typing import *

import hydra
import mlflow
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from optuna import Trial, visualization

import approaches
import utils
from approaches.abst_appr import AbstractAppr
from dataloader import get_shuffled_dataloader
from mymetrics import MyMetrics
from utils import BColors, myprint as print, suggest_float, suggest_int

import matplotlib.pyplot as plt
import wandb

WANDB = True


def instance_appr(trial: Trial, cfg: DictConfig,
                  list__ncls: List[int], inputsize: Tuple[int, ...]) -> AbstractAppr:
    if cfg.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = cfg.device
    # endif
    print(f'device: {device}', bcolor=BColors.OKBLUE)

    seed_pt = suggest_int(trial, cfg, 'seed_pt')
    utils.set_seed_pt(seed_pt)
    # ppr_args 是传递给 Appr 类构造函数的参数。这些参数可能包含了用于初始化这个对象的配置、模型参数
    appr_args = {
        'device': device,
        'list__ncls': list__ncls,
        'inputsize': inputsize,
        'lr': cfg.lr,
        'lr_factor': cfg.lr_factor,
        'lr_min': cfg.lr_min,
        'epochs_max': cfg.epochs_max,
        'patience_max': cfg.patience_max,
        'backbone': cfg.backbone.name,
        'nhid': cfg.nhid,
    }

    # 从配置路径cfg, 'appr', 'tuned', cfg.seq.name,以及变量 pnames[-1] 中获取想要的数值
    def fetch_param_float(*pnames: str) -> float:
        v = suggest_float(trial, cfg, 'appr', 'tuned', cfg.seq.name, pnames[-1])

        return v

    # enddef

    if cfg.appr.name.lower() == 'spg':
        if appr_args['backbone'] in ['alexnet']:
            appr_args['drop1'] = fetch_param_float('drop1')
            appr_args['drop2'] = fetch_param_float('drop2')
            appr_args['lamb'] = 0
            # appr_args['scenario'] = 'ci'
        else:
            raise NotImplementedError
        # endif
    elif cfg.appr.name.lower() == 'gnr':
        if appr_args['backbone'] in ['alexnet']:
            appr_args['drop1'] = fetch_param_float('drop1')
            appr_args['drop2'] = fetch_param_float('drop2')
            appr_args['r'] = fetch_param_float('r')
            appr_args['alpha'] = fetch_param_float('alpha')
            appr_args['lamb'] = appr_args['r'] * appr_args['alpha']
        else:
            raise NotImplementedError
    elif cfg.appr.name.lower() == 'str':
        if appr_args['backbone'] in ['alexnet']:
            appr_args['drop1'] = fetch_param_float('drop1')
            appr_args['drop2'] = fetch_param_float('drop2')
            appr_args['lamb'] = 0
        else:
            raise NotImplementedError
        # endif
    else:
        raise NotImplementedError(cfg.appr.name)
    # endif
    # 上面是给方法设置一堆对应的超参数
    # 下面是实例化一个appr方法
    appr = approaches.appr_spg.Appr(appr_args)
    return appr


def load_dataloader(cfg: DictConfig) -> Dict[int, Dict[str, Any]]:
    basename_data = f'seq={cfg.seq.name}_bs={cfg.seq.batch_size}_seed={cfg.seed}'
    dirpath_data = os.path.join(hydra.utils.get_original_cwd(), 'data')

    # load data
    filepath_pkl = os.path.join(dirpath_data, f'{basename_data}.pkl')
    if os.path.exists(filepath_pkl):
        with open(filepath_pkl, 'rb') as f:
            dict__idx_task__dataloader = pickle.load(f)
        # endwith

        print(f'Loaded from {filepath_pkl}', bcolor=BColors.OKBLUE)
    else:
        dict__idx_task__dataloader = get_shuffled_dataloader(cfg)
        with open(filepath_pkl, 'wb') as f:
            pickle.dump(dict__idx_task__dataloader, f)
        # endwith
    # endif

    # compute hash
    num_tasks = len(dict__idx_task__dataloader.keys())
    hash = []
    for idx_task in range(num_tasks):
        name = dict__idx_task__dataloader[idx_task]['fullname']
        ncls = dict__idx_task__dataloader[idx_task]['ncls']
        num_train = len(dict__idx_task__dataloader[idx_task]['train'].dataset)
        num_val = len(dict__idx_task__dataloader[idx_task]['val'].dataset)
        num_test = len(dict__idx_task__dataloader[idx_task]['test'].dataset)

        msg = f'idx_task: {idx_task}, name: {name}, ncls: {ncls}, num: {num_train}/{num_val}/{num_test}'
        hash.append(msg)
    # endfor
    hash = '\n'.join(hash)

    # check hash
    filepath_hash = os.path.join(dirpath_data, f'{basename_data}.txt')
    if os.path.exists(filepath_hash):
        with open(filepath_hash, 'rt') as f:
            hash_target = f.read()
        # endwith

        assert hash_target == hash

        print(f'Succesfully matched to {filepath_hash}', bcolor=BColors.OKBLUE)
        print(hash)
    else:
        # save hash
        with open(filepath_hash, 'wt') as f:
            f.write(hash)
        # endwith
    # endif

    return dict__idx_task__dataloader


def outer_objective(cfg: DictConfig, expid: str) -> Callable[[Trial], float]:
    # load_dataloader(cfg) 依据cfg获得某一数据集（按照任务）
    dict__idx_task__dataloader = load_dataloader(cfg)
    # .keys() 方法返回字典中所有键
    num_tasks = len(dict__idx_task__dataloader.keys())
    # 从 dict__idx_task__dataloader 字典中为每个任务提取键为 'name' 的值名称，并将这些名称作为一个列表存储在 list__name 变量中
    list__name = [dict__idx_task__dataloader[idx_task]['name'] for idx_task in range(num_tasks)]
    # ncls 为每个任务中的类的个数
    list__ncls = [dict__idx_task__dataloader[idx_task]['ncls'] for idx_task in range(num_tasks)]
    # 样本的尺寸 inputsize 这个都是一样的
    inputsize = dict__idx_task__dataloader[0]['inputsize']  # type: Tuple[int, ...]

    # outer_objective return 内部方法： objective
    # 参数 trial 在这种上下文中是超参数优化过程的一部分，用于获取超参数的值、记录试验过程中的信息，并帮助确定最佳的参数组合
    def objective(trial: Trial) -> float:
        # 实例化一个特定的学习模型
        appr = instance_appr(trial, cfg, list__ncls=list__ncls, inputsize=inputsize)

        # MLflow 中启动一个运行（run）并记录试验参数
        with mlflow.start_run(experiment_id=expid):
            # 这一行记录了试验的参数。trial.params 可能包含了各种超参数的值
            mlflow.log_params(trial.params)
            print(f'\n'
                  f'******* trial params *******\n'
                  f'{trial.params}\n',
                  f'****************************', bcolor=BColors.OKBLUE)

            # 获取测试集
            list__dl_test = [dict__idx_task__dataloader[idx_task]['test'] for idx_task in range(num_tasks)]
            # 实例化mm，用于记录测试集的性能
            # 该实例可能被用于评估和监控在不同数据集或任务上的模型性能
            mm = MyMetrics(list__name, list__dl_test=list__dl_test)

            for idx_task in range(num_tasks):
                # get 训练集和测试集
                dl_train = dict__idx_task__dataloader[idx_task]['train']
                dl_val = dict__idx_task__dataloader[idx_task]['val']

                # 训练前执行相似权重检索
                # appr.before_learning(idx_task=idx_task, dl_train=dl_train, dl_val=dl_val)
                # appr.执行训练
                # 返回值是一个字典
                results_train = appr.train(idx_task=idx_task, dl_train=dl_train, dl_val=dl_val)
                # 从返回的结果中提取了训练轮次和时间消耗
                epoch = results_train['epoch']
                time_consumed = results_train['time_consumed']
                # complete_learning 标志着一个特定任务的学习过程的完成
                appr.complete_learning(idx_task=idx_task, dl_train=dl_train, dl_val=dl_val)
                # 记录与该任务相关的关键信息，如训练轮次和时间消耗， 评估模型的效能和效率
                mm.add_record_misc(idx_task, epoch=epoch, time_consumed=time_consumed)

                # test for all previous tasks
                # 对该任务及其之前的所有任务进行测试
                for t_prev in range(idx_task + 1):
                    results_test = appr.test(t_prev, dict__idx_task__dataloader[t_prev]['test'])
                    # 传递了测试过程中的损失和准确率。
                    loss_test, acc_test = results_test['loss_test'], results_test['acc_test']
                    print(f'[{t_prev}] acc: {acc_test:.3f}')
                    # idx_task_learned=idx_task, idx_task_tested=t_prev 分别指明了学习时的任务索引和当前测试的任务索引。
                    mm.add_record(idx_task_learned=idx_task, idx_task_tested=t_prev,
                                  loss=loss_test, acc=acc_test)
                # endfor

                # save artifacts ；tempfile 模块来创建一个临时目录，并在 MLflow 中记录度量指标和文件
                # tempfile.TemporaryDirectory() 创建一个临时目录，这个目录在 with 语句块结束后会被自动删除。
                with tempfile.TemporaryDirectory() as dir:
                    print(f'ordinary train/test after learning {idx_task}')
                    for mmm in [mm]:
                        idxs = []  # 如果有想忽略的任务，把他的idx_task填到这里
                        metrics_final, list__artifacts = mmm.save(dir, idx_task, indices_task_ignored=idxs)
                        for k, v in metrics_final[idx_task].items():
                            trial.set_user_attr(k, v)
                        # endfor
                        # 记录当前任务的所有度量指标。step=idx_task 表示这些指标记录在训练过程的哪个阶段
                        mlflow.log_metrics(metrics_final[idx_task], step=idx_task)
                        for artifact in list__artifacts:
                            mlflow.log_artifact(artifact)
                        # endfor
                    # endfor
                # endwith
            # endfor
            # 提取最后一个任务的总体准确率，并将这个值赋值给变量 obj
            obj = metrics_final[num_tasks - 1]['acc__Overall']
            if WANDB:
                wandb.log({"acc__Overall": obj,
                           "btf__Overall": metrics_final[num_tasks - 1]['btf__Overall'],
                           "fgt__Overall": metrics_final[num_tasks - 1]['fgt__Overall']
                           })
        # endwith

        print(f'Emptying CUDA cache...')
        torch.cuda.empty_cache()

        return obj

    # enddef

    return objective


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    print(f'\n{OmegaConf.to_yaml(cfg)}')

    utils.set_seed(cfg.seed)
    mlflow.pytorch.autolog()
    expname = cfg.expname
    expid = mlflow.create_experiment(expname)
    n_trials = cfg.n_trials

    # 创建一个优化研究（实验）direction=cfg.optuna.direction 最大化acc
    # 采样器，这里使用的是 TPESampler（树的结构化 Parzen 估计器）
    study = optuna.create_study(direction=cfg.optuna.direction,
                                storage=cfg.optuna.storage,
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed),
                                load_if_exists=False,
                                study_name=expname,
                                )
    # 设置用户自定义的属性
    study.set_user_attr('Completed', False)
    # 执行超参数优化
    # outer_objective(cfg, expid): 这是要优化的目标函数
    # n_trials=n_trials: 这个参数指定了总共要运行的试验次数
    study.optimize(outer_objective(cfg, expid), n_trials=n_trials,
                   gc_after_trial=True, show_progress_bar=True)
    study.set_user_attr('Completed', True)

    print(f'best params: {study.best_params}')
    print(f'best value: {study.best_value}')
    print(study.trials_dataframe())
    print(f'{expname}')

    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.savefig('plot_optimization_history.png')
    optuna.visualization.matplotlib.plot_slice(study)
    plt.savefig('plot_slice.png')
    optuna.visualization.matplotlib.plot_contour(study)
    plt.savefig('plot_contour.png')
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.savefig('plot_param_importances.png')
    optuna.visualization.matplotlib.plot_parallel_coordinate(study, params=list(study.best_params.keys()))
    plt.savefig('plot_parallel_coordinate.png')

    plt.show()


if __name__ == '__main__':
    OmegaConf.register_new_resolver('now', lambda pattern: datetime.now().strftime(pattern))
    if WANDB:
        wandb.init(project='FE-10', name=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), save_code=True)
    main()
    if WANDB:
        wandb.finish()
