"""训练辅助工具：日志目录、全局种子、无限迭代器、配置加载与 checkpoint 查找等杂项函数。"""
import os
import time
import random
import logging
from typing import OrderedDict
import torch
import torch.linalg
import numpy as np
import yaml
from easydict import EasyDict
from glob import glob


class BlackHole(object):
    """什么都不做的"黑洞"对象：调试模式下代替 TensorBoard writer，
    任何属性写入/调用都被静默吞掉。"""

    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


class Counter(object):
    """简单递增计数器；step() 返回步进前的值。"""

    def __init__(self, start=0):
        super().__init__()
        self.now = start

    def step(self, delta=1):
        prev = self.now
        self.now += delta
        return prev


def get_logger(name, log_dir=None):
    # 同时输出到 stderr 与（可选）log_dir/log.txt；
    # 注意重复调用会叠加 handler 导致日志重复，每个进程对同名 logger 只调用一次
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    # 以时间戳创建唯一的日志目录，可附加 prefix/tag 便于区分实验
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    # 统一设置 torch/CUDA/XPU/numpy/random 的全局种子；
    # cudnn.deterministic=True 以牺牲部分速度换取卷积结果可复现
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def inf_iterator(iterable):
    # 无限迭代器：DataLoader 耗尽后自动重新创建，训练循环可无限取批
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k: v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def get_checkpoint_path(folder, it=None):
    # it 给定时返回对应编号的 checkpoint；否则返回迭代数最大的（最新）一个
    if it is not None:
        return os.path.join(folder, '%d.pt' % it), it
    all_iters = list(map(lambda x: int(os.path.basename(x[:-3])), glob(os.path.join(folder, '*.pt'))))
    all_iters.sort()
    return os.path.join(folder, '%d.pt' % all_iters[-1]), all_iters[-1]


def load_config(config_path):
    # 加载 YAML 训练配置并包装为 EasyDict（支持属性式访问）；
    # 返回 (config, 去掉扩展名的配置文件名)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name


def extract_weights(weights: OrderedDict, prefix):
    # 从 state_dict 中抽取以 prefix 开头的键，并去掉前缀——
    # 用于把某个子模块（如编码器）的权重迁移到另一个命名空间
    extracted = OrderedDict()
    for k, v in weights.items():
        if k.startswith(prefix):
            extracted.update({
                k[len(prefix):]: v
            })
    return extracted


def current_milli_time():
    # 当前 Unix 时间（毫秒），用于训练耗时统计
    return round(time.time() * 1000)
