# -*- coding: utf-8 -*-

import torch
import numpy as np
from os import environ
import random
import logging
from dataclasses import dataclass
from typing import Union, Tuple

logger = logging.getLogger(__name__)


""" Floating-point precision
"""

def init_torch(double_precision=False, cmd_args=None):
    logger.info("Initializing torch")
    logger.info("double_precision={}".format(double_precision))
    if cmd_args is not None:
        logger.info("Device={}".format(cmd_args.device))
    if double_precision:
        torch.set_default_dtype(torch.float64)

""" Reproducible PRNG streams
"""

def init_prngs(cmd_args):
    torch.manual_seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    random.seed(cmd_args.seed)
    if cmd_args.device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if "PYTHONHASHSEED" not in environ:
        logger.warn(
            "PYTHONHASHSEED is not defined, this may cause reproducibility issues"
        )


"""Logging utils
"""

def init_logging(logger_name, logfile, log_level: str, cmd_args):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError("Invalid log level: %s" % log_level)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    f_handler = logging.FileHandler(logfile)
    f_handler.setLevel(numeric_level)
    c_handler = logging.StreamHandler()
    c_handler.setLevel(numeric_level)
    f_handler.setFormatter(formatter)
    c_handler.setFormatter(formatter)
    
    logging.basicConfig(level=numeric_level, handlers=[f_handler, c_handler])
    logger = logging.getLogger(logger_name)
    return logger


def log_dir_base_args(base, name=None):
    if name:
        base = "{}/{}".format(base, name)
    return base


def log_dir_base(cmd_args):
    return log_dir_base_args(cmd_args.save_dir, cmd_args.name)


def log_additions(additions, l1reg, l2reg, augmentation, dropout=None, weight_decay=None):
    additions_list = [ item + "_" + str(dropout) if (item == "dropout" and dropout is not None) else item for item in additions ]
    name = "-".join(additions_list)
    if l1reg:
        name = "{}-l1-{:0.5f}".format(name, l1reg)
    if l2reg:
        name = "{}-l2-{:0.5f}".format(name, l2reg)
    if augmentation:
        name = "{}-{}".format(name, "augmentation")
    if weight_decay is not None:
        name = "{}-{}".format(name, "wd_" + str(weight_decay))
    if name == "":
        return "default"
    return name.strip("-")


def all_log_dir_args(
    base,
    model,
    data,
    label_noise,
    seed,
    l1reg=None,
    l2reg=None,
    name=None,
    augmentation=None,
    model_additions=(),
    dropout=0.,
    weight_decay=0.,
):
    if weight_decay == 0.:
        weight_decay = None
    if dropout == 0.:
        dropout = None
    dir_name = "{}/{}/{}/{}".format(
        log_dir_base_args(base, name),
        model,
        data,
        log_additions(model_additions, l1reg, l2reg, augmentation, dropout, weight_decay),
    )
    sub_dir_name = ""
    if label_noise != 0:
        sub_dir_name = "noise-{:.4f}".format(label_noise)
    if seed is not None:
        sub_dir_name = "-".join([sub_dir_name, "seed-{}".format(seed)])
    else:
        sub_dir_name = "-".join([sub_dir_name, "no_seed"])
    return "{}/{}".format(dir_name, sub_dir_name.strip("-"))


def all_log_dir(cmd_opts):
    return all_log_dir_args(
        cmd_opts.save_dir,
        cmd_opts.model,
        cmd_opts.data,
        cmd_opts.label_noise,
        cmd_opts.seed,
        cmd_opts.l1_regularization,
        cmd_opts.l2_regularization,
        cmd_opts.name,
        cmd_opts.augmentation,
        cmd_opts.model_additions,
        cmd_opts.dropout,
        cmd_opts.weight_decay,
    )

def prepare_dirs(args):
    """Prepare directories to store results and logs"""
    import os
    
    logs_path = all_log_dir(args)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    logfile = os.path.join(logs_path, args.logfile)
    # configure root logger
    init_logging(None, logfile, args.log_level, args)
    return logs_path


""" Checkpointing utils
"""

def checkpoint_dir_args(
    base,
    model,
    data,
    label_noise,
    seed,
    l1reg=None,
    l2reg=None,
    name=None,
    augmentation=None,
    model_additions=(),
    dropout=0.,
    weight_decay=0.,
):
    return "{}/checkpoints".format(
        all_log_dir_args(
            base, model, data, label_noise, seed, l1reg, l2reg, name, augmentation, model_additions, dropout, weight_decay
        )
    )

def checkpoint_dir(cmd_opts):
    return checkpoint_dir_args(
        cmd_opts.save_dir,
        cmd_opts.model,
        cmd_opts.data,
        cmd_opts.label_noise,
        cmd_opts.seed,
        cmd_opts.l1_regularization,
        cmd_opts.l2_regularization,
        cmd_opts.name,
        cmd_opts.augmentation,
        cmd_opts.model_additions,
        cmd_opts.dropout,
        cmd_opts.weight_decay,
    )


"""Training utils
"""

def global_iteration_from_engine(engine):
    def _wrap_global_step(engine_, event_name_):
        return engine.state.iteration

    return _wrap_global_step


"""Dataset utils
"""

@dataclass(frozen=True)
class DatasetInfo:
    __slots__ = ["name", "input_shape", "output_dimension"]
    name: str
    input_shape: Union[Tuple[int, int, int], int]
    output_dimension: int

