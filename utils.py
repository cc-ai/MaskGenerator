import os
import re
from pathlib import Path
import subprocess
from copy import copy
import yaml
from addict import Dict
from torch.nn import init
import torch
import numpy as np
import importlib
from models.base_model import BaseModel
import functools
import torch.nn as nn


def load_opts(path=None, default=None):
    """Loads a configuration Dict from 2 files:
    1. default files with shared values across runs and users
    2. an overriding file with run- and user-specific values
    Args:
        path (pathlib.Path): where to find the overriding configuration
            default (pathlib.Path, optional): Where to find the default opts.
            Defaults to None. In which case it is assumed to be a default config
            which needs processing such as setting default values for lambdas and gen
            fields
    Returns:
        addict.Dict: options dictionnary, with overwritten default values
    """
    if default is None:
        default_opts = Dict()
    else:
        with open(default, "r") as f:
            default_opts = Dict(yaml.safe_load(f))

    with open(path, "r") as f:
        overriding_opts = Dict(yaml.safe_load(f))

    default_opts.update(overriding_opts)

    return set_data_paths(default_opts)


def set_data_paths(opts):
    """Update the data files paths in data.files.train and data.files.val
    from data.files.base
    Args:
        opts (addict.Dict): options
    Returns:
        addict.Dict: updated options
    """

    for mode in ["train", "val"]:
        for domain in opts.data.files[mode]:
            opts.data.files[mode] = str(Path(opts.data.files.base) / opts.data.files[mode])

    return opts


def set_mode(mode, opts):
    if mode == "train":
        opts.mode.is_train = True
    elif mode == "test":
        opts.mode.is_train = False

    return opts


def create_model(opt):
    # Find model in "models" folder
    model_name = str(opt.model.model_name)
    modellib = importlib.import_module("models." + model_name)
    target_model_name = model_name.replace("_", "")
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls
    instance = model()
    instance.initialize(opt)
    # print("model [%s] was created" % (instance.name()))
    return instance


def get_scheduler(optimizer, opt):
    if opt.lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError("learning rate policy [%s] is not implemented", opt.lr_policy)
    return scheduler
