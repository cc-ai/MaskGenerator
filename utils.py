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
import time
import torchvision.utils as vutils


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
        opts.model.is_train = True
    elif mode == "test":
        opts.model.is_train = False

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


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def prepare_sub_folder(output_directory):
    """Create images and checkpoints subfolders in output directory
    Arguments:
        output_directory {str} -- output directory
    Returns:
        checkpoint_directory, image_directory-- checkpoints and images directories
    """
    image_directory = os.path.join(output_directory, "images")
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, "checkpoints")
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_images(image_outputs, curr_iter, im_per_row=3, comet_exp=None, store_im=False, test = True):
    """Save output image
    Arguments:
        image_outputs {Tensor list} -- list of output images
        im_per_row {int} -- number of images to be displayed (per row)
        file_name {str} -- name of the file where to save the images
    """

    image_outputs = torch.stack(image_outputs)
    image_grid = vutils.make_grid(image_outputs, nrow=im_per_row, normalize=True, scale_each=True)
    image_grid = image_grid.permute(1, 2, 0).cpu().detach().numpy()

    if comet_exp is not None:
        if test:
            comet_exp.log_image(image_grid, name="test_iter_" + str(curr_iter))
        else: 
            comet_exp.log_image(image_grid, name="train_iter_" + str(curr_iter))
