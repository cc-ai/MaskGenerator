import os
from pathlib import Path
import yaml
from addict import Dict
import torch
import importlib
from models.base_model import BaseModel
import time
import torchvision.utils as vutils
from torch.optim import lr_scheduler
import numpy as np
from PIL import Image
from copy import deepcopy


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


def env_to_path(path):
    """Transorms an environment variable mention in a json
    into its actual value. E.g. $HOME/clouds -> /home/vsch/clouds

    Args:
        path (str): path potentially containing the env variable

    """
    path = str(path)
    path_elements = path.split("/")
    new_path = []
    for el in path_elements:
        if "$" in el:
            new_path.append(os.environ[el.replace("$", "")])
        else:
            new_path.append(el)
    return "/".join(new_path)


def set_data_paths(opts):
    """Update the data files paths in data.files.train and data.files.val
    from data.files.base
    Args:
        opts (addict.Dict): options
    Returns:
        addict.Dict: updated options
    """

    for mode in ["train", "val"]:
        opts.data.files[mode] = str(
            Path(env_to_path(opts.data.files.base)) / opts.data.files[mode]
        )
        if opts.data.use_real:
            opts.data.real_files[mode] = str(
                Path(env_to_path(opts.data.real_files.base))
                / opts.data.real_files[mode]
            )
    return opts


def set_mode(mode, opts):
    opts = deepcopy(opts)
    if mode == "train":
        opts.model.is_train = True
    elif mode == "test":
        opts.model.is_train = False
    return opts


def create_model(opts):
    # Find model in "models" folder
    model_name = str(opts.model.model_name)
    modellib = importlib.import_module("models." + model_name)
    target_model_name = model_name.replace("_", "")
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, BaseModel):
            model = cls
    instance = model()
    instance.initialize(opts)
    # print("model [%s] was created" % (instance.name()))
    return instance


def get_scheduler(optimizer, opts):
    if opts.lr_policy == "lambda":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opts.epoch_count - opts.niter) / float(
                opts.niter_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opts.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opts.lr_decay_iters, gamma=0.1
        )
    elif opts.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif opts.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opts.niter, eta_min=0
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", opts.lr_policy
        )
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


def write_images(
    image_outputs, curr_iter, im_per_row=3, comet_exp=None, store_im=False, is_test=True
):
    """Save output image
    Arguments:
        image_outputs {Tensor list} -- list of output images
        im_per_row {int} -- number of images to be displayed (per row)
        file_name {str} -- name of the file where to save the images
    """

    image_outputs = torch.stack(image_outputs)
    image_grid = vutils.make_grid(
        image_outputs, nrow=im_per_row, normalize=True, scale_each=True
    )
    image_grid = image_grid.permute(1, 2, 0).cpu().detach().numpy()

    if comet_exp is not None:
        if is_test:
            comet_exp.log_image(
                image_grid, name="test_iter_" + str(curr_iter), step=curr_iter
            )
        else:
            comet_exp.log_image(
                image_grid, name="train_iter_" + str(curr_iter), step=curr_iter
            )


def avg_duration(times, batch_size=1):
    """Given a list of times, return the average duration (i.e. difference of times)
    of processing 1 single sample (therefore / batch_size)

    Args:
        times (iterable): Iterable containing the absolute time

    Returns:
        float: Average duration per sample
    """
    t = list(times)
    return (np.array(t + [0]) - np.array([0] + t))[1:-1].mean() / batch_size


def flatten_opts(opts):
    """Flattens a multi-level addict.Dict or native dictionnary into a single
    level native dict with string keys representing the keys sequence to reach
    a value in the original argument.

    d = addict.Dict()
    d.a.b.c = 2
    d.a.b.d = 3
    d.a.e = 4
    d.f = 5
    flatten_opts(d)
    >>> {
        "a.b.c": 2,
        "a.b.d": 3,
        "a.e": 4,
        "f": 5,
    }

    Args:
        opts (addict.Dict or dict): addict dictionnary to flatten

    Returns:
        dict: flattened dictionnary
    """
    values_list = []

    def p(d, prefix="", vals=[]):
        for k, v in d.items():
            if isinstance(v, (Dict, dict)):
                p(v, prefix + k + ".", vals)
            elif isinstance(v, list):
                if v and isinstance(v[0], (Dict, dict)):
                    for i, m in enumerate(v):
                        p(m, prefix + k + "." + str(i) + ".", vals)
                else:
                    vals.append((prefix + k, str(v)))
            else:
                if isinstance(v, Path):
                    v = str(v)
                vals.append((prefix + k, v))

    p(opts, vals=values_list)
    return dict(values_list)


def print_opts(flats):
    """print flatenned opts

    Args:
        flats (dict): flatenned options
    """
    print(
        "\n".join(
            "{:30}: {:15}".format(k, v if v is not None else "")
            for k, v in flats.items()
        )
    )


def convert_depth_unity(im_array, far=1000):
    """
    convert RGB depth image as np.array to array containing metric depth values.
    The depth is encoded in the following way: 
    - The information from the simulator is (1 - LinearDepth (in [0,1])). 
        far corresponds to the furthest distance to the camera included in the depth map. 
        LinearDepth * far gives the real metric distance to the camera. 
    - depth is first divided in 31 slices encoded in R channel with values ranging from 0 to 247
    - each slice is divided again in 31 slices, whose value is encoded in G channel
    - each of the G slices is divided into 256 slices, encoded in B channel
    In total, we have a discretization of depth into N = 31*31*256 - 1 possible values, covering a range of 
    far/N meters.   
    Note that, what we encode here is 1 - LinearDepth so that the furthest point is [0,0,0] (that is sky) 
    and the closest point[255,255,255] 
    The metric distance associated to a pixel whose depth is (R,G,B) is : 
        d = (far/N) * [((255 - R)//8)*256*31 + ((255 - G)//8)*256 + (255 - B)]
                
    """
    im_array = (im_array * 255).type(torch.IntTensor)
    R = im_array[0, :, :]
    G = im_array[1, :, :]
    B = im_array[2, :, :]

    R = ((247 - R) / 8).type(torch.FloatTensor)
    G = ((247 - G) / 8).type(torch.FloatTensor)
    B = (255 - B).type(torch.FloatTensor)
    depth = ((R * 256 * 31 + G * 256 + B).type(torch.FloatTensor)) / (256 * 31 * 31 - 1)
    return (depth * far).unsqueeze(0)


def convert_depth_megadepth(im_array):
    """
    im_array: PIL image of the depth map as torch.Tensor
    The image obtained with megadepth is actually the inverse depth 
    """
    assert torch.Tensor.all(im_array > 0), "MegaDepth depths > 0 "
    return (1 / im_array).type(torch.FloatTensor)


def normalize(arr, min_val=-1, max_val=1):
    """
    Normalize between min and max
    """
    return (max_val - min_val) * (arr - torch.min(arr)) / (
        torch.max(arr) - torch.min(arr)
    ) + min_val


def get_normalized_depth(image_array, mode="unity"):
    """ 
    Args:
        image_array (np.array): np.array of depth map
        mode (str) : "unity" if depth maps come from our simulated world
                    "megadepth" if they were computed with megadepth model
    """
    if mode == "unity":
        depth = convert_depth_unity(image_array, far=1000)
        return normalize(depth)
    elif mode == "megadepth":
        depth = convert_depth_megadepth(image_array)
        return normalize(depth)
        print("depth mode not supported")

def get_model_list(dirname, key):
    """get last model in dirname, whose name contain key
    Arguments:
        dirname {str} -- directory name
        key {str} -- "key" in the model name
    Returns:
        last_model_name {str} -- last model name
    """
    if os.path.exists(dirname) is False:
        return None
    gen_models = [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f
    ]
    if not gen_models:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name