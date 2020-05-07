from pathlib import Path
import yaml
import json
from torch.utils.data import DataLoader, Dataset
from imageio import imread
from torchvision import transforms
import numpy as np
from .transforms import get_transforms
from PIL import Image
from addict import Dict
from utils import get_normalized_depth

IMG_EXTENSIONS = set(
    [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP"]
)


class SimDataset(Dataset):
    def __init__(self, opts, transform=None, no_check=False):

        isTrain = opts.model.is_train
        if isTrain:
            file_list_path = Path(opts.data.files.train)
        else:
            file_list_path = Path(opts.data.files.val)

        if file_list_path.suffix == ".json":
            self.samples_paths = self.json_load(file_list_path)
        elif file_list_path.suffix in {".yaml", ".yml"}:
            self.samples_paths = self.yaml_load(file_list_path)
        else:
            raise ValueError("Unknown file list type in {}".format(file_list_path))

        self.file_list_path = str(file_list_path)
        if not no_check:
            self.check_samples()
        self.transform = transform
        self.opts = opts

    def check_samples(self):
        """Checks that every file listed in samples_paths actually
        exist on the file-system
        """
        l, p = (len(self.samples_paths), self.file_list_path)
        print(f"Checking {l} samples in {p}...", end="", flush=True)
        for s in self.samples_paths:
            for k, v in s.items():
                assert Path(v).exists(), f"{k} {v} does not exist"
        print(" ok.")

    def json_load(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def yaml_load(self, file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def __getitem__(self, i):
        """Return an item in the dataset with fields:
        {
            data: transform({),
            paths: [{task: path}],
            mode: [train|val]
        }
        Args:
            i (int): index of item to retrieve
        Returns:
            dict: dataset item where tensors of data are in item["data"] which is a dict
                  {task: tensor}
        """
        paths = self.samples_paths[i]
        if self.transform:
            return Dict(
                {
                    "data": self.transform(
                        {
                            task: pil_image_loader(path, task)
                            for task, path in paths.items()
                        }
                    ),
                    "paths": paths,
                }
            )

        return Dict(
            {
                "data": {
                    task: pil_image_loader(path, task) for task, path in paths.items()
                },
                "paths": paths,
            }
        )

    def __len__(self):
        return len(self.samples_paths)


def pil_image_loader(path, task):
    if Path(path).suffix == ".npy":
        arr = np.load(path).astype(np.uint8)
    elif is_image_file(path):
        arr = imread(path).astype(np.uint8)
    else:
        raise ValueError("Unknown data type {}".format(path))
    if task == "m" or task == "rm":
        # Check if >1 channel:
        if len(arr.shape) > 2 and arr.shape[-1] > 1:
            mask_thresh = (np.max(arr) - np.min(arr)) / 2.0
            arr = np.squeeze((arr > mask_thresh).astype(np.float)[:, :, 0])

    # if task == "s":
    #     arr = decode_segmap(arr)
    # assert len(arr.shape) == 3, (path, task, arr.shape)

    return Image.fromarray(arr)


def is_image_file(filename):
    """Check that a file's name points to a known image format
    """
    return Path(filename).suffix in IMG_EXTENSIONS


class RealSimDataset(Dataset):
    def __init__(self, opts, transform=None, no_check=False):

        isTrain = opts.model.is_train
        if isTrain:
            file_list_path = Path(opts.data.files.train)
            real_file_list_path = Path(opts.data.real_files.train)
        else:
            file_list_path = Path(opts.data.files.val)
            real_file_list_path = Path(opts.data.real_files.val)

        if file_list_path.suffix == ".json":
            self.samples_paths = self.json_load(file_list_path)
            self.real_samples_paths = self.json_load(real_file_list_path)
        elif file_list_path.suffix in {".yaml", ".yml"}:
            self.samples_paths = self.yaml_load(file_list_path)
            self.real_samples_paths = self.yaml_load(real_file_list_path)
        else:
            raise ValueError("Unknown file list type in {}".format(file_list_path))

        self.file_list_path = str(file_list_path)
        self.real_file_list_path = str(real_file_list_path)
        if not no_check:
            self.check_samples()
        self.transform = transform
        self.opts = opts

    def check_samples(self):
        """Checks that every file listed in samples_paths actually
        exist on the file-system
        """
        l, p = (len(self.samples_paths), self.file_list_path)
        print(f"Checking {l} samples in {p}...", end="", flush=True)

        for s in self.samples_paths:
            for k, v in s.items():
                assert Path(v).exists(), f"{k} {v} does not exist"

        print(" ok.")

        l, p = (len(self.real_samples_paths), self.real_file_list_path)
        print(f"Checking {l} samples in {p}...", end="", flush=True)

        for s in self.real_samples_paths:
            for k, v in s.items():
                assert Path(v).exists(), f"{k} {v} does not exist"

        print(" ok.")

    def json_load(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def yaml_load(self, file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def __getitem__(self, i):
        """Return an item in the dataset with fields:
        {
            data: transform({),
            paths: [{task: path}],
            mode: [train|val]
        }
        Args:
            i (int): index of item to retrieve
        Returns:
            dict: dataset item where tensors of data are in item["data"] which is a dict
                  {task: tensor}
        """

        real_i = i % len(self.real_samples_paths) - 1
        paths = self.samples_paths[i]
        paths["rx"] = self.real_samples_paths[real_i]["x"]
        paths["rm"] = self.real_samples_paths[real_i]["m"]

        if self.transform:
            return Dict(
                {
                    "data": self.transform(
                        {
                            task: pil_image_loader(path, task)
                            for task, path in paths.items()
                        }
                    ),
                    "paths": paths,
                }
            )

        return Dict(
            {
                "data": {
                    task: pil_image_loader(path, task) for task, path in paths.items()
                },
                "paths": paths,
            }
        )

    def __len__(self):
        return len(self.samples_paths)


def get_loader(opts, real=True, depth=True, no_check=False):
    if real:
        if depth is True:
            return DataLoader(
                RealSimDepthDataset(
                    opts,
                    transform=transforms.Compose(get_transforms(Dict(opts))),
                    no_check=no_check,
                ),
                batch_size=opts.data.loaders.get("batch_size", 4),
                shuffle=True,
                num_workers=opts.data.loaders.get("num_workers", 8),
            )
        else:
            return DataLoader(
                RealSimDataset(
                    opts,
                    transform=transforms.Compose(get_transforms(Dict(opts))),
                    no_check=no_check,
                ),
                batch_size=opts.data.loaders.get("batch_size", 4),
                shuffle=True,
                num_workers=opts.data.loaders.get("num_workers", 8),
            )
    else:
        if depth is True:
            return DataLoader(
                SimDepthDataset(
                    opts,
                    transform=transforms.Compose(get_transforms(Dict(opts))),
                    no_check=no_check,
                ),
                batch_size=opts.data.loaders.get("batch_size", 4),
                shuffle=True,
                num_workers=opts.data.loaders.get("num_workers", 8),
            )
        else:
            return DataLoader(
                SimDataset(
                    opts,
                    transform=transforms.Compose(get_transforms(Dict(opts))),
                    no_check=no_check,
                ),
                batch_size=opts.data.loaders.get("batch_size", 4),
                shuffle=True,
                num_workers=opts.data.loaders.get("num_workers", 8),
            )


class RealSimDepthDataset(Dataset):
    def __init__(self, opts, transform=None, no_check=False):

        isTrain = opts.model.is_train
        if isTrain:
            file_list_path = Path(opts.data.files.train)
            real_file_list_path = Path(opts.data.real_files.train)
        else:
            file_list_path = Path(opts.data.files.val)
            real_file_list_path = Path(opts.data.real_files.val)

        if file_list_path.suffix == ".json":
            self.samples_paths = self.json_load(file_list_path)
            self.real_samples_paths = self.json_load(real_file_list_path)
        elif file_list_path.suffix in {".yaml", ".yml"}:
            self.samples_paths = self.yaml_load(file_list_path)
            self.real_samples_paths = self.yaml_load(real_file_list_path)
        else:
            raise ValueError("Unknown file list type in {}".format(file_list_path))

        if not no_check:
            self.check_samples()
        self.file_list_path = str(file_list_path)
        self.real_file_list_path = str(real_file_list_path)
        self.transform = transform
        self.opts = opts

    def check_samples(self):
        """Checks that every file listed in samples_paths actually
        exist on the file-system
        """
        for s in self.samples_paths:
            for k, v in s.items():
                assert Path(v).exists(), f"{k} {v} does not exist"

        for s in self.real_samples_paths:
            for k, v in s.items():
                assert Path(v).exists(), f"{k} {v} does not exist"

    def json_load(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def yaml_load(self, file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def __getitem__(self, i):
        """Return an item in the dataset with fields:
        {
            data: transform({),
            paths: [{task: path}],
            mode: [train|val]
        }
        Args:
            i (int): index of item to retrieve
        Returns:
            dict: dataset item where tensors of data are in item["data"] which is a dict
                  {task: tensor}
        """

        real_i = i % len(self.real_samples_paths) - 1
        paths = self.samples_paths[i]
        paths["rx"] = self.real_samples_paths[real_i]["x"]
        paths["rm"] = self.real_samples_paths[real_i]["m"]
        paths["rd"] = self.real_samples_paths[real_i]["d"]

        if self.transform:
            # first perform transform on input images and then convert depth image
            # to depth array
            datadict = self.transform(
                {task: pil_image_loader(path, task) for task, path in paths.items()}
            )
            datadict["d"] = get_normalized_depth(
                datadict["d"], mode=self.opts.data.depth.sim_mode
            )
            datadict["rd"] = get_normalized_depth(
                datadict["rd"], mode=self.opts.data.depth.real_mode
            )
            return Dict({"data": datadict, "paths": paths})
        datadict = self.transform(
            {task: pil_image_loader(path, task) for task, path in paths.items()}
        )
        datadict["d"] = get_normalized_depth(
            np.array(datadict["d"]), mode=self.opts.depth.sim_mode
        )
        datadict["rd"] = get_normalized_depth(
            np.array(datadict["rd"]), mode=self.opts.depth.real_mode
        )

        return Dict({"data": datadict, "paths": paths})

    def __len__(self):
        return len(self.samples_paths)


class SimDepthDataset(Dataset):
    def __init__(self, opts, transform=None, no_check=False):

        isTrain = opts.model.is_train
        if isTrain:
            file_list_path = Path(opts.data.files.train)
        else:
            file_list_path = Path(opts.data.files.val)

        if file_list_path.suffix == ".json":
            self.samples_paths = self.json_load(file_list_path)
        elif file_list_path.suffix in {".yaml", ".yml"}:
            self.samples_paths = self.yaml_load(file_list_path)
        else:
            raise ValueError("Unknown file list type in {}".format(file_list_path))

        if not no_check:
            self.check_samples()
        self.file_list_path = str(file_list_path)
        self.transform = transform
        self.opts = opts

    def check_samples(self):
        """Checks that every file listed in samples_paths actually
        exist on the file-system
        """
        for s in self.samples_paths:
            for k, v in s.items():
                assert Path(v).exists(), f"{k} {v} does not exist"

    def json_load(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def yaml_load(self, file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def __getitem__(self, i):
        """Return an item in the dataset with fields:
        {
            data: transform({),
            paths: [{task: path}],
            mode: [train|val]
        }
        Args:
            i (int): index of item to retrieve
        Returns:
            dict: dataset item where tensors of data are in item["data"] which is a dict
                  {task: tensor}
        """
        paths = self.samples_paths[i]

        if self.transform:
            # first perform transform on input images and then convert depth image to depth array
            datadict = self.transform(
                {task: pil_image_loader(path, task) for task, path in paths.items()}
            )
            datadict["d"] = get_normalized_depth(
                np.array(datadict["d"]), mode=self.opts.depth.sim_mode
            )

            return Dict({"data": datadict, "paths": paths})
        datadict = self.transform(
            {task: pil_image_loader(path, task) for task, path in paths.items()}
        )
        datadict["d"] = get_normalized_depth(
            np.array(datadict["d"]), mode=self.opts.depth.sim_mode
        )

        return Dict({"data": datadict, "paths": paths})

    def __len__(self):
        return len(self.samples_paths)
