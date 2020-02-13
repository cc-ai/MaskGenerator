from pathlib import Path
import yaml
import json
from torch.utils.data import DataLoader, Dataset
from imageio import imread
from torchvision import transforms
import numpy as np
from .transforms import get_transforms
from PIL import Image


class SimDataset(Dataset):
    def __init__(self, mode, opts, transform=None):

        file_list_path = Path(opts.data.files.base)
        print(file_list_path)
        '''

        self.check_samples()
        self.file_list_path = str(file_list_path)
        self.transform = transform


    def __getitem__(self, i):
        """Return an item in the dataset with fields:
        {
            data: transform({
                domains: values
            }),
            paths: [{task: path}],
            domain: [domain],
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
            return {
                "data": self.transform(
                    {task: pil_image_loader(path, task) for task, path in paths.items()}
                ),
                "paths": paths,
                "domain": self.domain,
                "mode": self.mode,
            }

        return {
            "data": {task: pil_image_loader(path, task) for task, path in paths.items()},
            "paths": paths,
            "domain": self.domain,
            "mode": self.mode,
        }

    def __len__(self):
        return len(self.samples_paths)

    def json_load(self, file_path):
        with open(file_path, "r") as f:
            return json.load(f)

    def yaml_load(self, file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    def check_samples(self):
        """Checks that every file listed in samples_paths actually
        exist on the file-system
        """
        for s in self.samples_paths:
            for k, v in s.items():
                assert Path(v).exists(), f"{k} {v} does not exist"


def get_loader(domain, mode, opts):
    return DataLoader(
        OmniListDataset(domain, mode, opts, transform=transforms.Compose(get_transforms(opts))),
        batch_size=opts.data.loaders.get("batch_size", 4),
        # shuffle=opts.data.loaders.get("shuffle", True),
        shuffle=True,
        num_workers=opts.data.loaders.get("num_workers", 8),
    )


def get_all_loaders(opts):
    loaders = {}
    for mode in ["train", "val"]:
        loaders[mode] = {}
        for domain in ["rf", "rn", "sf", "sn"]:
            if mode in opts.data.files:
                if domain in opts.data.files[mode]:
                    loaders[mode][domain] = get_loader(mode, domain, opts)
    return loaders
'''
