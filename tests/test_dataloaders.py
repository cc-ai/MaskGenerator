import os
from pathlib import Path
import argparse
import torch
import sys
from addict import Dict

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.networks import Generator, NLayerDiscriminator
from data.datasets import get_loader
from tests.run import opts

if __name__ == "__main__":
    loader = get_loader(opts, real=True)
    for data_dict in loader:
        data_dict = Dict(data_dict)
        print(data_dict.data.keys())

