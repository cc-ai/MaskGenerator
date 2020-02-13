import os
from pathlib import Path
import argparse
import torch
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from models.networks import Generator, NLayerDiscriminator
from data.datasets import SimDataset
from tests.run import opts

if __name__ == "__main__":
    dataset = SimDataset("train", opts)
