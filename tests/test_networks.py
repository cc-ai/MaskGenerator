import os
from pathlib import Path
import argparse
import torch
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from models.BaseModel import Generator
from tests.run import opts

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = Generator(opts).to(device)

    batch_size = 2
    test_image = torch.Tensor(batch_size, 3, 128, 128).uniform_(-1, 1)

    output = gen(test_image)
    print("Testing generator...")
    print("Output shape: ", output.shape)

