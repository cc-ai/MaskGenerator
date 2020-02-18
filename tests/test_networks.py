import os
from pathlib import Path
import argparse
import torch
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.networks import define_G, define_D
from tests.run import opts

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = define_G(opts)
    gen = gen.to(device)

    dis = define_D(opts)
    dis = dis.to(device)

    batch_size = 2
    test_image = torch.Tensor(batch_size, 3, 128, 128).uniform_(-1, 1).to(device)

    gen_output = gen(test_image)
    print("Testing generator...")
    print("Gen output shape: ", gen_output.shape)

    dis_output = dis(test_image)
    print("Testing discriminator...")
    print("Dis output shape: ", dis_output.shape)

