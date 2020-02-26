import os
from pathlib import Path
import argparse
import torch
import sys
import math

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
    test_image = torch.Tensor(batch_size, 3, 256, 256).uniform_(-1, 1).to(device)

    latent_vec, gen_output = gen(test_image)
    print("Testing generator...")
    print("Gen output shape: ", gen_output.shape)
    print("Latent vec shape: ", latent_vec.shape)

    test_image = torch.Tensor(batch_size, 4, 128, 128).uniform_(-1, 1).to(device)
    dis_output = dis(test_image)
    print("Testing discriminator...")
    print("Dis output shape: ", dis_output.shape)

    print("Testing sim/real feature discriminator...")
    opts.dis.default.input_nc = (2 ** opts.gen.encoder.n_downsample) * opts.gen.encoder.dim
    print(opts.dis.default.input_nc)
    dis = define_D(opts)
    dis = dis.to(device)

    test_image = (
        torch.Tensor(batch_size, opts.dis.default.input_nc, 128, 128).uniform_(-1, 1).to(device)
    )
    dis_output = dis(test_image)
    print("Dis Feature DA output shape: ", dis_output.shape)
