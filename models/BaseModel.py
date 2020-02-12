from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.nn import functional as F

from models.blocks import Conv2dBlock, ConvTranspose2dBlock, ResBlocks


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()

        activ = opts.gen.encoder.activ
        dim = opts.gen.encoder.dim
        input_dim = opts.gen.encoder.input_dim
        n_downsample = opts.gen.encoder.n_downsample
        n_res = opts.gen.encoder.n_res
        enc_norm = opts.gen.encoder.norm
        pad_type = opts.gen.encoder.pad_type
        output_dim = opts.gen.decoder.output_dim
        output_activ = opts.gen.decoder.output_activ
        dec_norm = opts.gen.decoder.norm
        res_norm = opts.gen.encoder.res_norm

        # --------------------ENCODER----------------------------
        self.encoder = [Conv2dBlock(input_dim, dim, 7, 1, 3)]

        for i in range(n_downsample):
            self.encoder += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=enc_norm)]
            dim = 2 * dim

        self.res_blocks = ResBlocks(n_res, dim, norm="instance")
        self.encoder = nn.Sequential(*self.encoder)

        # --------------------DECODER----------------------------
        self.decoder = []
        for i in range(n_downsample):
            self.decoder += [ConvTranspose2dBlock(dim, int(dim / 2), 2, 2, 0, norm=dec_norm)]
            dim = int(dim / 2)
        self.decoder += [Conv2dBlock(dim, output_dim, 3, 1, 1)]

        self.decoder = nn.Sequential(*self.decoder)

    def encode(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        return x

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, input):
        # Encode spectrogram

        z = self.encode(input)
        x = self.decode(z)
        return x
