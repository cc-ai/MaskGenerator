from __future__ import absolute_import
import torch
import torch.nn as nn
import functools
from torch.nn import init
from models.blocks import Conv2dBlock, ConvTranspose2dBlock, ResBlocks
import torch.autograd as autograd


###############################################################################
# Helper Functions
######################### ######################################################
def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print("initialize network with %s" % init_type)
    net.apply(init_func)


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def get_norm_layer(norm_type="instance"):
    if not norm_type:
        print("norm_type is {}, defaulting to instance")
        norm_type = "instance"
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


# def WassersteinLoss(input, target_tensor):
#    #Use 0.5 threshold to determine whether real or fake


def calc_gradient_penalty(opts, netD, real_data, fake_data):
    DIM = opts.data.img_size
    LAMBDA = 10
    nc = opts.dis.default.input_nc
    alpha = torch.rand(real_data.shape)
    # alpha = alpha.view(batch_size, nc, DIM, DIM)
    # alpha = alpha.expand(
    #     batch_size, int(real_data.nelement() / batch_size)
    # ).contiguous()

    alpha = alpha.cuda()
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


class GANLoss(nn.Module):
    def __init__(self, loss_name="lsgan", target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer("real_label", torch.tensor(target_real_label))
        self.register_buffer("fake_label", torch.tensor(target_fake_label))
        self.loss_name = loss_name
        if loss_name == "lsgan":
            self.loss = nn.MSELoss()
        elif loss_name == "wgan":
            self.loss = lambda x: torch.mean(x)
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if self.loss_name == "wgan":
            if target_is_real:  # Miminize value
                return self.loss(input)
            else:  # Maximize value
                return -self.loss(input)
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.loss(input, target_tensor)


def define_G(opts):
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

    init_type = opts.gen.optim.init_type
    init_gain = opts.gen.optim.init_gain

    net = None

    net = Generator(
        activ,
        dim,
        input_dim,
        n_downsample,
        n_res,
        enc_norm,
        pad_type,
        output_dim,
        output_activ,
        dec_norm,
        res_norm,
    )

    init_weights(net, init_type, init_gain)

    return net


def define_D(opts):
    input_nc = opts.dis.default.input_nc
    ndf = opts.dis.default.ndf
    n_layers = opts.dis.default.n_layers
    norm_layer = get_norm_layer(opts.dis.default.norm)
    use_sigmoid = opts.dis.default.use_sigmoid
    kw = opts.dis.default.kw
    padw = opts.dis.default.padw
    nf_mult = opts.dis.default.nf_mult
    nf_mult_prev = opts.dis.default.nf_mult_prev

    init_type = opts.gen.optim.init_type
    init_gain = opts.gen.optim.init_gain

    net = None

    net = NLayerDiscriminator(
        input_nc,
        ndf,
        n_layers,
        norm_layer,
        use_sigmoid,
        kw,
        padw,
        nf_mult,
        nf_mult_prev,
    )

    init_weights(net, init_type, init_gain)

    return net


##############################################################################
# Classes
##############################################################################


class Generator(nn.Module):
    def __init__(
        self,
        activ,
        dim,
        input_dim,
        n_downsample,
        n_res,
        enc_norm,
        pad_type,
        output_dim,
        output_activ,
        dec_norm,
        res_norm,
    ):
        super(Generator, self).__init__()
        # --------------------ENCODER----------------------------
        self.encoder = [Conv2dBlock(input_dim, dim, 7, 1, 3)]

        for i in range(n_downsample):
            self.encoder += [
                Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=enc_norm, activation=activ)
            ]
            dim = 2 * dim

        self.res_blocks = ResBlocks(n_res, dim, norm=res_norm)
        self.encoder = nn.Sequential(*self.encoder)

        # --------------------DECODER----------------------------
        self.decoder = []
        for i in range(n_downsample):
            self.decoder += [
                ConvTranspose2dBlock(
                    dim, int(dim / 2), 2, 2, 0, norm=dec_norm, activation=activ
                )
            ]
            dim = int(dim / 2)
        self.decoder += [Conv2dBlock(dim, output_dim, 3, 1, 1, activation=output_activ)]

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
        return z, x


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        ndf,
        n_layers,
        norm_layer,
        use_sigmoid,
        kw,
        padw,
        nf_mult,
        nf_mult_prev,
    ):
        super(NLayerDiscriminator, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        sequence = [
            # Use spectral normalization
            SpectralNorm(
                nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)
            ),
            nn.LeakyReLU(0.2, True),
        ]

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                # Use spectral normalization
                SpectralNorm(
                    nn.Conv2d(
                        ndf * nf_mult_prev,
                        ndf * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=use_bias,
                    )
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            # Use spectral normalization
            SpectralNorm(
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=1,
                    padding=padw,
                    bias=use_bias,
                )
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        # Use spectral normalization
        sequence += [
            SpectralNorm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
            )
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
