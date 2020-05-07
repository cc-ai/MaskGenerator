import torch
import itertools
from .base_model import BaseModel
from . import networks
from utils import write_images
from time import time
from ExtraAdam import ExtraAdam


class MaskDepthGenerator(BaseModel):
    def name(self):
        return "MaskDepthGeneratorModel"

    @staticmethod
    def modify_commandline_options(opts, is_train=True):
        print("modifying opts")
        return opts

    def initialize(self, opts):
        BaseModel.initialize(self, opts)

        # specify the training losses you want to print out.
        # The program will call base_model.get_current_losses
        self.loss_names = []
        self.loss_name = opts.model.loss_name

        # specify the models you want to save to the disk.
        # The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ["G", "D", "D_F", "D_P"]

            # self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G"]
            # self.model_names = ["G_A", "G_B"]

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG = networks.define_G(opts).to(self.device)
        self.netD = networks.define_D(opts).to(self.device)

        # Use the latent vector to define input_nc
        opts.dis.default.input_nc = (
            2 ** opts.gen.encoder.n_downsample
        ) * opts.gen.encoder.dim
        opts.dis.default.n_layers = opts.dis.feature_DA.n_layers
        self.netD_F = networks.define_D(opts).to(
            self.device
        )  # Feature domain adaptation discriminator

        opts.dis.default.input_nc = 1
        self.netD_P = networks.define_D(opts).to(
            self.device
        )  # Pixel domain adaptation discriminator

        self.comet_exp = opts.comet.exp
        self.store_image = opts.val.store_image
        self.overlay = opts.val.overlay
        self.opts = opts

        if self.isTrain:
            GenOpt = (
                ExtraAdam
                if "extra" in opts.gen.optim.optimizer.lower()
                else torch.optim.Adam
            )
            DisOpt = (
                ExtraAdam
                if "extra" in opts.dis.optim.optimizer.lower()
                else torch.optim.Adam
            )
            # define loss functions
            self.criterionGAN = networks.GANLoss(self.loss_name).to(self.device)
            # initialize optimizers

            self.optimizer_G = GenOpt(
                itertools.chain(self.netG.parameters()),
                lr=opts.gen.optim.lr,
                betas=(opts.gen.optim.beta1, 0.999),
            )
            self.optimizer_D = DisOpt(
                itertools.chain(self.netD.parameters()),
                lr=opts.dis.optim.lr,
                betas=(opts.dis.optim.beta1, 0.999),
            )
            self.optimizer_D_F = DisOpt(
                itertools.chain(self.netD_F.parameters()),
                lr=opts.dis.optim.lr,
                betas=(opts.dis.optim.beta1, 0.999),
            )
            self.optimizer_D_P = DisOpt(
                itertools.chain(self.netD_P.parameters()),
                lr=opts.dis.optim.lr,
                betas=(opts.dis.optim.beta1, 0.999),
            )
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_D_F)
            self.optimizers.append(self.optimizer_D_P)

    def set_input(self, input):
        # Sim data

        self.input = torch.cat(
            [
                input.data.x.to(self.device),
                input.data.d.type(torch.FloatTensor).to(self.device),
            ],
            dim=1,
        )
        self.image = input.data.x.to(self.device)
        mask = input.data.m.to(self.device)
        self.mask = mask[:, 0, :, :].unsqueeze(1)
        self.paths = input.paths

        # Real data
        self.r_input = torch.cat(
            [
                input.data.rx.to(self.device),
                input.data.rd.type(torch.FloatTensor).to(self.device),
            ],
            dim=1,
        )
        self.r_im = input.data.rx.to(self.device)
        self.r_mask = input.data.rm.to(self.device)  # From segmentation, or whatever

    def forward(self):
        self.sim_latent_vec, self.fake_mask = self.netG(self.input)
        self.real_latent_vec, self.r_fake_mask = self.netG(self.r_input)

    def backward_D(self, steps=0):
        # Real

        real_mask_d = torch.cat([self.input, self.mask], dim=1)
        fake_mask_d = torch.cat([self.input, self.fake_mask], dim=1)

        pred_real = self.netD(real_mask_d)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = self.netD(fake_mask_d.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        if self.loss_name == "wgan":  # Get gradient penalty loss
            grad_penalty = networks.calc_gradient_penalty(
                self.opts, self.netD, real_mask_d, fake_mask_d
            )
            self.loss_D += grad_penalty

        # Log D loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metrics(
                {
                    "loss D": self.loss_D.cpu().detach(),
                    "loss D real": self.loss_D_real.cpu().detach(),
                    "loss D fake": self.loss_D_fake.cpu().detach(),
                },
                step=steps,
            )

        # backward
        self.loss_D.backward()

    def backward_D_P(self, steps=0):
        # Real (sim)
        pred_domain_sim = self.netD_P(self.mask)
        self.loss_D_P_sim = self.criterionGAN(pred_domain_sim, True)

        # Fake (real)
        pred_domain_real = self.netD_P(self.r_fake_mask.detach())
        self.loss_D_P_real = self.criterionGAN(pred_domain_real, False)
        self.loss_D_P = (self.loss_D_P_sim + self.loss_D_P_real) * 0.5

        if self.loss_name == "wgan":  # Get gradient penalty loss
            grad_penalty = networks.calc_gradient_penalty(
                self.opts, self.netD_P, self.mask, self.r_fake_mask
            )
            self.loss_D_P += 0.5 + grad_penalty

        # Log D loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metrics(
                {
                    "loss D Pixel DA": self.loss_D_P.cpu().detach(),
                    "loss D Pixel DA sim": self.loss_D_P_sim.cpu().detach(),
                    "loss D Pixel DA real": self.loss_D_P_real.cpu().detach(),
                },
                step=steps,
            )

        # backward
        self.loss_D_P.backward()

    def backward_D_F(self, steps=0):
        # Feature Domain adaptation
        # Treat sim as "True" and real as "False"
        pred_domain_sim = self.netD_F(self.sim_latent_vec.detach())
        self.loss_D_F_sim = self.criterionGAN(pred_domain_sim, True)

        pred_domain_real = self.netD_F(self.real_latent_vec.detach())
        self.loss_D_F_real = self.criterionGAN(pred_domain_real, False)
        self.loss_D_F = (self.loss_D_F_sim + self.loss_D_F_real) * 0.5

        if self.loss_name == "wgan":  # Get gradient penalty loss
            grad_penalty = networks.calc_gradient_penalty(
                self.opts, self.netD_F, self.sim_latent_vec, self.real_latent_vec
            )
            self.loss_D_F += 0.5 + grad_penalty

        # Log D loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metrics(
                {
                    "loss D Feature DA": self.loss_D_F.cpu().detach(),
                    "loss D Feature DA sim": self.loss_D_F_sim.cpu().detach(),
                    "loss D Feature DA real": self.loss_D_F_real.cpu().detach(),
                },
                step=steps,
            )

        # backward
        self.loss_D_F.backward()

    def backward_G(self, steps=0):
        # Standard G loss
        self.loss_G_standard = self.criterionGAN(
            self.netD(torch.cat([self.input, self.fake_mask], dim=1)), True
        )

        # Domain adaptation feature loss
        self.loss_G_DA_F = (
            self.criterionGAN(self.netD_F(self.sim_latent_vec), False)
            + self.criterionGAN(self.netD_F(self.real_latent_vec), True)
        ) * 0.5

        # Domain adaptation pixel loss
        self.loss_G_DA_P = self.criterionGAN(self.netD_P(self.r_fake_mask), True)

        self.loss_G = self.loss_G_standard + (self.loss_G_DA_F + self.loss_G_DA_P) * 0.5
        # Log G loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metrics(
                {
                    "loss G": self.loss_G.cpu().detach(),
                    "loss G standard": self.loss_G_standard.cpu().detach(),
                    "loss G Feature DA": self.loss_G_DA_F.cpu().detach(),
                    "loss G Pixel DA": self.loss_G_DA_P.cpu().detach(),
                },
                step=steps,
            )

        self.loss_G.backward()

    def optimizer_G_step(self):
        bs = self.opts.data.loaders.batch_size
        if "extra" in self.opts.gen.optim.optimizer and self.curr_iter // bs % 2 == 0:
            self.optimizer_G.extrapolation()
        else:
            self.optimizer_G.step()

    def optimizer_D_step(self):
        bs = self.opts.data.loaders.batch_size
        if "extra" in self.opts.dis.optim.optimizer and self.curr_iter // bs % 2 == 0:
            self.optimizer_D.extrapolation()
        else:
            self.optimizer_D.step()

    def optimizer_D_F_step(self):
        bs = self.opts.data.loaders.batch_size
        if "extra" in self.opts.dis.optim.optimizer and self.curr_iter // bs % 2 == 0:
            self.optimizer_D_F.extrapolation()
        else:
            self.optimizer_D_F.step()

    def optimizer_D_P_step(self):
        bs = self.opts.data.loaders.batch_size
        if "extra" in self.opts.dis.optim.optimizer and self.curr_iter // bs % 2 == 0:
            self.optimizer_D_P.extrapolation()
        else:
            self.optimizer_D_P.step()

    def optimize_parameters(self, curr_iter=0):
        # forward
        self.curr_iter = curr_iter
        self.forward()

        # G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netD_F, False)
        self.set_requires_grad(self.netD_P, False)
        self.optimizer_G.zero_grad()
        self.backward_G(curr_iter)
        self.optimizer_G_step()

        # D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D(curr_iter)
        self.optimizer_D_step()

        # D Feature - Domain adaptation
        self.set_requires_grad(self.netD_F, True)
        self.optimizer_D_F.zero_grad()
        self.backward_D_F(curr_iter)
        self.optimizer_D_F_step()

        # D Pixel - Domain adaptation
        self.set_requires_grad(self.netD_P, True)
        self.optimizer_D_P.zero_grad()
        self.backward_D_P(curr_iter)
        self.optimizer_D_P_step()

    def normalize_depth_display(self, depth):
        n_depth = (depth - min(depth)) / (max(depth) - min(depth))
        return 255 * n_depth

    def set_input_display(self, input):
        # for image log
        # Sim data
        self.image = input.data.x.unsqueeze(0).to(self.device)
        self.depth = (
            self.normalize_depth_display(input.data.d).unsqueeze(0).to(self.device)
        )

        self.mask = input.data.m.unsqueeze(0).to(self.device)
        self.paths = input.paths

        # Real data
        self.r_im = input.data.rx.unsqueeze(0).to(self.device)
        self.r_depth = (
            self.normalize_depth_display(input.data.rd).unsqueeze(0).to(self.device)
        )

        self.r_mask = input.data.rm.unsqueeze(0).to(self.device)

    def save_test_images(self, test_display_data, curr_iter, is_test=True):
        st = time()
        overlay = self.overlay
        save_images = []
        for i in range(len(test_display_data)):
            self.set_input_display(test_display_data[i])
            self.test()
            save_images.append(self.image[0])
            # Overlay mask:
            save_mask = (
                self.image[0]
                - (self.image[0] * self.mask[0].repeat(3, 1, 1))
                + self.mask[0].repeat(3, 1, 1)
            )

            save_fake_mask = (
                self.image[0]
                - (self.image[0] * self.fake_mask[0].repeat(3, 1, 1))
                + self.fake_mask[0].repeat(3, 1, 1)
            )

            if overlay:
                save_images.append(save_mask)
                save_images.append(save_fake_mask)
            else:
                save_images.append(self.mask[0].repeat(3, 1, 1))
                save_images.append(self.fake_mask[0].repeat(3, 1, 1))

        for i in range(len(test_display_data)):
            # Append real masks (overlayed and itself):
            self.set_input_display(test_display_data[i])
            self.test()
            save_images.append(self.r_im[0])
            save_real_mask_seg = (
                self.r_im[0]
                - (self.r_im[0] * self.r_mask[0].repeat(3, 1, 1))
                + self.r_mask[0].repeat(3, 1, 1)
            )
            save_real_mask = (
                self.r_im[0]
                - (self.r_im[0] * self.r_fake_mask[0].repeat(3, 1, 1))
                + self.r_fake_mask[0].repeat(3, 1, 1)
            )
            save_images.append(save_real_mask_seg)
            save_images.append(save_real_mask)
        write_images(
            save_images,
            curr_iter,
            comet_exp=self.comet_exp,
            store_im=self.store_image,
            is_test=is_test,
        )

        return time() - st

