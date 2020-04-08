import torch
import itertools
from .base_model import BaseModel
from . import networks
from utils import write_images


# Domain adaptation for real data
class MaskGenerator(BaseModel):
    def name(self):
        return "MaskGeneratorModel"

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
            # define loss functions
            self.criterionGAN = networks.GANLoss(self.loss_name).to(self.device)
            # initialize optimizers

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG.parameters()),
                lr=opts.gen.optim.lr,
                betas=(opts.gen.optim.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD.parameters()),
                lr=opts.dis.optim.lr,
                betas=(opts.dis.optim.beta1, 0.999),
            )
            self.optimizer_D_F = torch.optim.Adam(
                itertools.chain(self.netD_F.parameters()),
                lr=opts.dis.optim.lr,
                betas=(opts.dis.optim.beta1, 0.999),
            )
            self.optimizer_D_P = torch.optim.Adam(
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
        self.image = input.data.x.to(self.device)
        mask = input.data.m.to(self.device)
        self.mask = mask[:, 0, :, :].unsqueeze(1)
        self.paths = input.paths

        # Real data
        self.r_im = input.data.rx.to(self.device)
        self.r_mask = input.data.rm.to(self.device)  # From segmentation, or whatever

    def forward(self):
        self.sim_latent_vec, self.fake_mask = self.netG(self.image)
        self.real_latent_vec, self.r_fake_mask = self.netG(self.r_im)

    def backward_D(self):
        # Real

        real_mask_d = torch.cat([self.image, self.mask], dim=1)
        fake_mask_d = torch.cat([self.image, self.fake_mask], dim=1)

        pred_real = self.netD(real_mask_d)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = self.netD(fake_mask_d.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        if self.loss_name == "wgan":  # Get gradient penalty loss
            grad_penalty = networks.calc_gradient_penalty(
                self.opts, self.netD, real_mask_d, fake_mask_d
            )
            self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5 + grad_penalty
        else:
            # Combined loss
            self.loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5

        # Log D loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metric("loss D", self.loss_D.cpu().detach())
            self.comet_exp.log_metric("loss D real", self.loss_D_real.cpu().detach())
            self.comet_exp.log_metric("loss D fake", self.loss_D_fake.cpu().detach())

        # backward
        self.loss_D.backward()

    def backward_D_P(self):
        # Real (sim)
        pred_domain_sim = self.netD_P(self.mask)
        self.loss_D_P_sim = self.criterionGAN(pred_domain_sim, True)

        # Fake (real)
        pred_domain_real = self.netD_P(self.r_fake_mask.detach())
        self.loss_D_P_real = self.criterionGAN(pred_domain_real, False)

        if self.loss_name == "wgan":  # Get gradient penalty loss
            grad_penalty = networks.calc_gradient_penalty(
                self.opts, self.netD_P, self.mask, self.r_fake_mask
            )
            self.loss_D_P = (
                self.loss_D_P_sim + self.loss_D_P_real
            ) * 0.5 + grad_penalty
        else:
            # Combined loss
            self.loss_D_P = (self.loss_D_P_sim + self.loss_D_P_real) * 0.5

        # Log D loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metric("loss D Pixel DA", self.loss_D_P.cpu().detach())
            self.comet_exp.log_metric(
                "loss D Pixel DA sim", self.loss_D_P_sim.cpu().detach()
            )
            self.comet_exp.log_metric(
                "loss D Pixel DA real", self.loss_D_P_real.cpu().detach()
            )

        # backward
        self.loss_D_P.backward()

    def backward_D_F(self):
        # Feature Domain adaptation
        # Treat sim as "True" and real as "False"
        pred_domain_sim = self.netD_F(self.sim_latent_vec.detach())
        self.loss_D_F_sim = self.criterionGAN(pred_domain_sim, True)

        pred_domain_real = self.netD_F(self.real_latent_vec.detach())
        self.loss_D_F_real = self.criterionGAN(pred_domain_real, False)
        ###
        if self.loss_name == "wgan":  # Get gradient penalty loss
            grad_penalty = networks.calc_gradient_penalty(
                self.opts, self.netD_F, self.sim_latent_vec, self.real_latent_vec
            )
            self.loss_D_F = (
                self.loss_D_F_sim + self.loss_D_F_real
            ) * 0.5 + grad_penalty
        else:
            # Combined loss
            self.loss_D_F = (self.loss_D_F_sim + self.loss_D_F_real) * 0.5

        # Log D loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metric("loss D Feature DA", self.loss_D_F.cpu().detach())
            self.comet_exp.log_metric(
                "loss D Feature DA sim", self.loss_D_F_sim.cpu().detach()
            )
            self.comet_exp.log_metric(
                "loss D Feature DA real", self.loss_D_F_real.cpu().detach()
            )

        # backward
        self.loss_D_F.backward()

    def backward_G(self):
        # Standard G loss
        self.loss_G_standard = self.criterionGAN(
            self.netD(torch.cat([self.image, self.fake_mask], dim=1)), True
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
            self.comet_exp.log_metric("loss G", self.loss_G.cpu().detach())
            self.comet_exp.log_metric(
                "loss G standard", self.loss_G_standard.cpu().detach()
            )
            self.comet_exp.log_metric(
                "loss G Feature DA", self.loss_G_DA_F.cpu().detach()
            )
            self.comet_exp.log_metric(
                "loss G Pixel DA", self.loss_G_DA_P.cpu().detach()
            )
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        # G
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netD_F, False)
        self.set_requires_grad(self.netD_P, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # D Feature - Domain adaptation
        self.set_requires_grad(self.netD_F, True)
        self.optimizer_D_F.zero_grad()
        self.backward_D_F()
        self.optimizer_D_F.step()

        # D Pixel - Domain adaptation
        self.set_requires_grad(self.netD_P, True)
        self.optimizer_D_P.zero_grad()
        self.backward_D_P()
        self.optimizer_D_P.step()

    def save_test_images(self, test_display_data, curr_iter):
        overlay = self.overlay
        save_images = []
        for i in range(len(test_display_data)):
            self.set_input(test_display_data[i])
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
            self.set_input(test_display_data[i])
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
            save_images, curr_iter, comet_exp=self.comet_exp, store_im=self.store_image
        )
