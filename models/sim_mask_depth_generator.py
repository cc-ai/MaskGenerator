import torch
import itertools
from .base_model import BaseModel
from . import networks
from utils import write_images
from addict import Dict


# To be used on simulated data only
class SimMaskDepthGenerator(BaseModel):
    def name(self):
        return "SimMaskDepthGeneratorModel"

    @staticmethod
    def modify_commandline_options(opts, is_train=True):
        print("modifying opts")
        return opts

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = []
        self.loss_name = opt.model.loss_name

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ["G", "D"]

            # self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            self.model_names = ["G"]
            # self.model_names = ["G_A", "G_B"]

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG = networks.define_G(opt).to(self.device)
        self.netD = networks.define_D(opt).to(self.device)
        self.comet_exp = opt.comet.exp
        self.store_image = opt.val.store_image
        self.overlay = opt.val.overlay
        self.opt = opt

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(self.loss_name).to(self.device)
            # initialize optimizers

            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG.parameters()),
                lr=opt.gen.opt.lr,
                betas=(opt.gen.opt.beta1, 0.999),
            )
            self.optimizer_D = torch.optim.Adam(
                itertools.chain(self.netD.parameters()),
                lr=opt.dis.opt.lr,
                betas=(opt.dis.opt.beta1, 0.999),
            )
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):

        self.image_depth = torch.cat([input.data.x.to(self.device), input.data.d.to(self.device)], dim = 1)
        self.image = input.data.x.to(self.device)
        mask = input.data.m.to(self.device)
        self.mask = mask[:, 0, :, :].unsqueeze(1)
        self.paths = input.paths

    def forward(self):
        self.latent_vec, self.fake_mask = self.netG(self.image_depth)

    def backward_D(self):
        # Real

        real_mask_d = torch.cat([self.image_depth, self.mask], dim=1)
        fake_mask_d = torch.cat([self.image_depth, self.fake_mask], dim=1)

        pred_real = self.netD(real_mask_d)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Fake
        pred_fake = self.netD(fake_mask_d.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        if self.loss_name == "wgan":  # Get gradient penalty loss
            grad_penalty = networks.calc_gradient_penalty(
                self.opt, self.netD, real_mask_d, fake_mask_d
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

    def backward_G(self):
        self.loss_G = self.criterionGAN(
            self.netD(torch.cat([self.image_depth, self.fake_mask], dim=1)), True
        )
        # Log G loss to comet:
        if self.comet_exp is not None:
            self.comet_exp.log_metric("loss G", self.loss_G.cpu().detach())
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        # G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        # D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

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
            save_depth = model.image_depth[0][3]

            if overlay:
                save_images.append(save_depth.repeat(3, 1, 1))
                save_images.append(save_mask)
                save_images.append(save_fake_mask)

            else:
                save_images.append(save_depth.repeat(3, 1, 1))
                save_images.append(self.mask[0].repeat(3, 1, 1))
                save_images.append(self.fake_mask[0].repeat(3, 1, 1))
        write_images(save_images, curr_iter, comet_exp=self.comet_exp, store_im=self.store_image)
        return(save_images)
