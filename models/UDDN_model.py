import torch
import itertools, os
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .ssim import *
from torch.cuda.amp import autocast, GradScaler
from .networks import init_net
from .UDDN import Dual_Domain_GEN


class UDDNModel(BaseModel):
    """
    This class implements the UDDN model, an unsupervised method to remove motion artifacts in MRI without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.


        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the UDDN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'art']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        if self.isTrain:
            visual_names_A = ['real_A', 'fake_B', 'rec_A']
            visual_names_B = ['real_B', 'fake_A', 'rec_B']
        else:
            visual_names_A = ['real_A', 'fake_B']
            visual_names_B = ['real_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G']

        self.amp = False  # use mix precision

        self.netG = Dual_Domain_GEN()
        init_net(self.netG, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCycle1 = torch.nn.L1Loss()
            self.criterionCycle2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = hvd.DistributedOptimizer(self.optimizer_G,named_parameters=self.netG.named_parameters())
            if self.amp:
                self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            else:
                self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D_A)

                self.optimizers.append(self.optimizer_D_B)
            if self.amp:
                self.amp_scaler = GradScaler(enabled=True)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def forward(self):
        self.fake_B, self.fake_A = self.netG.forward1(self.real_A, self.real_B)
        self.rec_B, self.rec_A = self.netG.forward1(self.fake_A, self.fake_B)

    def return_img(self):
        return self.real_A, self.real_B, self.fake_B

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        if self.amp:
            self.amp_scaler.scale(loss_D).backward()
        else:
            loss_D.backward()

        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_B, self.idt_A = self.netG.forward2(self.real_A, self.real_B)
            # self.idt_A,_ = self.netG.forward1(self.real_A)  #my_network
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            # self.idt_B,_ =  self.netG.forward2(self.real_A,self.real_B)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # self.loss_G_A = (self.criterionGAN(self.netD_A(self.fake_B), True) + self.criterionGAN(self.netD_A(self.fake_B_cro), True)) *0.5
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # self.loss_G_B = (self.criterionGAN(self.netD_B(self.fake_A), True) + self.criterionGAN(self.netD_A(self.fake_A_cro), True)) * 0.5
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B

        # self.loss_art1 = self.criterionCycle1(self.fake_B, self.real_A) * 10
        # self.loss_art2 = self.criterionCycle1(self.fake_A, self.real_B) * 10
        # self.loss_art = self.criterionCycle1((self.real_A - self.fake_B), (self.fake_A - self.real_B)) * 5

        # #ssim loss
        if self.amp:
            self.real_A = self.real_A.half()
            self.real_B = self.real_B.half()
        self.loss_ssim = (1 - ms_ssim(self.real_A, self.rec_A)) + (1 - ms_ssim(self.real_B, self.rec_B)) * 10

        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_ssim + self.loss_idt_B + self.loss_art  # self.loss_vgg_A + self.loss_vgg_B+self.loss_ssim+
        if self.amp:
            self.amp_scaler.scale(self.loss_G).backward()
        else:

            self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero

        if self.amp:
            with autocast(enabled=True):

                self.forward()
                self.backward_G()
            self.amp_scaler.unscale_(self.optimizer_G)
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 1.0)
            self.amp_scaler.step(self.optimizer_G)



            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D.zero_grad()
            with autocast(enabled=True):
                self.backward_D_A()  # calculate gradients for D_A
                self.backward_D_B()  # calculate graidents for D_B
            # self.amp_scaler.scale(self.loss_D_B).backward()

            self.amp_scaler.unscale_(self.optimizer_D)
            torch.nn.utils.clip_grad_norm_(self.netD_A.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.netD_B.parameters(), 1.0)
            self.amp_scaler.step(self.optimizer_D)

            self.amp_scaler.update()
            #torch.autograd.set_detect_anomaly(True)
        else:
            self.backward_G()  # calculate gradients for G_A and G_B
            self.optimizer_G.step()  # update G_A and G_B's weights
            # D_A and D_B
            self.set_requires_grad([self.netD_A, self.netD_B], True)
            self.optimizer_D_A.zero_grad()  # set D_A and D_B's gradients to zero
            self.optimizer_D_B.zero_grad()  # set D_A and D_B's gradients to zero
            self.backward_D_A()  # calculate gradients for D_A
            self.backward_D_B()  # calculate graidents for D_B
            self.optimizer_D_A.step()  # update D_A and D_B's weights
            self.optimizer_D_B.step()  # update