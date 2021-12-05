import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import sys  
sys.path.insert(0, '../../')
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=40, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--num_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--ds_num", type=int, default=1, help="largest dataset is which")
parser.add_argument("--mixture", type=int, default=1, help="mixture")
opt = parser.parse_args()
print(opt)


os.makedirs("images_cs229", exist_ok=True)
os.makedirs(f"images_cs229/ds_100", exist_ok=True)
os.makedirs(f"images_cs229/ds_{opt.ds_num+1}", exist_ok=True)
os.makedirs("models", exist_ok=True)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.num_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), 
                     nn.LeakyReLU(0.4, inplace=True), 
                     nn.Dropout2d(0.5)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), 
                                       nn.Sigmoid())
        
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 
                                                 opt.num_classes + 1), 
                                       nn.Softmax())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity, label
    


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)

transform = transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor()]
)

# load the training data
mnist_train = datasets.FashionMNIST('../../data/mnist', train=True, download=True,
                                    transform = transform)

mnist_test = datasets.FashionMNIST('../../data/mnist', train=False, download=True,
                                    transform = transform)

mnist_train = list(mnist_train)
mnist_test = list(mnist_test)

test_size = [0.9, 0.5, 0.1]
std_list = [0.1, 0.2, 0.1]

item = {
    0: ['T-shirt', 'Top'],
    1: ['Trouser', 'Bottom'],
    2: ['Pullover', 'Top'],
    3: ['Dress', 'Bottom'],
    4: ['Coat', 'Top'],
    5: ['Sandal', 'Shoe'],
    6: ['Shirt', 'Top'],
    7: ['Sneaker', 'Shoe'],
    8: ['Bag', 'Bag'],
    9: ['Ankle boot', 'Shoe']
}


# biased toward target
category_wt1 = {
    'Top': [0.5],
    'Bottom': [0.5],
    'Shoe': [0.9],
    'Bag': [0.9]
}

# unbiased
category_wt2 = {
    'Top': [0.9],
    'Bottom': [0.9],
    'Shoe': [0.9],
    'Bag': [0.9]
}

# biased toward non-target
category_wt3 = {
    'Top': [0.9],
    'Bottom': [0.9],
    'Shoe': [0.3],
    'Bag': [0.3]
}

category_wt = [category_wt1, category_wt2, category_wt3]



(obs_X_list, obs_y_list, 
 nonobs_X_list, nonobs_y_list) = generate_data_mixture(0, mnist_train, 
                                                       test_size, 
                                                       std_list, 
                                                       category_wt, 
                                                       item, opt.batch_size)

if opt.mixture == 1:
    (trainX_largest, trainy_largest, 
     trainX_rest_largest, trainy_rest_largest) = create_mixture(obs_X_list, 
                                                            obs_y_list, 
                                                            nonobs_X_list, 
                                                            nonobs_y_list)
    opt.ds_num = 99
        
else:
    (trainX_largest, trainy_largest, 
     trainX_rest_largest, trainy_rest_largest) = create_largest(obs_X_list, 
                                                            obs_y_list, 
                                                            nonobs_X_list, 
                                                            nonobs_y_list, 
                                                            opt.ds_num)

# batch_size = 84
indexes = torch.randperm(trainX_largest.shape[0])
trainX_largest = trainX_largest[indexes]
trainy_largest = trainy_largest[indexes]
dataloader = loaders(trainX_largest, trainy_largest, opt.batch_size)[0]
train_rest_loader = loaders(trainX_rest_largest, trainy_rest_largest, opt.batch_size)[0]

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------
G_list = []
D_list = []
acc_list = []
acc_fake_real_list = []
real_share_list = []
pred_real_share_list = []
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        fake_aux_gt = Variable(LongTensor(batch_size).fill_(opt.num_classes), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        validity, _ = discriminator(gen_imgs)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels))

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, fake_aux_gt))
        
        # print(f'real:{auxiliary_loss(real_aux, labels)}, fake:{auxiliary_loss(fake_aux, fake_aux_gt)}')

        # Total discriminator loss
        d_loss = (d_real_loss*3 + d_fake_loss) / 4

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        pred_adv = np.concatenate([real_pred.data.cpu().numpy(), fake_pred.data.cpu().numpy()], axis=0)
        
        gt = np.concatenate([labels.data.cpu().numpy(), fake_aux_gt.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)
        
        gt_real_fake = np.concatenate([valid.data.cpu().numpy(), fake.data.cpu().numpy()], axis=0)
        d_real_fake_acc = np.mean(np.argmax(pred_adv, axis=1) == gt_real_fake)
        
        pred_aux_real = np.argmax(pred, axis=1)
        # print(pred_aux_real, np.sum((pred_aux_real==0)*1), len(real_aux), labels.data.cpu().numpy())
        real_share = float(np.sum((labels.data.cpu().numpy()==0)*1)/len(labels))
        pred_real_share = float(np.sum((pred_aux_real==0)*1)/len(pred_aux_real))

        d_loss.backward()
        optimizer_D.step()
        
        if i % 100 == 0:
            print('===== discriminator classificer =====')
            print(gt, gt.shape)
            print(np.argmax(pred, axis=1))
            # print('===== fake_real classificer =====')
            # print(gt_real_fake, gt_real_fake.shape)
            # print(np.argmax(pred_adv, axis=1))
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%, real_fake_acc: %d%%] [G loss: %f] \
                 [real target share: %d%%, predict target share: %d%%, ]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 
                   100 * d_acc, 100 * d_real_fake_acc, g_loss.item(),
                   real_share*100, pred_real_share*100)
            )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], f"images_cs229/ds_{opt.ds_num+1}/%d.png" % batches_done, 
                       nrow=5, normalize=True)
    
    G_list.append(float(g_loss.item()))
    D_list.append(float(d_loss.item()))
    acc_list.append(float(d_acc))
    acc_fake_real_list.append(float(d_real_fake_acc))
    real_share_list.append(float(real_share))
    pred_real_share_list.append(float(pred_real_share))
            
torch.save(generator.state_dict(), f'models/generator_torch_single_with_noise_100_bias_ds{opt.ds_num+1}_90.pth')
torch.save(discriminator.state_dict(), f'models/discriminator_single_with_noise_100_bias_ds{opt.ds_num+1}_90.pth')    


data = {'Data_Set': [opt.ds_num+1]*opt.n_epochs,
        'Iter': range(opt.n_epochs), 
        'Generator_Loss': G_list,
        'Discriminator_Loss': D_list, 
        'Accuracy': acc_list, 
        'Fake_Real_Accuracy': acc_fake_real_list, 
        'Real_Share': real_share_list, 
        'Pred_Real_Share': pred_real_share_list}

data = pd.DataFrame(data)

data.to_csv(f'results/loss_torch_single_bias_ds{opt.ds_num+1}.csv')

