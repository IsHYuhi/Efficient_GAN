from collections import OrderedDict
from models.Generator import Generator
from models.Discriminator import Discriminator
from models.Encoder import Encoder
from utils.data_set import make_datapath_list, GAN_Img_Dataset, ImageTransform
from torchvision import models
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import torch
import os

torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #initialize Conv2d and ConvTranspose2d
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        #initialize BatchNorm2d
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:] # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

def plot_log(data, save_model_name='model'):
    plt.cla()
    plt.plot(data['G'], label='G_loss ')
    plt.plot(data['D'], label='D_loss ')
    plt.plot(data['E'], label='E_loss ')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.savefig('./logs/'+save_model_name+'.png')

def check_dir():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

def train_model(G, D, E, dataloader, num_epochs, save_model_name='model'):

    check_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    G.to(device)
    D.to(device)
    E.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G = torch.nn.DataParallel(G)
        D = torch.nn.DataParallel(D)
        E = torch.nn.DataParallel(E)
        print("parallel mode")

    print("device:{}".format(device))
    ge_lr, d_lr = 0.0001, 0.000025
    beta1, beta2 = 0.5, 0.999
    g_optimizer = torch.optim.Adam(G.parameters(), ge_lr, [beta1, beta2])
    d_optimizer = torch.optim.Adam(D.parameters(), d_lr, [beta1, beta2])
    e_optimizer = torch.optim.Adam(E.parameters(), ge_lr, [beta1, beta2])

    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    z_dim = 20
    mini_batch_size = 64

    G.train()
    D.train()
    E.train()

    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size

    iteration = 1
    logs = []

    g_losses = []
    d_losses = []
    e_losses = []
    losses = {'G':g_losses, 'D':d_losses, 'E':e_losses}

    for epoch in range(num_epochs+1):

        t_epoch_start = time.time()
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        epoch_e_loss = 0.0

        print('-----------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('(train)')

        for images in tqdm(dataloader):

            # Train Discriminator
            # if size of minibatch is 1, an error would be occured.
            if images.size()[0] == 1:
                continue

            images = images.to(device)

            mini_batch_size = images.size()[0]
            label_real = torch.full((mini_batch_size,), 1).to(device)
            label_fake = torch.full((mini_batch_size,), 0).to(device)

            z_out_real = E(images)
            d_out_real, _ = D(images, z_out_real)

            input_z = torch.randn(mini_batch_size, z_dim).to(device)# from 1d
            #input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)# from C, H, W
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images, input_z)

            d_loss_real = criterion(d_out_real.view(-1), label_real)
            d_loss_fake = criterion(d_out_fake.view(-1), label_fake)
            d_loss = d_loss_real + d_loss_fake


            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            input_z = torch.randn(mini_batch_size, z_dim).to(device)# from 1d
            #input_z = input_z.view(input_z.size(0), input_z.size(1), 1, 1)# from C, H, W
            fake_images = G(input_z)
            d_out_fake, _ = D(fake_images, input_z)

            g_loss = criterion(d_out_fake.view(-1), label_real)

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Train Encoder
            z_out_real = E(images)
            d_out_real, _ = D(images, z_out_real)

            e_loss = criterion(d_out_real.view(-1), label_fake)

            e_optimizer.zero_grad()
            e_loss.backward()
            e_optimizer.step()

            # Record
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_e_loss += e_loss.item()
            iteration += 1

        t_epoch_finish = time.time()
        print('-----------')
        print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f} || Epoch_E_Loss:{:.4f}'.format(epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size, epoch_e_loss/batch_size))
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        losses['G'].append(epoch_g_loss/batch_size)
        losses['D'].append(epoch_d_loss/batch_size)
        losses['E'].append(epoch_e_loss/batch_size)

        plot_log(losses, save_model_name)

        if(epoch%10 == 0):
            torch.save(G.state_dict(), 'checkpoints/G_'+save_model_name+'_'+str(epoch)+'.pth')
            torch.save(D.state_dict(), 'checkpoints/D_'+save_model_name+'_'+str(epoch)+'.pth')
            torch.save(E.state_dict(), 'checkpoints/E_'+save_model_name+'_'+str(epoch)+'.pth')

    return G, D, E

def main():
    G = Generator(z_dim=20)
    D = Discriminator(z_dim=20)
    E = Encoder(z_dim=20)
    G.apply(weights_init)
    D.apply(weights_init)
    E.apply(weights_init)

    train_img_list=make_datapath_list(num=200)
    mean = (0.5,)
    std = (0.5,)
    train_dataset = GAN_Img_Dataset(file_list=train_img_list, transform=ImageTransform(mean, std))

    batch_size = 64
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_epochs = 1500
    G_update, D_update, E_update = train_model(G, D, E, dataloader=train_dataloader, num_epochs=num_epochs, save_model_name='Efficient_GAN')


if __name__ == "__main__":
    main()