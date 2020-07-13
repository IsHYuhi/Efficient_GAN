from utils.data_set import GAN_Img_Dataset, ImageTransform
from collections import OrderedDict
from models.Generator import Generator
from models.Discriminator import Discriminator
from models.Encoder import Encoder
from utils.Anomaly_score import Anomaly_score
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import torch
import os

def make_datapath_list(root_path="./data/img_28_test", num=5):
    """
    make filepath list for train and validation image and annotation.
    """

    #numberごとに均一枚数を取得
    train_img_list = []
    for img_idx in range(num):
        img_path = root_path + "/img_" + str(7) + "_" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

        img_path = root_path + "/img_" + str(8) + "_" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

        img_path = root_path + "/img_" + str(2) + "_" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

    return train_img_list

def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:] # remove 'module.' of dataparallel
        new_state_dict[name] = v
    return new_state_dict

#torch.manual_seed(44)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


device = "cuda" if torch.cuda.is_available() else "cpu"
G = Generator(z_dim=20)
D = Discriminator(z_dim=20)
E = Encoder(z_dim=20)
'''-------load weights-------'''
G_load_weights = torch.load('./checkpoints/G_Efficient_GAN_1500.pth')
G.load_state_dict(fix_model_state_dict(G_load_weights))

D_load_weights = torch.load('./checkpoints/D_Efficient_GAN_1500.pth')
D.load_state_dict(fix_model_state_dict(D_load_weights))

E_load_weights = torch.load('./checkpoints/E_Efficient_GAN_1500.pth')
E.load_state_dict(fix_model_state_dict(E_load_weights))

G.to(device)
D.to(device)
E.to(device)

"""use GPU in parallel"""
if device == 'cuda':
    G = torch.nn.DataParallel(G)
    D = torch.nn.DataParallel(D)
    E = torch.nn.DataParallel(E)
    print("parallel mode")


batch_size = 5

train_img_list = make_datapath_list(num=5)
mean = (0.5,)
std = (0.5,)
train_dataset = GAN_Img_Dataset(file_list=train_img_list, transform=ImageTransform(mean, std))

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

batch_iterator = iter(train_dataloader)

# fetch first element
images = next(batch_iterator)



x = images[0:5]
x = x.to(device)

z_out_real = E(images.to(device))
images_reconstract = G(z_out_real)

loss, loss_each, residual_loss_each = Anomaly_score(x, images_reconstract, z_out_real, D, Lambda=0.1)

loss_each = loss_each.cpu().detach().numpy()
print("total loss：", np.round(loss_each, 0))


fig = plt.figure(figsize=(15, 6))
for i in range(0, 5):
    # testdata
    plt.subplot(2, 5, i+1)
    plt.imshow(images[i][0].cpu().detach().numpy(), 'gray')

    # generated
    plt.subplot(2, 5, 5+i+1)
    plt.imshow(images_reconstract[i][0].cpu().detach().numpy(), 'gray')

plt.show()
