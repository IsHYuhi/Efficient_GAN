import os
import glob
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

def make_datapath_list(root_path="./data/img_28", num=200):
    """
    make filepath list for train and validation image and annotation.
    """

    train_img_list = []
    for img_idx in range(num):
        img_path = root_path + "/img_" + str(7) + "_" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

        img_path = root_path + "/img_" + str(8) + "_" + str(img_idx) + '.jpg'
        train_img_list.append(img_path)

    return train_img_list

class ImageTransform():
    """
    preprocessing images
    """

    def __init__(self, mean, std):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, img):
        return self.data_transform(img)


class GAN_Img_Dataset(data.Dataset):
    """
    Dataset class. Inherit Dataset class from PyTrorch.
    """

    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        '''
        get tensor type preprocessed Image
        '''

        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img)

        return img_transformed
