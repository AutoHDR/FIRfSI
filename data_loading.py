import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import glob
import cv2
from random import randint
import random
import os
from skimage import io
from PIL import Image
import pandas as pd
from random import shuffle
from utils_tools import save_image, denorm, get_normal_in_range
import numpy as np

IMAGE_SIZE = 256

def generate_data_S_Img1024_csv(dir):

    floder = dir + '/*'
    floderlist = glob.glob(floder)
    floderlist.sort()
    ALLList = []
    for i in range(len(floderlist)):
        tp = glob.glob(floderlist[i] + '/*')
        ALLList += tp
        print()



    test_flag = int(len(ALLList)*0.2)
    face_list_1_test = ALLList[0:test_flag:1]
    name_to_list = {'face_1': face_list_1_test}
    df = pd.DataFrame(data=name_to_list)
    df.to_csv(dir+'../Img1024_test.csv')

    face_list_1_train = ALLList[test_flag:len(ALLList):1]
    name_to_list = {'face_1': face_list_1_train}
    df = pd.DataFrame(data=name_to_list)
    df.to_csv(dir+'../Img1024_train.csv')

    print('train and test csv file is saved.')

def get_dataset_Img1024_Sin(syn_dir=None, read_from_csv=None, read_celeba_csv=None, read_first=None, validation_split=0):

    df = pd.read_csv(read_from_csv)
    df = df[:read_first]
    face = list(df['face'])
    normal = list(df['normal'])
    mask = list(df['mask'])

    dataset_size = len(face)
    validation_count = int(validation_split * dataset_size / 100)
    train_count = dataset_size - validation_count

    # Build custom datasets
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    full_dataset = SfSNetDataset_Img1024_Sin(face, normal, mask, transform)
    # TODO: This will vary dataset run-to-run
    # Shall we just split manually to ensure run-to-run train-val dataset is same?
    train_dataset, val_dataset = random_split(full_dataset, [train_count, validation_count])
    return train_dataset, val_dataset

class SfSNetDataset_Img1024_Sin(Dataset):
    def __init__(self, face, normal, mask, transform=None):

        self.face = face
        self.mask = mask
        self.normal =normal
        self.transform = transform
        self.dataset_len = len(self.face)
        self.mask_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])
        self.normal_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        tpF = self.face[index]
        tpN = self.normal[index]
        tpM = self.mask[index]
        face = self.transform(Image.open(tpF))
        mask = self.transform(Image.open(tpM))[0,:,:].reshape([1,IMAGE_SIZE,IMAGE_SIZE])
        normal = self.transform(Image.open(tpN))
        normal = 2.0*(normal - 0.5)

        return face, normal, mask, tpF

    def __len__(self):
        return self.dataset_len


