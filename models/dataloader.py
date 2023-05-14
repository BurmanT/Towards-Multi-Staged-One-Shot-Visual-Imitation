from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import pickle as pkl

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class TrajectoryDataset(Dataset):
    def __init__(self, traj_dir, transform=None):
        self.traj_dir = traj_dir
        self.transform = transform
        self.data = sorted(glob.glob(traj_dir + 'traj0*.pkl'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.traj = self.data[idx]
        f = open(self.traj, 'rb')
        data = pkl.load(f)
        poses = []
        for iter in range(data['len']):
            eef = data[str(iter)]['obs']['eef_pos']
            poses.append(eef)
        hand_loc = poses[0]
        start_img = data['0']['obs']['image']
        end_img = data[str(len(poses) - 1)]['obs']['image']
        # print(self.traj, torch.Tensor(poses).shape)

        sample= {'start_img': self.transform(start_img), 'end_img': self.transform(end_img), 
                 'hand_loc': torch.Tensor(hand_loc), 'poses': np.array(poses)}

        # return self.transform(start_img), self.transform(end_img), torch.Tensor(hand_loc), torch.Tensor(poses)
        return sample
    

def collate_func(batch):
    # count = 0
    s_i = []
    e_i = []
    h_l = []
    poses = []
    max_iter = 0
    for i in batch:
        # print('count', count)
        # print(i)
        # count+=1
        s_i.append(i['start_img'])
        e_i.append(i['end_img'])
        h_l.append(i['hand_loc'])
        poses.append(i['poses'])
        if(len(i['poses']) > max_iter):
            max_iter = len(i['poses'])
            # print('len(i[\'poses\'])', len(i['poses']))
        
    #pad with last element:
    paded_poses = []
    for i in poses:
        if len(i) < max_iter:
            # print('True')
            paded = np.tile(i[-1], (max_iter - len(i), 1))
            # paded = np.repeat( i[-1], max_iter - len(i), axis=0)
            # print('i',i[-1])
            new_arr = np.concatenate((i, paded), axis=0)
            # print(i.shape, paded.shape)
            # print('new arr', new_arr)
            paded_poses.append(torch.Tensor(new_arr))
        else:
            paded_poses.append(torch.Tensor(i))

    # print('max_iter', max_iter)
    return [s_i, e_i, h_l, paded_poses]
