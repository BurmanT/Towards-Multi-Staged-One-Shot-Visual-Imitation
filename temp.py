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

# traj_dir = '/Users/naira/PycharmProjects/MS-OSVI/data/panda_trajs/'
# data = glob.glob(traj_dir + 'traj0*.pkl')
# print(sorted(data))
from torch.utils.data import DataLoader
from dataloader import TrajectoryDataset, collate_func

from models.latent_planner import LatentPlanner
from models.latent_planner_encoder import LatentPlannerEncoder
from models.latent_planner_decoder import LatentPlannerDecoder, mixture_density_loss


training_data = TrajectoryDataset(traj_dir='/Users/naira/PycharmProjects/MS-OSVI/data/panda_trajs/', transform=transforms.ToTensor())
train_dataloader = DataLoader(training_data, batch_size=10, collate_fn=collate_func, shuffle=True)

# start_img, end_img, hand_loc, poses = next(iter(train_dataloader))
# sample = next(iter(train_dataloader))
# print(sample)


model = LatentPlanner(latent_dim=256, n_gaussians=5)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(50):
    for i, data in enumerate(train_dataloader):

        print('#####################')
        # print(i)
        transform = transforms.ToTensor()
        curr_image, goal_image, hand_loc, poses = data[0], data[1], data[2], data[3]
        curr_image, goal_image, hand_loc, poses = torch.stack(curr_image, dim=0), torch.stack(goal_image, dim=0), torch.stack(hand_loc, dim=0), torch.stack(poses, dim=0)
        pi_variable, sigma_variable, mu_variable = model(curr_image, goal_image, hand_loc)
        loss = mixture_density_loss(poses, mu_variable, sigma_variable, pi_variable)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(loss.item())
        if epoch % 1 == 0:
            print('epoch: ',epoch, 'loss: ', loss)
    # curr_image = transform(np.array(data[0]))
    # print('tens curr', curr_image.shape)
    # print(type(curr_image))
    # print(len(curr_image))
    # print(type(goal_image))
    # print(len(goal_image))
    # print(type(hand_loc))
    # print(len(hand_loc))
    # print(type(poses))
    # print(len(poses))
    
    


        # transform = transforms.ToTensor()
        # curr_image, goal_image, hand_loc = sawyer_data['0']['obs']['image'], sawyer_data['15']['obs']['image'], torch.Tensor(poses[0])
        # # print('tens curr', transform(curr_image).shape)
        # pi_variable, sigma_variable, mu_variable = model(transform(curr_image)[None, :], transform(goal_image)[None, :], hand_loc)
        # # print('res',pi_variable, sigma_variable, mu_variable)
        # loss = mixture_density_loss(y_variable, mu_variable, sigma_variable, pi_variable)
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        # # print(loss.item())
        # if epoch % 10 == 0:
        #     print(epoch, loss.item())