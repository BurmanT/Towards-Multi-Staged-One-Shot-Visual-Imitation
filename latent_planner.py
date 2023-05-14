import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np

from models.latent_planner_encoder import LatentPlannerEncoder
from models.latent_planner_decoder import LatentPlannerDecoder


class LatentPlanner(nn.Module):
    def __init__(self, latent_dim=256, n_gaussians=5) -> None:
        super(LatentPlanner, self).__init__()
        self.encoder = LatentPlannerEncoder(latent_dim=latent_dim)
        self.decoder = LatentPlannerDecoder(in_dim=latent_dim, out_dim=3, n_mixtures=n_gaussians)

    def forward(self, curr_image, goal_img, hand_loc ):
        # print('hello')
        # print('curr from lp', curr_image.shape)
        # print(goal_img.shape)
        latent_plan = self.encoder(curr_image, goal_img, hand_loc)
        # print('encoded:', latent_plan.shape)
        pi, sigma, mu = self.decoder(latent_plan)
        return pi , sigma, mu