import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np

# from gmm import MDN
# from traj_embd import BaseTraj


class LatentPlannerEncoder(nn.Module):
    def __init__(self, latent_dim=256, max_len=500, nhead=8, ntrans=3, dropout=0) -> None:
        super(LatentPlannerEncoder, self).__init__()
        resnet50 = models.resnet50(pretrained=True)
        #encode the current and goal images 
        self._resnet_features_image = nn.Sequential(*list(resnet50.children())[:-1])
        #get the current hand location
        # self.hand_location = 
        #mlp encoder
        self.encoder_input = nn.Sequential(
            nn.Linear(2048*2 +3 , 150), 
            nn.BatchNorm1d(150),
            nn.Linear(150, latent_dim)

        )
        

    def forward(self, curr_image, goal_img, hand_loc ):
        curr_embed = self._resnet_features_image(curr_image)
        goal_embed = self._resnet_features_image(goal_img)
        # print('curr_emb', torch.reshape(curr_embed, (10,2048)).shape)
        # print('goal_emb', goal_embed.shape)
        # print('hand_loc', hand_loc.shape)
        # curr_embed = torch.squeeze(curr_embed)
        # goal_embed = torch.squeeze(goal_embed)
        curr_embed = torch.reshape(curr_embed, (10,2048))
        goal_embed = torch.reshape(goal_embed, (10,2048))
        print(torch.cat((curr_embed, goal_embed, hand_loc), dim=1).shape)
        x = torch.cat((curr_embed, goal_embed, hand_loc), dim=1)
        latent_embed = self.encoder_input(x)
        
        return latent_embed



