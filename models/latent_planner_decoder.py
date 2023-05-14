import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np

# from gmm import MDN
# from traj_embd import BaseTraj

class LatentPlannerDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, n_mixtures=5):
        super().__init__()
        self._n_mixtures = n_mixtures
        self._out_dim = out_dim
        self._mean = nn.Linear(in_dim, out_dim * n_mixtures)
        self._alpha = nn.Linear(in_dim, n_mixtures)
        self._sigma_inv = nn.Linear(in_dim, n_mixtures)

    def forward(self, inputs):
        prev_shape = inputs.shape[:-1]
        mean = self._mean(inputs).reshape(list(prev_shape) + [self._n_mixtures, self._out_dim])
        sigma_inv = torch.exp(self._sigma_inv(inputs))
        alpha = F.softmax(self._alpha(inputs), -1)
        return alpha, sigma_inv, mean
    
def mixture_density_loss(real, mean, sigma_inv, alpha, eps=1e-5):
    output_test = []
    for i in range(real.shape[0]):
        # temp = torch.sub
        # output_test.append((real.unsqueeze(-2)[i] - mean[i]))
        # print('success', (real.unsqueeze(-2)[i] - mean[i]).shape)
        C = real.shape[-1]
        sub = real.unsqueeze(-2)[i] - mean[i]
        exp_term = -0.5 * torch.sum((sub ** 2), -1) * (sigma_inv[i] ** 2)
        ln_frac_term = torch.log(alpha[i] * sigma_inv[i] + eps) - 0.5 * C * np.log(2 * np.pi)
        expected_loss = -torch.logsumexp(ln_frac_term + exp_term, -1)
        output_test.append(expected_loss)
        # print('expected loss', expected_loss)
        # print('success 2')

    output = torch.stack(output_test, dim=1)
    # print('output', output)
    # print('output shape', output.shape)
    return torch.mean(output)
    # print('output shape', output.shape)
    # print('sigma_inv', sigma_inv.shape)
    # print('alpha', alpha.shape)
    # C = real.shape[-1]
    # exp_term = -0.5 * torch.sum((output ** 2), -1) * (sigma_inv ** 2)
    # ln_frac_term = torch.log(alpha * sigma_inv + eps) - 0.5 * C * np.log(2 * np.pi)
    # expected_loss = -torch.logsumexp(ln_frac_term + exp_term, -1)
    # # print('exp loss, ', expected_loss)
    # return torch.mean(expected_loss)

    # C = real.shape[-1]
    # print('real', real.shape)
    # print('real.unsqueeze(-2):', real.unsqueeze(-2).shape)
    # print('mean:', mean.shape)
    # print('np sub ', np.subtract(real.unsqueeze(-2).detach().numpy(), mean.detach().numpy()))
    # print('torch.sub(real.unsqueeze(-2), mean)', torch.sub(real.unsqueeze(-2), mean).shape)
    # print('((real.unsqueeze(-2) - mean) ** 2)', ((real.unsqueeze(-2) - mean) ** 2))
    # exp_term = -0.5 * torch.sum(((real.unsqueeze(-2) - mean) ** 2), -1) * (sigma_inv ** 2)
    # ln_frac_term = torch.log(alpha * sigma_inv + eps) - 0.5 * C * np.log(2 * np.pi)
    # expected_loss = -torch.logsumexp(ln_frac_term + exp_term, -1)
    # # print('exp loss, ', expected_loss)
    # return torch.mean(expected_loss)
