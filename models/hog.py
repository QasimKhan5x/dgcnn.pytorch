import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import knn
from models.layers import ResidualConvLayer


def compute_hog_1x1(x, k, use_cpu=False):
    '''
    Compute histogram of oriented gradients using cell size of 1
    so that every point gets information of its neighbors

    x (B x 3 x N)
    k (number of nbrs to consider)
    '''
    batch_size = x.size(0)
    num_pts = x.size(2)
    dev = x.get_device()
    nn_idx = knn(x, k).view(-1)
    # B x N x k x 3
    x_nn = x.contiguous().view(batch_size * num_pts, -
                               1)[nn_idx, :].view(batch_size,
                                                  num_pts, k, 3)
    # center the pointcloud
    mean = x_nn.mean(dim=2, keepdim=True)  # B x N x 1 x 3
    centered = x_nn - mean
    # perform svd to obtain gradients & magnitudes
    # considering s as mag because |v|=1 for all points
    _, s, v = np.linalg.svd(
        centered.detach().cpu().numpy(), full_matrices=False)
    # convert to tensors
    v = torch.from_numpy(v)
    s = torch.from_numpy(np.sqrt(s))
    # move to appropriate device
    if "LOCAL_RANK" in os.environ:
        v = v.cuda(int(os.environ["LOCAL_RANK"]))
        s = s.cuda(int(os.environ["LOCAL_RANK"]))
    elif not use_cpu:
        v = v.cuda(dev)
        s = s.cuda(dev)
    # get the first element (largest variance)
    gradients = v[:, :, 0]  # BxNx3x3 -> BxNx3
    magnitudes = s[:, :, 0].unsqueeze(-1)  # BxNx3 -> BxNx1

    # orient grads and mags into knn shape
    gradients_nn = gradients.view(
        batch_size * num_pts, -1)[nn_idx, :].view(batch_size, num_pts, k, 3)
    magnitudes_nn = magnitudes.view(
        batch_size * num_pts, -1)[nn_idx, :].view(batch_size, num_pts, k, 1)
    # compute angles
    zenith = torch.acos(gradients_nn[:, :, :, 2]).unsqueeze(-1) * 180 / np.pi
    azimuth = torch.atan(
        gradients_nn[:, :, :, 1] / gradients_nn[:, :, :, 0]).unsqueeze(-1) * 180 / np.pi
    # stack into cells (zenith, azimuth, magnitude)
    cells = torch.cat((zenith.int(), azimuth.int(), magnitudes_nn), dim=-1)
    # don't differentiate between signed and unsigned
    cells[cells < 0] += 180
    # init histogram
    if use_cpu:
        histogram = torch.zeros((batch_size, num_pts, 9, 2))
    else:
        if 'LOCAL_RANK' in os.environ:
            histogram = torch.zeros((batch_size, num_pts, 9, 2), device=torch.device(
                int(os.environ['LOCAL_RANK'])))
        else:
            dev = x.get_device()
            histogram = torch.zeros(
                (batch_size, num_pts, 9, 2), device=torch.device(dev))
    # 20 degrees bins computed from angles
    bins = torch.floor(cells[:, :, :, :2] / 20.0 - 0.5) % 9
    # vote for bin i
    width = 20.0
    num_bins = 9
    first_centers = width * ((bins + 1) % num_bins + 0.5)
    first_votes = cells[:, :, :, 2].unsqueeze(-1) * \
        ((first_centers - cells[:, :, :, :2]) % 180) / width
    # vote for next bin
    second_centers = width * (bins + 0.5)
    second_votes = cells[:, :, :, 2].unsqueeze(-1) * \
        ((cells[:, :, :, :2] - second_centers) % 180) / width
    for c in range(9):
        histogram[:, :, c] += (first_votes * (bins == c)).sum(dim=2)
        histogram[:, :, (c+1) % 9] += (second_votes * (bins == c)).sum(dim=2)
    histogram = F.normalize(histogram, p=2.0, dim=2)
    histogram = histogram.view(batch_size, -1, num_pts)
    return histogram


class HOG_Embedding(nn.Module):
    def __init__(self, k, c_in=18, c_out=512) -> None:
        '''c_out >>> c_in'''
        super().__init__()

        self.k = k

        self.network = nn.Sequential(
            ResidualConvLayer(c_in, c_out // 8, c_out // 4),
            ResidualConvLayer(c_out // 4, c_out // 2, c_out)
        )

    def forward(self, x):
        # x (B x 3 x N)
        # (B x 9 x N)
        hog = compute_hog_1x1(x[:, :3], k=self.k)
        x = torch.cat((hog, x[:, 3:]), dim=1)
        # B x c_out, N
        return self.network(x)
