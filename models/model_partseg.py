#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import MultiHeadedAttention
from models.dgcnn import DGCNN, knn
from models.layers import PositionEmbedding
from models.transformer import Transformer


def compute_hog_1x1(x, k, use_cpu=False):
    '''
    Compute histogram of oriented gradients using cell size of 1
    so that every point gets information of its neighbors

    x (B x 3 x N)
    k (number of nbrs to consider)
    '''
    batch_size = x.size(0)
    num_pts = x.size(2)

    nn_idx = knn(x, k).view(-1)
    # B x N x k x 3
    x_nn = x.contiguous().view(batch_size * num_pts, -
                               1)[nn_idx, :].view(batch_size, 
                                                  num_pts, k, 3)
    # center the pointcloud
    mean = x_nn.mean(dim=2, keepdim=True) # B x N x 1 x 3 
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
        v = v.cuda()
        s = s.cuda()
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
            histogram = torch.zeros(
                (batch_size, num_pts, 9, 2), device=torch.device('cuda'))
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
    histogram = histogram.view(batch_size, num_pts, -1)
    return histogram


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dim
        self.nn = nn.Sequential(nn.Conv1d(emb_dims * 5 + 64, emb_dims // 2, 1, bias=False),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Dropout(p=args.dropout),
                                nn.Conv1d(emb_dims // 2, emb_dims //
                                          4, 1, bias=False),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Dropout(p=args.dropout),
                                nn.Conv1d(emb_dims // 4, emb_dims //
                                          8, 1, bias=False),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Dropout(p=args.dropout),
                                nn.Conv1d(emb_dims // 8, args.nclasses, 1)
                                )
        self.label_conv = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(64),
                                        nn.LeakyReLU(negative_slope=0.2)
                                        )

    def forward(self, *input):
        # (batch_size, emb_dims, num_points)
        src = input[0].transpose(1, 2)
        # (batch_size, emb_dims, num_points)
        tgt = input[1].transpose(1, 2)
        # (batch_size, num_categories)
        lbl = input[2]
        # (batch_size, emb_dims, num_points)
        attn = input[3].transpose(1, 2)

        N = src.size(2)

        # B x emb_dims x N  (emb_dims * 4 total)
        src_max = src.max(dim=2, keepdim=True)[0].repeat(1, 1, N)
        tgt_max = tgt.max(dim=2, keepdim=True)[0].repeat(1, 1, N)
        src_mean = src.mean(dim=2, keepdim=True).repeat(1, 1, N)
        tgt_mean = tgt.mean(dim=2, keepdim=True).repeat(1, 1, N)
        # (batch_size, num_categories, 1)
        lbl = lbl.unsqueeze(-1)
        # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
        lbl = self.label_conv(lbl)
        # B x 64 x N
        lbl = lbl.repeat(1, 1, N)
        # (batch_size, emb_dim * 5 + 64, 2048)
        x = torch.cat((src_max, tgt_max, src_mean, tgt_mean, lbl, attn), dim=1)
        return self.nn(x)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.k = args.k
        # self.tnet = Transform_Net(args)
        # self.tnet.load_state_dict(torch.load('ckpts/tnet.pt'))
        self.emb_nn = DGCNN(args)
        self.grads_emb = nn.Sequential(
            nn.Conv1d(18, args.emb_dim // 8, 1, bias=False),
            nn.BatchNorm1d(args.emb_dim // 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(args.emb_dim // 8, args.emb_dim // 4, 1, bias=False),
            nn.BatchNorm1d(args.emb_dim // 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(args.emb_dim // 4, args.emb_dim // 2, 1, bias=False),
            nn.BatchNorm1d(args.emb_dim // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv1d(args.emb_dim // 2, args.emb_dim, 1, bias=False),
            nn.BatchNorm1d(args.emb_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.pos_mlp = nn.Sequential(
            PositionEmbedding(args),
            nn.Conv1d(3, args.emb_dim, 1, bias=False),
            nn.BatchNorm1d(args.emb_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.transformer = Transformer(args)
        self.attention = MultiHeadedAttention(
            h=args.n_heads, d_model=args.emb_dim, dropout=args.dropout)
        self.head = MLPHead(args)

    def forward(self, src, lbl):
        # src (batch_size, 3, num_points)
        # (batch_size, emb_dims, num_points)
        src_embedding = self.emb_nn(src)
        # (batch_size, num_points, 9 * 2)
        tgt = compute_hog_1x1(src, k=self.k)
        # (batch_size, emb_dims, num_points)
        tgt_embedding = self.grads_emb(tgt.transpose(1, 2).contiguous())
        # (batch_size, emb_dims, num_points)
        canonical = self.pos_mlp(src)
        src_embedding = src_embedding + canonical
        tgt_embedding = tgt_embedding + canonical
        # (batch_size, emb_dims, num_points)
        src_embedding_p, tgt_embedding_p = self.transformer(src_embedding, tgt_embedding)
        # (batch_size, num_points, emb_dims)
        src_embedding = (src_embedding_p).transpose(1, 2).contiguous()
        tgt_embedding = (tgt_embedding_p).transpose(1, 2).contiguous()
        # (batch_size, num_points, emb_dims)
        scores = self.attention(query=src_embedding, key=tgt_embedding, value=tgt_embedding)
        # (batch_size, nclasses, num_points)
        # logits = self.head(scores.transpose(1, 2).contiguous(), lbl)
        logits = self.head(src_embedding, tgt_embedding, lbl, scores)
        return logits
