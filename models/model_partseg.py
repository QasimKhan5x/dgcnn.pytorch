#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dgcnn import DGCNN, knn
from models.layers import PositionEmbedding, ResidualConvLayer
from models.transformer import (Decoder, DecoderLayer, Encoder, EncoderLayer,
                                PositionwiseFeedForward)


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
    histogram = histogram.view(batch_size, -1, num_pts)
    return histogram


class HOG_Embedding(nn.Module):
    def __init__(self, c_in=18, c_out=512) -> None:
        '''c_out >>> c_in'''
        super().__init__()

        self.network = nn.Sequential(
            ResidualConvLayer(c_in, c_out // 8, c_out // 4),
            ResidualConvLayer(c_out // 4, c_out // 2, c_out)
        )

    def forward(self, x):
        return self.network(x)


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.nn = nn.Sequential(ResidualConvLayer(emb_dims, emb_dims // 2, emb_dims // 4, 
                                                  dp1=args.dropout, dp2=args.dropout),
                                nn.Conv1d(emb_dims // 4, emb_dims //
                                          8, 1, bias=False),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                )
        self.label_conv = nn.Sequential(nn.Conv1d(16, emb_dims // 8, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(emb_dims // 8),
                                        nn.LeakyReLU(negative_slope=0.2)
                                        )
        self.clf = nn.Sequential(ResidualConvLayer(emb_dims + emb_dims // 8 * 2,
                                                   512, 256, dp1=args.dropout, dp2=args.dropout),
                                 ResidualConvLayer(256, 128, args.nclasses, dp1=args.dropout, dp2=0))

    def forward(self, *input):
        # (batch_size, 16)
        lbl = input[0]
        # (batch_size, emb_dims, num_points)
        attn = input[1].transpose(1, 2)
        # num_points
        N = attn.size(2) 

        # (batch_size, emb_dims / 8, num_points)
        score = self.nn(attn)
        # (batch_size, emb_dim / 8, 1)
        score_max = score.max(dim=1)[0]
        # (batch_size, emb_dim / 8, num_points)
        score_max = score_max.repeat(1, 1, N)

        # (batch_size, num_categories, 1)
        lbl = lbl.unsqueeze(-1)
        # (batch_size, num_categories, 1) -> (batch_size, emb_dims / 8, 1)
        lbl = self.label_conv(lbl)
        # B x (emb_dims / 8) x N
        lbl = lbl.repeat(1, 1, N)

        # (batch_size, emb_dim + 2 * (emb_dims / 8), 2048)
        x = torch.cat((score, attn, lbl), dim=1)
        return self.clf(x)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.k = args.k
        # dgcnn graph features
        self.emb_nn = DGCNN(args)
        # embedding for hog
        self.grads_emb = HOG_Embedding(c_in=18, c_out=args.emb_dims)
        # positional embeddings
        self.pos_mlp = nn.Sequential(
            PositionEmbedding(args),
            nn.Conv1d(3, args.emb_dims, 1, bias=False),
            nn.BatchNorm1d(args.emb_dims),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # transformer layers
        c = copy.deepcopy
        attn = nn.MultiheadAttention(embed_dim=args.emb_dims,
                                     num_heads=args.n_heads,
                                     batch_first=True)
        ff = PositionwiseFeedForward(args.emb_dims, args.ff_dims)
        # encoder for graph features
        self.graph_encoder = Encoder(EncoderLayer(size=args.emb_dims, self_attn=c(
            attn), feed_forward=c(ff), dropout=0), N=args.n_blocks)
        # encoder for HOG features
        self.hog_encoder = Encoder(EncoderLayer(size=args.emb_dims, self_attn=c(
            attn), feed_forward=c(ff), dropout=0), N=args.n_blocks)
        # fuse these features
        self.attention = c(attn)
        # produce segmap for mlp
        self.decoder = Decoder(DecoderLayer(size=args.emb_dims, self_attn=c(
            attn), src_attn=c(attn), feed_forward=c(ff), dropout=0), N=args.n_blocks)
        self.head = MLPHead(args)

    def forward(self, src, lbl):
        '''
        src (batch_size, 3, num_points)
        lbl (batch_size, 16)
        '''
        # (batch_size, emb_dims, num_points)
        src_embedding = self.emb_nn(src)
        # (batch_size, 9 * 2, num_points)
        tgt = compute_hog_1x1(src, k=self.k)
        # (batch_size, emb_dims, num_points)
        tgt_embedding = self.grads_emb(tgt)
        # (batch_size, emb_dims, num_points)
        canonical = self.pos_mlp(src)
        # add position embedding & transpose to (batch_size, num_points, emb_dims)
        src_embedding = (src_embedding + canonical).transpose(1, 2)
        tgt_embedding = (tgt_embedding + canonical).transpose(1, 2)
        # (batch_size, num_points, emb_dims)
        src_embedding = self.graph_encoder(src_embedding)
        # (batch_size, num_points, emb_dims)
        tgt_embedding = self.hog_encoder(tgt_embedding)
        # (batch_size, num_points, emb_dims)
        memory, _ = self.attention(query=tgt_embedding,
                                   key=src_embedding,
                                   value=src_embedding,
                                   need_weights=False
                                   )
        # (batch_size, num_points, emb_dims)
        scores = self.decoder(src_embedding, memory)
        # (batch_size, nclasses, num_points)
        logits = self.head(lbl, scores)
        return logits
