#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from models.dgcnn import DGCNN_PNeXt
from models.hog import compute_hog_1x1
from models.layers import ResidualConvLayer


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims

        self.clf = nn.Sequential(ResidualConvLayer(emb_dims * 2 + 3,
                                                   512, 256, dp1=args.dropout, dp2=args.dropout),
                                 ResidualConvLayer(256, 128, args.nclasses, dp1=args.dropout, dp2=0))

    def forward(self, *input):
        # (batch_size, emb_dims, num_points)
        point_ftrs = input[0]#.transpose(1, 2)
        # (batch_size, emb_dims, num_points)
        global_ftrs = input[1]
        # number of points in individual PC
        N = point_ftrs.size(2) 

        # (batch_size, 3 + emb_dim * 2, num_points)
        x = torch.cat((point_ftrs, global_ftrs), dim=1)
        return self.clf(x)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.k = args.k
        # height appending
        extra_channels = 3 if args.use_height else 0
        # hog
        extra_channels += 18
        # dgcnn graph features
        self.emb_nn = DGCNN_PNeXt(args, c_in=9+extra_channels)
        # positional embeddings
        # self.pos_mlp = nn.Sequential(
            # PositionEmbedding(args, c_in=3+extra_channels))
        self.head = MLPHead(args)

    def forward(self, src):
        '''
        src (batch_size, channels, num_points)

        channels = xyz + rgb + normalized position xyz (9)
        '''
        hog = compute_hog_1x1(src[:, :3], k=self.k)
        src = torch.cat((src, hog), dim=1)
        # (batch_size, emb_dims, num_points) (check result with and without pos_mlp)
        # canonical = self.pos_mlp(src)
        graph_point_ftrs, graph_global_ftrs = self.emb_nn(src)
        # (batch_size, nclasses, num_points)
        logits = self.head(graph_point_ftrs, graph_global_ftrs)

        # seg_pred = logits.permute(0, 2, 1).contiguous()
        # seg_pred = seg_pred.view(-1, 50)
        return logits
