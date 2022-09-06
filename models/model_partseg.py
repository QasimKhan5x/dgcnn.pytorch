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
        # self.nn = nn.Sequential(ResidualConvLayer(emb_dims, emb_dims // 2, emb_dims // 4),
        #                         nn.Conv1d(emb_dims // 4, emb_dims,
        #                                   1, bias=False),
        #                         nn.BatchNorm1d(emb_dims),
        #                         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #                         )
        self.label_conv = nn.Sequential(nn.Conv1d(16, emb_dims // 8, kernel_size=1, bias=False),
                                        nn.BatchNorm1d(emb_dims // 8),
                                        nn.LeakyReLU(negative_slope=0.2)
                                        )
        self.clf = nn.Sequential(ResidualConvLayer(emb_dims * 2 + 64 + 3,
                                                   512, 256, dp1=args.dropout, dp2=args.dropout),
                                 ResidualConvLayer(256, 128, args.nclasses, dp1=args.dropout, dp2=0))

    def forward(self, *input):
        # (batch_size, num_categories)
        lbl = input[0]
        # (batch_size, emb_dims, num_points)
        point_ftrs = input[1]#.transpose(1, 2)
        # (batch_size, emb_dims, num_points)
        global_ftrs = input[2]
        # number of points in individual PC
        N = point_ftrs.size(2) 

        # (batch_size, emb_dims / 8, num_points)
        # score = self.nn(point_ftrs)
        # (batch_size, emb_dim / 8, 1)
        # score_max = score.max(dim=-1, keepdim=True)[0]
        # (batch_size, emb_dim / 8, num_points)
        # score_max = score_max.repeat(1, 1, N)

        # (batch_size, num_categories, 1)
        lbl = lbl.unsqueeze(-1)
        # (batch_size, num_categories, 1) -> (batch_size, emb_dims / 8, 1)
        lbl = self.label_conv(lbl)
        # B x (emb_dims / 8) x N
        lbl = lbl.repeat(1, 1, N)

        # (batch_size, emb_dim * 2 + 64, num_points)
        x = torch.cat((point_ftrs, global_ftrs, lbl), dim=1)
        return self.clf(x)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.k = args.k
        # height
        extra_channels = 3 if args.use_height else 0
        # histogram
        extra_channels += 18
        # dgcnn graph features
        self.emb_nn = DGCNN_PNeXt(args, c_in=3+extra_channels)
        # positional embeddings
        # self.pos_mlp = nn.Sequential(
            # PositionEmbedding(args, c_in=3+extra_channels))
        self.head = MLPHead(args)

    def forward(self, *input):
        '''
        src (batch_size, 3 or 6, num_points)
        lbl (batch_size, 16)
        '''
        src = input[0]
        lbl = input[1]
        hog = compute_hog_1x1(src[:, :3], k=self.k)
        src = torch.cat((src, hog), dim=1)
        # src[:, :3] = src[:, :3] - src[:, :3].mean(dim=-1, keepdim=True)
        # src[:, :3] = F.normalize(src[:, :3], p=2.0, dim=-1)
        # (batch_size, emb_dims, num_points) (check result with and without pos_mlp)
        # canonical = self.pos_mlp(src)
        graph_point_ftrs, graph_global_ftrs = self.emb_nn(src)
        # (batch_size, nclasses, num_points)
        logits = self.head(lbl, graph_point_ftrs, graph_global_ftrs)

        # seg_pred = logits.permute(0, 2, 1).contiguous()
        # seg_pred = seg_pred.view(-1, 50)
        return logits
