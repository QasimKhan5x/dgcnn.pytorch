#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

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
        point_ftrs = input[0]
        # (batch_size, emb_dims, num_points)
        global_ftrs = input[1]
        # number of points in individual PC
        N = point_ftrs.size(2) 
        # (batch_size, 3 + emb_dim * 2, num_points)
        x = torch.cat((point_ftrs, global_ftrs), dim=1)
        return self.clf(x)


# class Net(nn.Module):
#     def __init__(self, args):
#         super(Net, self).__init__()

#         # xyz + rgb + height
#         total_channels = 3 + 3 + 0

#         self.k = args.k
#         # hog
#         # total_channels += 18
#         # dgcnn graph features
#         self.emb_nn = DGCNN_PNeXt(args, c_in=total_channels)
#         # positional embeddings
#         # self.pos_mlp = nn.Sequential(
#             # PositionEmbedding(args, c_in=3+extra_channels))
#         self.head = MLPHead(args)

#     def forward(self, pos, feat):
#         '''
#         src (batch_size, channels, num_points)

#         channels = xyz + rgb + normalized position xyz (9)
#         '''
#         # hog = compute_hog_1x1(pos, k=self.k)
#         # src = torch.cat((src, hog), dim=1)
#         # (batch_size, emb_dims, num_points) (check result with and without pos_mlp)
#         # canonical = self.pos_mlp(src)
#         graph_point_ftrs, graph_global_ftrs = self.emb_nn(pos, feat)
#         # (batch_size, nclasses, num_points)
#         logits = self.head(graph_point_ftrs, graph_global_ftrs)
#         # seg_pred = logits.permute(0, 2, 1).contiguous()
#         # seg_pred = seg_pred.view(-1, 50)
#         return logits

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(512)
        self.bn8 = nn.BatchNorm1d(256)

        c_in = 3 + 3

        self.conv1 = nn.Sequential(nn.Conv2d(c_in * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(1216, 512, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(512, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv9 = nn.Conv1d(256, 13, kernel_size=1, bias=False)


    def knn(self, x, k):
        inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
        return idx


    def get_graph_feature(self, x, k, knn_only=False, disp_only=False):
        # x = x.squeeze()
        idx = self.knn(x, k=k)  # (batch_size, num_points, k)
        batch_size, num_points, _ = idx.size()
        dev_idx = x.get_device()
        device = torch.device(dev_idx if dev_idx != -1 else 'cpu')

        idx_base = torch.arange(
            0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

        _, num_dims, _ = x.size()

        x = x.transpose(2,
                        1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

        if knn_only:
            return feature
        elif disp_only:
            return (feature - x).permute(0, 3, 1, 2).contiguous()

        feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2).contiguous()

        return feature

    def forward(self, xyz, feat):
        x = torch.cat((xyz, feat), dim=1)

        batch_size = x.size(0)
        num_points = x.size(2)

        # (batch_size, 9, num_points) -> (batch_size, 9*2, num_points, k)
        x = self.get_graph_feature(x, k=self.k)
        # (batch_size, 9*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv1(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x1 = x.max(dim=-1, keepdim=False)[0]
        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.get_graph_feature(x1, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv3(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x2 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.get_graph_feature(x2, k=self.k)
        # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv5(x)
        # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x3 = x.max(dim=-1, keepdim=False)[0]

        # (batch_size, 64*3, num_points)
        x = torch.cat((x1, x2, x3), dim=1)

        # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = self.conv6(x)
        # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)
        x = x.max(dim=-1, keepdim=True)[0]

        # (batch_size, 1024, num_points)
        x = x.repeat(1, 1, num_points)
        # (batch_size, 1024+64*3, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)

        # (batch_size, 1024+64*3, num_points) -> (batch_size, 512, num_points)
        x = self.conv7(x)
        # (batch_size, 512, num_points) -> (batch_size, 256, num_points)
        x = self.conv8(x)
        x = self.dp1(x)
        # (batch_size, 256, num_points) -> (batch_size, 13, num_points)
        x = self.conv9(x)

        return x
