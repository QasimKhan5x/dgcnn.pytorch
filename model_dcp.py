#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Part of the code is referred from: http://nlp.seas.harvard.edu/annotated-transformer/


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, knn_only=False, disp_only=False):
    # x = x.squeeze()
    idx = knn(x, k=k)  # (batch_size, num_points, k)
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


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()

        self.emb_dims = args.emb_dim
        self.k = args.k


        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, self.emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.emb_dims),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1).unsqueeze(-1)

        x = self.conv5(x).view(batch_size, -1, num_points)
        return x


def get_gradients(x, k, do_pca=False):
    '''
    x (Bx3xN) batch of point clouds
    return gradients (BxNx3): direction of maximimal variance at each point
    '''
    x_nn = get_graph_feature(x, k=k, knn_only=True)  # Bx3xN -> BxNxkx3
    if do_pca:
        _, _, v = torch.pca_lowrank(x_nn)  # BxNxkx3 -> BxNx3x3

    else:
        mean = x_nn.mean(dim=2).unsqueeze(dim=2)
        centered = x_nn - mean
        # _, _, v = torch.linalg.svd(centered)  # BxNxkx3 -> BxNx3x3
        _, _, v = np.linalg.svd(centered.detach().cpu().numpy(), full_matrices=False)
        v = torch.from_numpy(v)
        if "LOCAL_RANK" in os.environ:
            v = v.cuda(int(os.environ["LOCAL_RANK"]))
        else:
            v = v.cuda()
    gradients = v[:, :, 0]  # BxNx3x3 -> BxNx3
    return gradients


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
    mean = x_nn.mean(dim=2).unsqueeze(dim=2)
    centered = x_nn - mean
    # perform svd to obtain gradients & magnitudes
    # considering s as mag because |v|=1
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
    first_centers = 20 * ((bins + 1) % 9 + 0.5)
    first_votes = cells[:, :, :, 2].unsqueeze(-1) * \
        ((first_centers - cells[:, :, :, :2]) % 180) / 20
    # vote for bin (i + 1) % 9
    second_centers = 20 * (bins + 0.5)
    second_votes = cells[:, :, :, 2].unsqueeze(-1) * \
        ((cells[:, :, :, :2] - second_centers) % 180) / 20
    for c in range(9):
        histogram[:, :, c] += (first_votes * (bins == c)).sum(dim=2)
        histogram[:, :, (c+1) % 9] += (second_votes * (bins == c)).sum(dim=2)
    histogram = F.normalize(histogram, p=2.0, dim=2)
    histogram = histogram.view(batch_size, num_pts, -1)
    return histogram


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(
        query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # TODO: replace vaswani attention with point transformer
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h,
                        self.d_k).transpose(1, 2).contiguous()
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.norm(F.leaky_relu(self.w_1(x), 0.1).transpose(2, 1).contiguous())).transpose(2, 1).contiguous())


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dim
        self.nn = nn.Sequential(nn.Conv1d(emb_dims * 5 + 64, emb_dims // 2, 1, bias=False),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                nn.Dropout(p=args.dropout),
                                nn.Conv1d(emb_dims // 2, emb_dims // 4, 1, bias=False),
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

        batch_size = src.size(0)
        N = src.size(2)

        # B x emb_dims x N  (emb_dims * 4 total)
        src_max = src.max(dim=2, keepdim=True).repeat(1, 1, N)
        tgt_max = tgt.max(dim=2, keepdim=True).repeat(1, 1, N)
        src_mean = src.mean(dim=2, keepdim=True).repeat(1, 1, N)
        tgt_mean = tgt.mean(dim=2, keepdim=True).repeat(1, 1, N)

        # (batch_size, num_categories, 1)
        lbl = lbl.unsqueeze(-1)
        # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
        lbl = self.label_conv(lbl)
        # B x 64 x N
        lbl = lbl.repeat(1, 1, N)
        # (batch_size, emb_dim * 4 + 64, 2048)
        x = torch.cat((src_max, tgt_max, src_mean, tgt_mean, lbl), dim=2)
        # B x (emb_dim * 5 + 64) x 2048
        x = torch.cat((x, attn), dim=1)
        return self.nn(x)


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dim
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads

        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims, self.dropout)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)

        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn),
                                                         c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        # (batch_size, emb_dims, num_points)
        src = input[0]
        # (batch_size, emb_dims, num_points)
        tgt = input[1]
        # (batch_size, emb_dims, num_points)
        src = src.transpose(2, 1).contiguous()
        # (batch_size, emb_dims, num_points)
        tgt = tgt.transpose(2, 1).contiguous()
        # (batch_size, num_points, emb_dims)
        tgt_embedding = self.model(src, tgt).transpose(2, 1).contiguous()
        # (batch_size, num_points, emb_dims)
        src_embedding = self.model(tgt, src).transpose(2, 1).contiguous()
        # (batch_size, nclasses, num_points)
        return src_embedding, tgt_embedding


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.k = args.k
        # self.tnet = Transform_Net(args)
        # self.tnet.load_state_dict(torch.load('ckpts/tnet.pt'))
        self.emb_nn = DGCNN(args)
        # self.grads_emb = nn.Sequential(
        #     nn.Linear(18, args.emb_dim // 8),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Linear(args.emb_dim // 8, args.emb_dim // 4),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Linear(args.emb_dim // 8, args.emb_dim // 2),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Linear(args.emb_dim // 2, args.emb_dim),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        # )
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
        # self.emb_nn.load_state_dict(torch.load('ckpts/dgcnn.pt'))
        self.transformer = Transformer(args)
        if args.use_custom_attention:
            self.attention = MultiHeadedAttention(h=args.n_heads, d_model=args.emb_dim, dropout=args.dropout)
        else:
            self.attention = None
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
        src_embedding_p, tgt_embedding_p = self.transformer(
            src_embedding, tgt_embedding)
        # (batch_size, num_points, emb_dims)
        src_embedding = (src_embedding + src_embedding_p).transpose(1, 2).contiguous()
        tgt_embedding = (tgt_embedding + tgt_embedding_p).transpose(1, 2).contiguous()
        # (batch_size, num_points, emb_dims)
        # TODO replace with another fusion mechanism
        if self.attention is not None:
            scores = self.attention(
                query=src_embedding, key=tgt_embedding, value=tgt_embedding)
        else:
            scores, _ = attention(query=src_embedding,
                                  key=tgt_embedding, value=tgt_embedding)
        # (batch_size, nclasses, num_points)
        # logits = self.head(scores.transpose(1, 2).contiguous(), lbl)
        logits = self.head(src_embedding, tgt_embedding, lbl, scores)
        return logits
