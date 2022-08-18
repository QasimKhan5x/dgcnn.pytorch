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


class Transform_Net(nn.Module):
    def __init__(self, args):
        super(Transform_Net, self).__init__()

        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn5 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        nn.init.constant_(self.transform.weight, 0)
        nn.init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        # x (B x 3 x N)
        batch_size = x.size(0)
        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = get_graph_feature(x, k=self.k)

        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        t = self.conv1(t)
        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        t = self.conv2(t)
        # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        t = t.max(dim=-1, keepdim=False)[0]

        # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        t = self.conv3(t)
        # (batch_size, 1024, num_points) -> (batch_size, 1024)
        t = t.max(dim=-1, keepdim=False)[0]

        # (batch_size, 1024) -> (batch_size, 512)
        t = F.leaky_relu(self.bn4(self.linear1(t)), negative_slope=0.2)
        # (batch_size, 512) -> (batch_size, 256)
        t = F.leaky_relu(self.bn5(self.linear2(t)), negative_slope=0.2)

        # (batch_size, 256) -> (batch_size, 3*3)
        t = self.transform(t)
        # (batch_size, 3*3) -> (batch_size, 3, 3)
        t = t.view(batch_size, 3, 3)

        # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1).contiguous()
        # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)
        # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = x.transpose(2, 1).contiguous()
        return x


# class DGCNN(nn.Module):
#     def __init__(self, args):
#         super(DGCNN, self).__init__()
#         self.k = args.k
#         self.emb_dim = args.emb_dim

#         self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(64),
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(64),
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(64),
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(64),
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#                                    nn.BatchNorm2d(64),
#                                    nn.LeakyReLU(negative_slope=0.2))
#         self.conv6 = nn.Sequential(nn.Conv1d(64*3, args.emb_dim, kernel_size=1, bias=False),
#                                    nn.BatchNorm1d(args.emb_dim),
#                                    nn.LeakyReLU(negative_slope=0.2))

#     def forward(self, x):
#         batch_size = x.size(0)
#         num_points = x.size(2)

#         # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
#         x = get_graph_feature(x, k=self.k)
#         # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x = self.conv1(x)
#         # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
#         x = self.conv2(x)
#         # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
#         x1 = x.max(dim=-1, keepdim=False)[0]

#         # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         x = get_graph_feature(x1, k=self.k)
#         # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x = self.conv3(x)
#         # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
#         x = self.conv4(x)
#         # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
#         x2 = x.max(dim=-1, keepdim=False)[0]

#         # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
#         x = get_graph_feature(x2, k=self.k)
#         # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
#         x = self.conv5(x)
#         # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
#         x3 = x.max(dim=-1, keepdim=False)[0]

#         # (batch_size, 64*3, num_points)
#         x = torch.cat((x1, x2, x3), dim=1)
#         # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
#         y = self.conv6(x)
#         # (batch_size, num_points, emb_dims)
#         # y = y.view(batch_size, num_points, self.emb_dim)

#         return y


class DGCNN(nn.Module):
    def __init__(self, args):
        super(DGCNN, self).__init__()

        self.emb_dims = args.emb_dim
        self.k = args.k

        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, self.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(self.emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()

        x = get_graph_feature(x, k=self.k)
        x = F.relu(self.bn1(self.conv1(x)))
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn2(self.conv2(x)))
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = F.relu(self.bn4(self.conv4(x)))
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = F.relu(self.bn5(self.conv5(x))).view(batch_size, -1, num_points)
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


class PointTransformerLayer(nn.Module):
    def __init__(self, d_points=256, d_model=64, k=16) -> None:
        super(PointTransformerLayer, self).__init__()

        self.k = k

        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)

        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model, bias=True),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)

    def _square_distance(self, src, dst):
        return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


    def _index_points(self, points, idx):
        raw_size = idx.size()
        idx = idx.reshape(raw_size[0], -1)
        res = torch.gather(
            points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
        return res.reshape(*raw_size, -1)

    def forward(self, xyz, features):
        # xyz: b x n x 3, features: b x n x f
        dists = self._square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = self._index_points(xyz, knn_idx)

        pre = features
        x = self.fc1(features)
        q, k, v = self.w_qs(x), self._index_points(
            self.w_ks(x), knn_idx), self._index_points(self.w_vs(x), knn_idx)

        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn, dim=-2)  # b x n x k x f
        attn = F.normalize(attn, p=1.0, dim=-2)

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre
        return res


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
        return self.w_2(self.dropout(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous())).transpose(2, 1).contiguous())


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dim
        self.nn = nn.Sequential(nn.Conv1d(emb_dims, emb_dims // 2, 1, bias=False),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Dropout(p=args.dropout),
                                nn.Conv1d(emb_dims // 2, emb_dims // 4, 1, bias=False),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Dropout(p=args.dropout),
                                nn.Conv1d(emb_dims // 4, emb_dims //
                                          8, 1, bias=False),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU(),
                                nn.Dropout(p=args.dropout),
                                nn.Conv1d(emb_dims // 8, args.nclasses, 1)
                                )

    def forward(self, x):
        # x = x.transpose(2, 1).contiguous()
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
        # self.emb_nn.load_state_dict(torch.load('ckpts/dgcnn.pt'))
        self.transformer = Transformer(args)
        if args.use_custom_attention:
            self.attention = MultiHeadedAttention(h=args.n_heads, d_model=args.emb_dim, dropout=args.dropout)
        else:
            self.attention = None
        self.head = MLPHead(args)

    def forward(self, src):
        # src (batch_size, 3, num_points)
        # (batch_size, emb_dims, num_points)
        src_embedding = self.emb_nn(src)
        # (batch_size, 3, num_points)
        tgt = get_gradients(src, k=self.k).transpose(1, 2).contiguous()
        # (batch_size, emb_dims, num_points)
        tgt_embedding = self.emb_nn(tgt)
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
        logits = self.head(scores.transpose(1, 2).contiguous())
        return logits
