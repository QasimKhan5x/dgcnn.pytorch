#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import torch
import torch.nn as nn

from models.dgcnn import DGCNN
from models.hog import HOG_Embedding
from models.layers import PositionEmbedding, ResidualConvLayer
from models.transformer import (Decoder, DecoderLayer, Encoder, EncoderLayer,
                                PositionwiseFeedForward)


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.nn = nn.Sequential(ResidualConvLayer(emb_dims, emb_dims // 2, emb_dims // 4),
                                nn.Conv1d(emb_dims // 4, emb_dims // 8,
                                          1, bias=False),
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
        # (batch_size, num_categories)
        lbl = input[0]
        # (batch_size, emb_dims, num_points)
        attn = input[1].transpose(1, 2)
        # num_points
        N = attn.size(2) 

        # (batch_size, emb_dims / 8, num_points)
        score = self.nn(attn)
        # (batch_size, emb_dim / 8, 1)
        score_max = score.max(dim=-1, keepdim=True)[0]
        # (batch_size, emb_dim / 8, num_points)
        score_max = score_max.repeat(1, 1, N)

        # (batch_size, num_categories, 1)
        lbl = lbl.unsqueeze(-1)
        # (batch_size, num_categories, 1) -> (batch_size, emb_dims / 8, 1)
        lbl = self.label_conv(lbl)
        # B x (emb_dims / 8) x N
        lbl = lbl.repeat(1, 1, N)

        # (batch_size, emb_dim + 2 * (emb_dims / 8), num_points)
        x = torch.cat((attn, score_max, lbl), dim=1)
        return self.clf(x)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        # dgcnn graph features
        self.emb_nn = DGCNN(args)
        # embedding for hog
        self.grads_emb = HOG_Embedding(k=args.k, c_in=18, c_out=args.emb_dims)
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
                                     batch_first=True,
                                     dropout=0.1)
        ff = PositionwiseFeedForward(args.emb_dims, args.ff_dims)
        # encoder for graph features
        self.graph_encoder = Encoder(EncoderLayer(size=args.emb_dims, self_attn=c(
            attn), feed_forward=c(ff), dropout=0.1), N=args.n_blocks)
        # encoder for HOG features
        self.hog_encoder = Encoder(EncoderLayer(size=args.emb_dims, self_attn=c(
            attn), feed_forward=c(ff), dropout=0.1), N=args.n_blocks)
        # fuse these features
        self.attention = c(attn)
        # produce segmap for mlp
        self.decoder = Decoder(DecoderLayer(size=args.emb_dims, self_attn=c(
            attn), src_attn=c(attn), feed_forward=c(ff), dropout=0.1), N=args.n_blocks)
        self.head = MLPHead(args)

    def forward(self, src, lbl):
        '''
        src (batch_size, 3, num_points)
        lbl (batch_size, 16)
        '''
        # (batch_size, emb_dims, num_points)
        src_embedding = self.emb_nn(src)
        # (batch_size, emb_dims, num_points)
        tgt_embedding = self.grads_emb(src)
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
        attn_scores = self.decoder(src_embedding, memory)
        # (batch_size, nclasses, num_points)
        logits = self.head(lbl, attn_scores)
        return logits
