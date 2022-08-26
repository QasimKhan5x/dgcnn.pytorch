import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.dgcnn import knn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(
        query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    scaled_values = torch.matmul(p_attn, value)
    return scaled_values, p_attn


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


class VectorAttention(nn.Module):
    def __init__(
        self,
        args,
        pos_mlp_hidden_dim=64,
        attn_mlp_hidden_mult=4,
    ):
        super(VectorAttention, self).__init__()
        inner_dim = args.d_qkv

        self.num_neighbors = args.k
        self.size = args.emb_dim

        self.w_q = nn.Linear(args.emb_dim, inner_dim, bias=False)
        self.w_k = nn.Linear(args.emb_dim, inner_dim, bias=False)
        self.w_v = nn.Linear(args.emb_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, args.emb_dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, inner_dim)
        )

        attn_inner_dim = inner_dim * attn_mlp_hidden_mult

        self.attn_mlp = nn.Sequential(
            nn.Linear(inner_dim, attn_inner_dim),
            nn.ReLU(),
            nn.Linear(attn_inner_dim, inner_dim),
        )

    def forward(self, query, key, value, canonical, mask=None):
        bs, n, num_neighbors = query.shape[0], query.shape[1], self.num_neighbors

        # get queries, keys, values (B x N x D)

        q, k, v = self.w_q(query), self.w_k(key), self.w_v(value)

        # calculate relative positional embeddings
        idx = knn(canonical, k=num_neighbors).view(-1)
        pos_nn = canonical.contiguous().view(bs * n, -1)[idx,
                                                         :].view(bs, n, num_neighbors, 3)
        pos_repeat = canonical.contiguous().view(
            bs, n, 1, 3).repeat(1, 1, num_neighbors, 1)
        # B N K C
        rel_pos_emb = self.pos_mlp(pos_nn - pos_repeat)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product

        q = q.contiguous().view(
            bs * n, -1)[idx, :].view(bs, n, num_neighbors, -1)
        k = k.contiguous().view(
            bs * n, -1)[idx, :].view(bs, n, num_neighbors, -1)
        # b n k c
        qk_rel = q - k

        # expand values (B x N x C) -> (B x N x k x C)
        v = v.contiguous().view(
            bs * n, -1)[idx, :].view(bs, n, num_neighbors, -1)

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first

        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # attention
        
        attn = sim.softmax(dim=-1)
        attn = F.normalize(attn, dim=-2)

        # aggregate

        agg = torch.einsum('b i j d, b i j d -> b i d', attn, v).contiguous()

        # combine heads

        del q
        del k
        del v
        return self.to_out(agg)


class MultiHeadVectorAttention(nn.Module):
    def __init__(
        self,
        args,
        dim_head=64,
        pos_mlp_hidden_dim=64,
        attn_mlp_hidden_mult=4,
    ):
        super(MultiHeadVectorAttention, self).__init__()
        self.heads = args.n_heads
        inner_dim = dim_head * self.heads

        self.num_neighbors = args.k
        self.size = args.emb_dim

        self.w_q = nn.Linear(args.emb_dim, inner_dim, bias=False)
        self.w_k = nn.Linear(args.emb_dim, inner_dim, bias=False)
        self.w_v = nn.Linear(args.emb_dim, inner_dim, bias=False)

        self.to_out = nn.Linear(inner_dim, args.emb_dim)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, inner_dim)
        )

        attn_inner_dim = inner_dim * attn_mlp_hidden_mult

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(inner_dim, attn_inner_dim, 1, groups=self.heads),
            nn.ReLU(),
            nn.Conv2d(attn_inner_dim, inner_dim, 1, groups=self.heads),
        )

    def forward(self, query, key, value, canonical, mask=None):
        bs, n, h, num_neighbors = query.shape[0], query.shape[1], self.heads, self.num_neighbors

        # get queries, keys, values

        q, k, v = self.w_q(query), self.w_k(key), self.w_v(value)

        # split out heads

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # calculate relative positional embeddings
        idx = knn(canonical, k=num_neighbors).view(-1)
        pos_nn = canonical.contiguous().view(bs * n, -1)[idx,
                                                   :].view(bs, n, num_neighbors, 3)
        pos_repeat = canonical.contiguous().view(
            bs, n, 1, 3).repeat(1, 1, num_neighbors, 1)
        rel_pos_emb = self.pos_mlp(pos_nn - pos_repeat)  # B N K C

        # split out heads for rel pos emb

        rel_pos_emb = rearrange(rel_pos_emb, 'b i j (h d) -> b h i j d', h=h)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product

        q = q.contiguous().view(
            bs * n, -1)[idx, :].view(bs, h, n, num_neighbors, -1)
        k = k.contiguous().view(
            bs * n, -1)[idx, :].view(bs, h, n, num_neighbors, -1)
        qk_rel = q - k

        # expand values (B x N x C) -> (B x N x k x C)
        v = v.contiguous().view(
            bs * n, -1)[idx, :].view(bs, h, n, num_neighbors, -1)

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first

        attn_mlp_input = qk_rel + rel_pos_emb
        attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')

        sim = self.attn_mlp(attn_mlp_input)

        # attention
        attn = sim.softmax(dim=-1)
        attn = F.normalize(attn, dim=-2)

        # aggregate

        v = rearrange(v, 'b h i j d -> b i j (h d)')
        agg = torch.einsum('b d i j, b i j d -> b i d', attn, v).contiguous()

        # combine heads

        del q
        del k
        del v
        return self.to_out(agg)
