import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange

from models.dgcnn import get_graph_feature, knn


def exists(val):
    return val is not None


def max_value(t):
    return torch.finfo(t.dtype).max


class PositionEmbedding(nn.Module):
    '''Adapted from Transform Block of DGCNN'''

    def __init__(self, args):
        super(PositionEmbedding, self).__init__()
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
        self.linear = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x0 = get_graph_feature(x, k=self.k)

        # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        t = self.conv1(x0)
        # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        t = self.conv2(t)
        # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        t = t.max(dim=-1, keepdim=False)[0]

        # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        t = self.conv3(t)
        # (batch_size, 1024, num_points) -> (batch_size, 1024)
        t = t.max(dim=-1, keepdim=False)[0]

        # (batch_size, 1024) -> (batch_size, 512) -> (batch_size, 256)
        t = self.linear(t)

        # (batch_size, 256) -> (batch_size, 3*3)
        t = self.transform(t)
        # (batch_size, 3*3) -> (batch_size, 3, 3)
        t = t.view(batch_size, 3, 3).contiguous()

        # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1).contiguous()
        # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)
        # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = x.transpose(2, 1)

        return x


class PointTransformerLayer(nn.Module):
    def __init__(
        self,
        args,
        pos_mlp_hidden_dim=64,
        attn_mlp_hidden_mult=4,
    ):
        super().__init__()
        self.num_neighbors = args.k
        
        input_dim = args.emb_dim
        inner_dim = 64

        self.size = input_dim

        self.to_qkv = nn.Linear(input_dim, inner_dim * 3, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, inner_dim)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(inner_dim, inner_dim * attn_mlp_hidden_mult),
            nn.ReLU(),
            nn.Linear(inner_dim * attn_mlp_hidden_mult, inner_dim),
        )

        self.reshape = nn.Linear(inner_dim, input_dim)

    def forward(self, x, mask, pos, ):

        bs, n, num_neighbors = x.shape[0], x.shape[1], self.num_neighbors

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # calculate relative positional embeddings
        idx = knn(pos, k=num_neighbors).view(-1)
        pos_nn = pos.contiguous().view(bs * n, -1)[idx,
                                      :].view(bs, n, num_neighbors, 3)
        pos_repeat = pos.contiguous().view(bs, n, 1, 3).repeat(1, 1, num_neighbors, 1)
        rel_pos_emb = self.pos_mlp(pos_nn - pos_repeat) # B N K C

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product
        q = q.contiguous().view(bs * n, -1)[idx, :].view(bs, n, num_neighbors, -1)
        k = k.contiguous().view(bs * n, -1)[idx, :].view(bs, n, num_neighbors, -1)
        qk_rel = q - k

        # prepare mask
        if exists(mask):
            mask = mask[:, :, None] * mask[:, None, :]

        # expand values (B x N x C) -> (B x N x k x C)
        v = v.contiguous().view(
            bs * n, -1)[idx, :].view(bs, n, num_neighbors, -1)
        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # masking
        if exists(mask):
            mask_value = -max_value(sim)
            sim.masked_fill_(~mask[..., None], mask_value)

        # attention
        attn = sim.softmax(dim=-1)
        attn = F.normalize(attn, dim=-2)
        # aggregate
        agg = torch.einsum('b i j d, b i j d -> b i d', attn, v)
        return self.reshape(agg)


class MultiheadPointTransformerLayer(nn.Module):
    def __init__(
        self,
        args,
        dim_head=64,
        pos_mlp_hidden_dim=64,
        attn_mlp_hidden_mult=4,
    ):
        super().__init__()
        self.heads = args.n_heads
        inner_dim = dim_head * self.heads

        self.num_neighbors = args.k
        self.size = args.emb_dim
    

        self.to_qkv = nn.Linear(args.emb_dim, inner_dim * 3, bias=False)
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

    def forward(self, x, mask, pos):
        bs, n, h, num_neighbors = x.shape[0], x.shape[1], self.heads, self.num_neighbors
        
        # get queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # split out heads

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        
        # calculate relative positional embeddings
        idx = knn(pos, k=num_neighbors).view(-1)
        pos_nn = pos.contiguous().view(bs * n, -1)[idx,
                                                   :].view(bs, n, num_neighbors, 3)
        pos_repeat = pos.contiguous().view(bs, n, 1, 3).repeat(1, 1, num_neighbors, 1)
        rel_pos_emb = self.pos_mlp(pos_nn - pos_repeat)  # B N K C

        # split out heads for rel pos emb

        rel_pos_emb = rearrange(rel_pos_emb, 'b i j (h d) -> b h i j d', h=h)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product

        q = q.contiguous().view(
            bs * n, -1)[idx, :].view(bs, h, n, num_neighbors, -1)
        k = k.contiguous().view(
            bs * n, -1)[idx, :].view(bs, h, n, num_neighbors, -1)
        qk_rel = q - k

        # prepare mask

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i 1') * \
                rearrange(mask, 'b j -> b 1 j')

        # expand values (B x N x C) -> (B x N x k x C)
        v = v.contiguous().view(
            bs * n, -1)[idx, :].view(bs, h, n, num_neighbors, -1)

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first

        attn_mlp_input = qk_rel + rel_pos_emb
        attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')

        sim = self.attn_mlp(attn_mlp_input)

        # masking

        if exists(mask):
            mask_value = -max_value(sim)
            mask = rearrange(mask, 'b i j -> b 1 i j')
            sim.masked_fill_(~mask, mask_value)

        # attention
        attn = sim.softmax(dim=-1)
        attn = F.normalize(attn, dim=-2)

        # aggregate

        v = rearrange(v, 'b h i j d -> b i j (h d)')
        agg = torch.einsum('b d i j, b i j d -> b i d', attn, v)

        # combine heads

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

    def forward(self, query, key, value, mask, pos):
        bs, n, h, num_neighbors = query.shape[0], query.shape[1], self.heads, self.num_neighbors
        
        # get queries, keys, values

        q, k, v = self.w_q(query), self.w_k(key), self.w_v(value)

        # split out heads

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        
        # calculate relative positional embeddings
        idx = knn(pos, k=num_neighbors).view(-1)
        pos_nn = pos.contiguous().view(bs * n, -1)[idx,
                                                   :].view(bs, n, num_neighbors, 3)
        pos_repeat = pos.contiguous().view(bs, n, 1, 3).repeat(1, 1, num_neighbors, 1)
        rel_pos_emb = self.pos_mlp(pos_nn - pos_repeat)  # B N K C

        # split out heads for rel pos emb

        rel_pos_emb = rearrange(rel_pos_emb, 'b i j (h d) -> b h i j d', h=h)

        # use subtraction of queries to keys. i suppose this is a better inductive bias for point clouds than dot product

        q = q.contiguous().view(
            bs * n, -1)[idx, :].view(bs, h, n, num_neighbors, -1)
        k = k.contiguous().view(
            bs * n, -1)[idx, :].view(bs, h, n, num_neighbors, -1)
        qk_rel = q - k

        # prepare mask

        if exists(mask):
            mask = rearrange(mask, 'b i -> b i 1') * \
                rearrange(mask, 'b j -> b 1 j')

        # expand values (B x N x C) -> (B x N x k x C)
        v = v.contiguous().view(
            bs * n, -1)[idx, :].view(bs, h, n, num_neighbors, -1)

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first

        attn_mlp_input = qk_rel + rel_pos_emb
        attn_mlp_input = rearrange(attn_mlp_input, 'b h i j d -> b (h d) i j')

        sim = self.attn_mlp(attn_mlp_input)

        # masking

        if exists(mask):
            mask_value = -max_value(sim)
            mask = rearrange(mask, 'b i j -> b 1 i j')
            sim.masked_fill_(~mask, mask_value)

        # attention
        attn = sim.softmax(dim=-1)
        attn = F.normalize(attn, dim=-2)

        # aggregate

        v = rearrange(v, 'b h i j d -> b i j (h d)')
        agg = torch.einsum('b d i j, b i j d -> b i d', attn, v)

        # combine heads

        return self.to_out(agg)



