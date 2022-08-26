import copy

import torch.nn as nn
import torch.nn.functional as F

from models.attention import VectorAttention

# Part of the code is referred from: http://nlp.seas.harvard.edu/annotated-transformer/

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, src, tgt, pointcloud):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, pointcloud), tgt, pointcloud)

    def encode(self, src, pointcloud):
        return self.encoder(src, pointcloud)

    def decode(self, memory, tgt, pointcloud):
        return self.decoder(tgt, memory, pointcloud)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.BatchNorm1d(layer.size)

    def forward(self, x, pointcloud):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, pointcloud)
        x = self.norm(x.transpose(1, 2).contiguous()
                      ).transpose(1, 2).contiguous()
        return x


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.BatchNorm1d(layer.size)

    def forward(self, x, memory, pointcloud):
        for layer in self.layers:
            x = layer(x, memory, pointcloud)
        x = self.norm(x.transpose(1, 2).contiguous()
                      ).transpose(1, 2).contiguous()
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.BatchNorm1d(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        x = self.norm(x.transpose(1, 2).contiguous()
                      ).transpose(1, 2).contiguous()
        return x + self.dropout(sublayer(x))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, pointcloud):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, pointcloud))
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

    def forward(self, x, memory, pointcloud):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, pointcloud))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, pointcloud))
        return self.sublayer[2](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.norm(F.leaky_relu(self.w_1(x),
         0.1).transpose(2,
          1).contiguous())).transpose(2,
           1).contiguous())


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.emb_dims = args.emb_dim
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.ff_dims = args.ff_dims
        self.n_heads = args.n_heads

        c = copy.deepcopy
        attn = VectorAttention(args)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)

        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(attn),
                                                         c(ff), self.dropout), self.N)
                                    )

    def forward(self, *input):
        # (batch_size, emb_dims, num_points)
        src = input[0]
        # (batch_size, emb_dims, num_points)
        tgt = input[1]
        # (batch_size, 3, num_points)
        pointcloud = input[2]
        # (batch_size, emb_dims, num_points)
        src = src.transpose(2, 1).contiguous()
        # (batch_size, emb_dims, num_points)
        tgt = tgt.transpose(2, 1).contiguous()
        # (batch_size, num_points, emb_dims)
        tgt_embedding = self.model(
            src, tgt, pointcloud).transpose(2, 1).contiguous()
        # (batch_size, num_points, emb_dims)
        src_embedding = self.model(
            tgt, src, pointcloud).transpose(2, 1).contiguous()
        # (batch_size, nclasses, num_points)
        return src_embedding, tgt_embedding
