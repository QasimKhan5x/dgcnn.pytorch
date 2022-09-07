
import torch
import torch.nn as nn

from dgl.geometry import farthest_point_sampler

from models.dgcnn import DGCNN_PNeXt, knn


def select_and_expand(pc, k, num_centroids):
    '''
    Select num_centroids points from the point cloud
    using FPS and expand those centroids into regions (sub-clouds)
    within which each centroid is paired with its k closest neighbors
    '''
    # pc (B x N x 3)
    B = pc.size(0)
    N = pc.size(1)
    # B x 3 x N
    pc = pc.transpose(1, 2)

    nn_idx = knn(pc, k)
    fps_idx = farthest_point_sampler(pc, num_centroids).view(-1)

    # point cloud in which every point is coupled with its k nearest neighbors
    pc_nn = pc.view(B * N, 3)[nn_idx, :].view(B, N, 3, k)
    # select the centroids from the above
    centroids = pc_nn.view(B * N, 3)[fps_idx, :].view(B, num_centroids, k, 3)
    return centroids


class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()


class Encoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        # number of centroids per encoder layer
        region_sizes = args.region_sizes
        num_layers = len(region_sizes)


    def forward(self, xyz, ftrs):
        return None

class Decoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

    def forward(self, xyz, ftrs):
        return None



class Net(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        emb_dims = args.emb_dims
        self.ftr_extractor = DGCNN_PNeXt(args, c_in=emb_dims)
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, xyz, x):
        '''
        xyz ( B x 3 x N) 3-dimensional coordinates
        x (B x C x N) C-dimensional features
        '''
        x = self.ftr_extractor(xyz, x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
