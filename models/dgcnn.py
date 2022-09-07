import torch
import torch.nn as nn


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1).contiguous()

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k, knn_only=False, disp_only=False):
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


class EdgeConv(nn.Module):
    def __init__(self, c_in, c_out, k, double_conv=False) -> None:
        super().__init__()

        self.k = k
        if double_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(c_in * 2, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(c_out, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(c_in * 2, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )

    def forward(self, x):
        # x (B x c_in x N)

        # B x c_in x N x k
        x = get_graph_feature(x, self.k)
        # B x c_out x N x k
        x = self.conv(x)
        # B x c_out x N x 1
        x_max = x.max(dim=-1, keepdim=False)[0]

        return x_max


class InvResMLP(nn.Module):
    '''https://arxiv.org/abs/2206.04670'''
    def __init__(self, c_in, c_out, k) -> None:
        super().__init__()

        # edge conv + high res mlp
        self.ec = nn.Sequential(
            EdgeConv(c_in, c_out, k=k, double_conv=True),
            nn.Conv1d(c_out, 1024, 1, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.revert = nn.Sequential(
            nn.Conv1d(1024, c_out, 1, bias=False),
            nn.BatchNorm1d(c_out)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.reshape = nn.Conv1d(c_in, c_out, 1, bias=False)

    def forward(self, x):
        y = self.ec(x)
        y = self.revert(y)

        return self.act(y + self.reshape(x))


class DGCNN(nn.Module):
    def __init__(self, args, c_in):
        super(DGCNN, self).__init__()

        emb_dims = args.emb_dims
        self.k = args.k

        self.conv1 = EdgeConv(c_in, emb_dims // 8, k=self.k, double_conv=True)
        self.conv2 = EdgeConv(emb_dims // 8 + 3, emb_dims //
                              8, k=self.k, double_conv=True)
        self.conv3 = EdgeConv(emb_dims // 8 + 3, emb_dims //
                              4, k=self.k, double_conv=True)
        self.conv4 = EdgeConv(emb_dims // 4 + 3, emb_dims //
                              2, k=self.k, double_conv=True)

        self.resize1 = nn.Conv1d(c_in, emb_dims // 8, 1, bias=False)
        self.resize2 = nn.Conv1d(emb_dims // 8, emb_dims // 4, 1, bias=False)
        self.resize3 = nn.Conv1d(emb_dims // 4, emb_dims // 2, 1, bias=False)

        self.conv5 = nn.Sequential(
            nn.Conv1d(emb_dims + 3, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, x):
        num_points = x.size(2)

        # B C N 
        x1 = self.conv1(x) + self.resize1(x)
        # add original points (C += 3)
        x1 = torch.cat((x[:, :3], x1), dim=1)
        # B x (64) x N (add features except first 3 dims bcz they are coordinates)
        x2 = self.conv2(x1) + x1[:, 3:]
        x2 = torch.cat((x[:, :3], x2), dim=1)
        x3 = self.conv3(x2) + self.resize2(x2[:, 3:])
        x3 = torch.cat((x[:, :3], x3), dim=1)
        x4 = self.conv4(x3) + self.resize3(x3[:, 3:])
        # B x C + 3 x N (x1 already has the required dims so no need to repeat it)
        point_ftrs = torch.cat((x1, x2[:, 3:], x3[:, 3:], x4), dim=1)
        # mlp on point features B x C x N
        x = self.conv5(point_ftrs)
        # global features (B x emb_dims x num_points)
        global_ftrs = x.max(dim=-1, keepdim=True)[0].repeat(1, 1, num_points)
        return point_ftrs, global_ftrs


class DGCNN_PNeXt(nn.Module):
    def __init__(self, args, c_in):
        super(DGCNN_PNeXt, self).__init__()

        emb_dims = args.emb_dims
        self.k = args.k

        self.conv1 = InvResMLP(c_in, emb_dims, k=self.k)

    def forward(self, xyz, x):
        num_points = xyz.size(2)

        # B C N
        x = torch.cat((xyz, x), dim=1)
        x = self.conv1(x)
        return x
