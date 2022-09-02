import torch
import torch.nn as nn
import torch.nn.init as init

from models.dgcnn import get_graph_feature


class PositionEmbedding(nn.Module):
    '''Adapted from Transform Block (Transform_Net) of DGCNN'''

    def __init__(self, args, c_in=3):
        super(PositionEmbedding, self).__init__()
        self.k = args.k
        self.c_in = c_in

        self.conv1 = ResidualConvLayer(c_in * 2, 64, 128, layer=2)
        self.conv2 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear = nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

        self.transform = nn.Linear(256, c_in * c_in)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(c_in, c_in))

    def forward(self, x):
        batch_size = x.size(0)

        # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = get_graph_feature(x, k=self.k)
        # (batch_size, 3*2, num_points, k) -> (batch_size, 128, num_points, k)
        t = self.conv1(t)
        # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        t = t.max(dim=-1, keepdim=False)[0]

        # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        t = self.conv2(t)
        # (batch_size, 1024, num_points) -> (batch_size, 1024)
        t = t.max(dim=-1, keepdim=False)[0]

        # (batch_size, 1024) -> (batch_size, 512) -> (batch_size, 256)
        t = self.linear(t)

        # (batch_size, 256) -> (batch_size, 3*3)
        t = self.transform(t)
        # (batch_size, 3*3) -> (batch_size, 3, 3)
        t = t.view(batch_size, self.c_in, self.c_in).contiguous()

        # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1).contiguous()
        # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)
        # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = x.transpose(2, 1)

        return x


class ResidualConvLayer(nn.Module):
    def __init__(self, c_in, c_mid, c_out, layer=1, dp1=0, dp2=0) -> None:
        super().__init__()

        assert 1 <= layer <= 2, "Convolution only allowed for 1D and 2D"

        if layer == 1:
            layer = nn.Conv1d
            bn = nn.BatchNorm1d
        else:
            layer = nn.Conv2d
            bn = nn.BatchNorm2d

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Sequential(
            layer(c_in, c_mid, 1, bias=False),
            bn(c_mid),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(dp1)
        )
        self.conv2 = nn.Sequential(
            layer(c_mid, c_out, 1, bias=False),
            bn(c_out),
        )
        self.reshape = layer(c_in, c_out, 1, bias=False)
        self.dp = nn.Dropout(dp2)        

    def forward(self, x):
        # initial activation
        a = self.conv1(x)
        # convolution after activation
        x1 = self.conv2(a)
        # F(a) + x
        y = self.act(x1 + self.reshape(x))
        # dropout
        return self.dp(y)
