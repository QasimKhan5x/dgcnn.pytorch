import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from models.dgcnn import get_graph_feature

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


