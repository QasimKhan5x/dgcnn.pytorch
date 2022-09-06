import math
import random

import torch
import torch.nn.functional as F


def translate_pointcloud(pointcloud):
    xyz1 = torch.distributions.uniform.Uniform(2/3, 3/2).sample([3])
    xyz2 = torch.distributions.uniform.Uniform(-0.2, 0.2).sample([3])
    pointcloud[:, :3] = torch.add(torch.multiply(pointcloud[:, :3], xyz1), xyz2)
    return pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N = pointcloud.size(0)
    C = 3
    pointcloud[:, :3] += torch.clamp(sigma * torch.randn(N, C), -1 * clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = math.pi * 2 * torch.randn(size=(1,))
    rotation_matrix = torch.tensor(
        [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
    pointcloud[:, [0, 2]] = torch.matmul(
        pointcloud[:, [0, 2]], rotation_matrix)  # random rotation (x,z)
    return pointcloud


def scale_pointcloud(pointcloud):
    s = torch.distributions.uniform.Uniform(0.9, 1.1).sample([3])
    s = torch.diag(s)
    pointcloud[:, :3] = torch.matmul(pointcloud[:, :3], s)
    return pointcloud


def center_and_normalize(pointcloud):
    pointcloud[:, :3] = pointcloud[:, :3] - pointcloud[:, :3].mean(axis=0, keepdim=True)
    pointcloud[:, :3] = F.normalize(pointcloud[:, :3], p=2.0, dim=0)
    return pointcloud


def normalize_rgb(pointcloud):
    color_mean = [0.5136457, 0.49523646, 0.44921124]
    color_std = [0.18308958, 0.18415008, 0.19252081]

    pointcloud[:, 3:6] /= 255.0
    pointcloud[:, 3:6] = (pointcloud[:, 3:6] - color_mean) / color_std
    return pointcloud


def drop_rgb(pointcloud, color_drop=0.2):
    should_drop = torch.rand(1).item() < color_drop
    pointcloud[:, 3:6] *= should_drop
    return pointcloud


def rgb_autocontrast(pointcloud):
    if torch.rand(1).item() < 0.2 and pointcloud[:, 3:6].mean() > 0.1:
        lo = pointcloud[:, 3:6].min(1, keepdims=True)[0]
        hi = pointcloud[:, 3:6].max(1, keepdims=True)[0]
        scale = 255 / (hi - lo)
        contrast = (pointcloud[:, 3:6] - lo) * scale
        blend = torch.rand(1).item()
        pointcloud[:, 3:6] = (1 - blend) * pointcloud[:,
                                                      3:6] + blend * contrast
    return pointcloud


def apply_transforms(pointcloud, norm_xyz=True, rgb=True, norm_rgb=True):
    if norm_xyz:
        pointcloud = center_and_normalize(pointcloud)
    transforms = [translate_pointcloud, jitter_pointcloud, rotate_pointcloud, scale_pointcloud]
    if rgb:
        if norm_rgb:
            pointcloud = normalize_rgb(pointcloud)
        transforms.extend([drop_rgb, rgb_autocontrast])
    choices = torch.randint(2, size=(len(transforms), ))
    random.shuffle(transforms)
    for func, choice in zip(transforms, choices):
        if choice.item():
            pointcloud = func(pointcloud)
    return pointcloud
