import numpy as np
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, coord, feat, label):
        for t in self.transforms:
            coord, feat, label = t(coord, feat, label)
        return coord, feat, label


class ToTensor(object):
    def __call__(self, coord, feat, label):
        coord = torch.from_numpy(coord)
        if not isinstance(coord, torch.FloatTensor):
            coord = coord.float()
        feat = torch.from_numpy(feat)
        if not isinstance(feat, torch.FloatTensor):
            feat = feat.float()
        label = torch.from_numpy(label)
        if not isinstance(label, torch.LongTensor):
            label = label.long()
        return coord, feat, label


class RandomRotate(object):
    def __init__(self, angle=[0, 0, 1]):
        self.angle = angle

    def __call__(self, coord, feat, label):
        angle_x = np.random.uniform(-self.angle[0], self.angle[0]) * np.pi
        angle_y = np.random.uniform(-self.angle[1], self.angle[1]) * np.pi
        angle_z = np.random.uniform(-self.angle[2], self.angle[2]) * np.pi
        cos_x, sin_x = np.cos(angle_x), np.sin(angle_x)
        cos_y, sin_y = np.cos(angle_y), np.sin(angle_y)
        cos_z, sin_z = np.cos(angle_z), np.sin(angle_z)
        R_x = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        R_y = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        R_z = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])
        R = np.dot(R_z, np.dot(R_y, R_x))
        coord = np.dot(coord, np.transpose(R))
        return coord, feat, label


class RandomScale(object):
    def __init__(self, scale=[0.9, 1.1], anisotropic=False):
        self.scale = scale
        self.anisotropic = anisotropic

    def __call__(self, coord, feat, label):
        scale = np.random.uniform(
            self.scale[0], self.scale[1], 3 if self.anisotropic else 1)
        coord *= scale
        return coord, feat, label


class RandomTranslate(object):
    def __init__(self, shift=[0.2, 0.2, 0]):
        self.shift = shift

    def __call__(self, coord, feat, label):
        shift_x = np.random.uniform(-self.shift[0], self.shift[0])
        shift_y = np.random.uniform(-self.shift[1], self.shift[1])
        shift_z = np.random.uniform(-self.shift[2], self.shift[2])
        coord += [shift_x, shift_y, shift_z]
        return coord, feat, label


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            coord[:, 0] = -coord[:, 0]
        if np.random.rand() < self.p:
            coord[:, 1] = -coord[:, 1]
        return coord, feat, label


class RandomJitter(object):
    def __init__(self, sigma=0.005, clip=0.02):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, coord, feat, label):
        assert (self.clip > 0)
        jitter = np.clip(
            self.sigma * np.random.randn(coord.shape[0], 3), -1 * self.clip, self.clip)
        coord += jitter
        return coord, feat, label


class ChromaticNormalize(object):
    def __init__(self):
        self.mean = np.array([0.5136457, 0.49523646, 0.44921124])
        self.std = np.array([0.18308958, 0.18415008, 0.19252081])

    def __call__(self, coord, feat, label):
        feat[:, :3] /= 255.0
        feat[:, :3] = (feat[:, :3] - self.mean) / self.std
        return coord, feat, label


class ChromaticAutoContrast(object):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            lo = feat[:, :3].min(axis=0, keepdims=True)
            hi = feat[:, :3].max(axis=0, keepdims=True)
            if np.count_nonzero(hi - lo) != (hi - lo).size:
                return coord, feat, label
            scale = 255 / (hi - lo)
            contrast_feat = (feat[:, :3] - lo) * scale
            blend_factor = np.random.rand() if self.blend_factor is None else self.blend_factor
            feat[:, :3] = (1 - blend_factor) * feat[:, :3] + \
                blend_factor * contrast_feat
        return coord, feat, label


class ChromaticTranslation(object):
    def __init__(self, p=0.95, ratio=0.05):
        self.p = p
        self.ratio = ratio

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.ratio
            feat[:, :3] = np.clip(tr + feat[:, :3], 0, 255)
        return coord, feat, label


class ChromaticJitter(object):
    def __init__(self, p=0.95, std=0.005):
        self.p = p
        self.std = std

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            noise = np.random.randn(feat.shape[0], 3)
            noise *= self.std * 255
            feat[:, :3] = np.clip(noise + feat[:, :3], 0, 255)
        return coord, feat, label


class RandomDropColor(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, coord, feat, label):
        if np.random.rand() < self.p:
            feat[:, :3] = 0
        return coord, feat, label


class PointCloudFloorCentering(object):
    """Centering the point cloud in the xy plane
    Args:
        object (_type_): _description_
    """

    def __init__(self,
                 gravity_dim=2,
                 **kwargs):
        self.gravity_dim = gravity_dim

    def __call__(self, coord, feat, label):
        coord -= coord.mean(axis=0, keepdims=True)
        coord[:, self.gravity_dim] -= np.min(coord[:, self.gravity_dim])
        return coord, feat, label


class AppendHeight(object):
    def __init__(self, gravity_dim=2) -> None:
        self.gravity_dim = gravity_dim

    def __call__(self, coord, feat, label):
        height = coord[:, self.gravity_dim:self.gravity_dim+1]
        feat = np.concatenate((feat, height - np.min(height)), axis=-1)
        return coord, feat, label
