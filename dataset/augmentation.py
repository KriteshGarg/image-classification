import torch
from torchvision.transforms import Compose, Pad, RandomCrop, RandomHorizontalFlip, FiveCrop, Lambda, Normalize

from collections.abc import Sequence


class CifarTransform(object):
    @staticmethod
    def get_transform(mean, std, config, transform_type="train"):
        mean = mean
        std = std
        pad = config.pad
        pad_mode = config.pad_mode
        size = config.image_size

        if transform_type == "train":
            flip_lr_p = config.horizontal_flip_prob
            mask_size = config.cutout_size
            cutout_prob = config.cutout_prob
            mask_color = config.cutout_color
            return lambda x, y: dict(images=Compose([Pad(pad, padding_mode=pad_mode), RandomCrop(size),
                                                     RandomHorizontalFlip(flip_lr_p),
                                                     Cutout(mask_size=mask_size, p=cutout_prob,
                                                            mask_color=mask_color), Normalize(mean, std)])(x),
                                     labels=torch.tensor(y, dtype=torch.long))
        elif transform_type == "visualize":
            return lambda x, y: dict(images=Compose([Normalize(mean, std)])(x),
                                     labels=torch.tensor(y, dtype=torch.long))
        else:
            return lambda x, y: dict(images=Compose([Pad(pad, padding_mode=pad_mode),  FiveCrop(size),
                                                     Lambda(lambda crops: torch.stack([
                                                                Normalize(mean, std)(crop)
                                                                for crop in crops]))])(x),
                                     labels=torch.stack([torch.tensor(y, dtype=torch.long) for i in range(5)], dim=0))


class Cutout(torch.nn.Module):

    def __init__(self, mask_size, p, mask_color=(0, 0, 0)):
        super().__init__()
        if not isinstance(mask_size, (int, Sequence)):
            raise TypeError(f"mask_size should be of type (int or sequence) found :{type(mask_size)}")
        if isinstance(mask_size, int):
            self.mask_size = [mask_size] * 2
        elif isinstance(mask_size, Sequence):
            self.mask_size = mask_size
            if len(mask_size) != 2:
                raise ValueError(f"mask_size sequence should be of size 2 found: {len(mask_size)}")
        self.p = p
        self.mask_color = torch.tensor(mask_color)
        self.mask_color = self.mask_color.view(3, 1, 1)

    def forward(self, img):
        if torch.rand(1) < self.p:
            d, h, w = img.shape
            x1 = int(torch.rand(1) * (h - self.mask_size[0]))
            y1 = int(torch.rand(1) * (w - self.mask_size[1]))
            patch = torch.ones((3, self.mask_size[0], self.mask_size[1]))
            patch = patch * self.mask_color
            img[:, y1:y1 + self.mask_size[0], x1:x1 + self.mask_size[1]] = patch
        return img

    def __repr__(self):
        return self.__class__.__name__ + f"(p={self.p}, mask_size={self.mask_size}, mask_color={self.mask_color})"
