from dataset import CifarTransform

from torch.utils.data import Dataset

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class CifarDataset(Dataset):
    def __init__(self, images, labels, mean, std, config, dataset_type="train"):
        self.images = images
        self.labels = labels
        self.mean = mean
        self.std = std
        self.transform = CifarTransform.get_transform(mean=mean, std=std, config=config, transform_type=dataset_type)
        logger.info(f"Initiated Cifar-10: {dataset_type} dataset")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            transformed_dict = self.transform(image, label)
        else:
            transformed_dict = {"images": image, "labels": label}

        return transformed_dict
