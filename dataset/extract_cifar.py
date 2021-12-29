import os
import cv2
import sys
import logging
import numpy as np
from sklearn.model_selection import train_test_split

import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


class Cifar10(object):
    def __init__(self, data_dir="../../data/resnet/cifar-10-batches-py", val_size=0.20):
        self.data_dir = data_dir
        self.val_size = val_size
        if os.path.exists(self.data_dir):
            self.data_files = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if
                               "data_batch" in file]
            self.test_data_files = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if
                                    "test_batch" in file]
            self.meta_data_files = [os.path.join(self.data_dir, file) for file in os.listdir(self.data_dir) if
                                    "batches.meta" in file]
        else:
            logger.error(f"path does not exist: {data_dir}", exc_info=True)
            sys.exit(1)

    @staticmethod
    def unpickle(data_file):
        import pickle
        with open(data_file, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        return data_dict

    def extract_file(self, data_file, save_dir=None, label_names=None):
        data_dict = self.unpickle(data_file)

        image_data = data_dict[b'data']
        image_data = np.stack(np.split(image_data, 3, axis=1), axis=1)
        images = image_data.reshape((-1, 3, 32, 32))
        file_names = data_dict[b'filenames']
        labels = data_dict[b'labels']

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            for image_idx, file_name in enumerate(data_dict[b'filenames']):
                label_name = label_names[labels[image_idx]] if label_names else labels[image_idx]
                label_dir = os.path.join(save_dir, label_name)
                os.makedirs(label_dir, exist_ok=True)
                # moving color channel to last and converting rgb to bgr
                image = np.moveaxis(images[image_idx], 0, 2)[:, :, ::-1]
                cv2.imwrite(os.path.join(label_dir, file_name), image)
            return images, file_names, labels
        else:
            return images, file_names, labels

    def extract(self, dataset_type="train", create_folder_dataset=False):
        label_names = list()
        for meta_file in self.meta_data_files:
            meta_dict = self.unpickle(meta_file)
            meta_label_names = meta_dict[b"label_names"]
            label_names.extend(meta_label_names)
        if dataset_type == "train":
            images = np.zeros((50000, 3, 32, 32), dtype=np.uint8)
            filenames, labels = list(), list()
            for idx, data_file in enumerate(self.data_files):
                if create_folder_dataset:
                    batch_images, batch_file_names, batch_labels = self.extract_file(data_file, save_dir=os.path.join(
                        self.data_dir, "train"), label_names=label_names)
                else:
                    batch_images, batch_file_names, batch_labels = self.extract_file(data_file)
                images[idx * 10000:(idx + 1) * 10000] = batch_images
                filenames.extend(batch_file_names)
                labels.extend(batch_labels)

            train_images, val_images, train_labels, val_labels = train_test_split(images, labels,
                                                                                  test_size=self.val_size,
                                                                                  random_state=1, stratify=labels)

            train_images = torch.tensor(train_images)/255.0
            val_images = torch.tensor(val_images)/255.0

            mean_r = train_images[:, 0, :, :].mean()
            mean_g = train_images[:, 1, :, :].mean()
            mean_b = train_images[:, 2, :, :].mean()

            std_r = train_images[:, 0, :, :].std()
            std_g = train_images[:, 1, :, :].std()
            std_b = train_images[:, 2, :, :].std()

            logger.info(f"mean and std of the training dataset : {(mean_r.item(), mean_g.item(), mean_b.item())}, "
                        f"{(std_r.item(), std_g.item(), std_b.item())}")

            return dict(train=dict(images=train_images, labels=train_labels),
                        val=dict(images=val_images, labels=val_labels), mean=(mean_r, mean_g, mean_b),
                        std=(std_r, std_g, std_b))

        if dataset_type == "test":
            test_images = np.zeros((10000, 3, 32, 32), dtype=np.uint8)
            test_filenames, test_labels = list(), list()
            for idx, data_file in enumerate(self.test_data_files):
                if create_folder_dataset:
                    batch_images, batch_file_names, batch_labels = self.extract_file(data_file, save_dir=os.path.join(
                        self.data_dir, "test"), label_names=label_names)
                else:
                    batch_images, batch_file_names, batch_labels = self.extract_file(data_file)
                test_images[idx * 10000:(idx + 1) * 10000] = batch_images
                test_filenames.extend(batch_file_names)
                test_labels.extend(batch_labels)
            test_images = torch.tensor(test_images)/255.0
            return dict(test=dict(images=test_images, labels=test_labels))
        logger.error(f"dataset_type can only be: (train, test) but found: {dataset_type}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO, filename=os.path.join("extract_dataset.log"))

    cifar10 = Cifar10()
    data_dict = cifar10.extract()
