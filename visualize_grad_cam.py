from models import Resnet18
from config import Config
from dataset import Cifar10, CifarDataset

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import os
import cv2
import random
import logging
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"Placed initial seed to : {seed}")


class VisualizeGradCam(object):
    def __init__(self, checkpoint_path, config):
        self.config = config
        self.device = torch.device("cuda") if torch.cuda.is_available() and self.config.device == "cuda" \
            else torch.device("cpu")

        self.model = Resnet18(self.config.num_classes)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.target_layers = [self.model.layer_4_basis_block_2]

        # create test data loader
        cifar = Cifar10(self.config.data_dir, val_size=self.config.val_ratio)

        test_data_dict = cifar.extract(dataset_type="test")
        self.test_data = test_data_dict["test"]

        test_dataset = CifarDataset(self.test_data["images"], self.test_data["labels"], self.config.mean,
                                    self.config.std, self.config, dataset_type='visualize')

        self.test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)

        # Construct the CAM object once, and then re-use it on many images:
        use_cuda = True if torch.cuda.is_available() and self.config.device == "cuda" \
            else False
        self.grad_cam = GradCAM(model=self.model, target_layers=self.target_layers, use_cuda=use_cuda)

    def gerate_grad_cam(self, is_wrong_label=False):
        for idx, item in enumerate(tqdm(self.test_loader)):
            images = item["images"].to(device=self.device)
            labels = item["labels"].to(device=self.device)

            input_tensor = images  # Create an input tensor image for your model..
            # Note: input_tensor can be a batch tensor with several images!

            target_category = labels

            if is_wrong_label:
                probs = self.model(input_tensor)
                pred_label = torch.max(probs, dim=-1)[1]
                input_tensor = input_tensor[torch.where(pred_label != target_category)]

            if input_tensor.shape[0] == 0:
                continue
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = self.grad_cam(input_tensor=input_tensor, target_category=target_category)

            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]

            rgb_img = self.test_data["images"][idx]
            rgb_img = torch.moveaxis(rgb_img, 0, 2)
            visualization = show_cam_on_image(rgb_img.numpy(), grayscale_cam, use_rgb=True)
            cv2.imwrite(os.path.join(image_dir, f"{idx}.jpg"), visualization)
            logger.info(f"created visualization for image at idx: {idx}")


if __name__ == "__main__":
    config_path = './config/config.yml'
    cifar_config = Config(config_path)
    base_run_dir = os.path.join(cifar_config.run_dir, "visualize")
    os.makedirs(base_run_dir, exist_ok=True)
    current_run_idx = len(os.listdir(base_run_dir))
    current_run_dir = os.path.join(base_run_dir, "experiment_" + str(current_run_idx))
    image_dir = os.path.join(current_run_dir, "images")
    os.makedirs(current_run_dir, exist_ok=False)
    os.makedirs(image_dir, exist_ok=False)

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO, filename=os.path.join(current_run_dir,
                                                                                               "visualize.log"))

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    logger.info(f"Placing all current run related documents in: {current_run_dir}")
    seed_everything(cifar_config.seed)

    visualize_grad_cam = VisualizeGradCam(checkpoint_path=cifar_config.checkpoint_path, config=cifar_config)

    visualize_grad_cam.gerate_grad_cam(is_wrong_label=True)
