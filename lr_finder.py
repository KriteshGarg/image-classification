from config import Config
from dataset import Cifar10, CifarDataset
from models import Resnet18

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

import logging
import matplotlib.pyplot as plt


class LRFinder(object):
    def __init__(self, min_lr=1e-8, lr_step=10, n_steps=10, num_classes=10, smoothing_factor=0.98, config=None):
        self.config = config if config is not None else Config('./config/config.yml')

        # Initiating device
        self.device = torch.device("cuda") if torch.cuda.is_available() and self.config.device == "cuda" \
            else torch.device("cpu")

        self.n_steps = n_steps
        self.smoothing_factor = smoothing_factor

        # initiating model
        # TODO update model to custom model
        self.model = Resnet18(num_classes)
        self.model = self.model.to(self.device)

        # initiate optimizer
        self.optimizer = Adam(self.model.parameters(), lr=min_lr, weight_decay=0)

        # read and split data from disk
        cifar = Cifar10(self.config.data_dir)
        data_dict = cifar.extract(dataset_type="train")
        train_data = data_dict["train"]

        # initiate loss/criterion
        self.criterion = CrossEntropyLoss()

        # initiate train, val and test data set
        train_dataset = CifarDataset(train_data["images"], train_data["labels"], data_dict["mean"], data_dict["std"],
                                     self.config)

        # initiate train, val and test data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.train_bs, shuffle=True,
                                       num_workers=self.config.workers, drop_last=True)

        # initiate lr
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=lr_step)

    def get_ema_loss_wrt_lr(self):
        idx = 0
        ema_losses = list()
        lr_ticks = list()
        while idx < self.n_steps:
            item = next(iter(self.train_loader))
            images = item["images"].to(device=self.device)
            labels = item["labels"].to(device=self.device)

            # generate probabilities
            probs = self.model(images)

            # get loss
            loss = self.criterion(probs, labels)

            if ema_losses:
                ema_loss = (1.0 - self.smoothing_factor) * ema_losses[-1] + self.smoothing_factor * loss.item()
            else:
                ema_loss = loss.item()

            ema_losses.append(ema_loss)
            lr_ticks.append('{:.1E}'.format(self.scheduler.get_last_lr()[0]))

            logger.info(f"For Lr: {lr_ticks[-1]}, loss: {ema_loss}")

            # break if ema loss is more then 4 times min ema loss
            if ema_loss > 4 * min(ema_losses):
                logger.info(f"Moving to plotting as ema loss {ema_loss} is more 4 times min exponential moving loss: "
                            f"{min(ema_losses)}")
                break

            # update for next iteration
            # clear gradient
            self.optimizer.zero_grad()
            # compute gradient backward
            loss.backward()
            # optimize params
            self.optimizer.step()
            self.scheduler.step()
            idx += 1

        logger.info(f"Value for losses : {ema_losses}")
        logger.info(f"Values for Learning Rates : {lr_ticks}")
        return lr_ticks, ema_losses

    @staticmethod
    def plot_loss(lr_ticks, ema_losses):
        plt.plot(lr_ticks, ema_losses, marker="o", label="ema_loss_wrt_lr")
        plt.xlabel('Learning Rate', labelpad=14)
        plt.ylabel('Loss')
        plt.title('Loss wrt LR ')
        plt.legend()
        plt.savefig("loss_wrt_lr.png")


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO, filename="lr_finder.log")

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    lr_finder = LRFinder()
    _lr_ticks, _ema_losses = lr_finder.get_ema_loss_wrt_lr()
    lr_finder.plot_loss(_lr_ticks, _ema_losses)
