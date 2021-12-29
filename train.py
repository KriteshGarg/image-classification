from config import Config
from dataset import CifarDataset
from dataset import Cifar10
from models import Resnet18

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import shutil
import random
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    logger.info(f"Placed initial seed to : {seed}")


class Trainer(object):
    def __init__(self, run_dir, config):
        self.run_dir = run_dir
        self.config = config

        self.device = torch.device("cuda") if torch.cuda.is_available() and self.config.device == "cuda" \
            else torch.device("cpu")

        self.model = Resnet18(num_classes=self.config.num_classes)
        self.model = self.model.to(self.device)
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Total number of params: {pytorch_total_params}")

        # initiate optimizer
        self.optimizer = Adam(self.model.parameters(), lr=float(self.config.min_lr), weight_decay=0.0005)

        # initiate loss/criterion
        self.criterion = CrossEntropyLoss()

        # initiate warmup steps

        # read and split data from disk
        cifar = Cifar10(self.config.data_dir, val_size=self.config.val_ratio)
        data_dict = cifar.extract(dataset_type="train")

        train_data = data_dict["train"]
        val_data = data_dict["val"]

        test_data_dict = cifar.extract(dataset_type="test")
        test_data = test_data_dict["test"]

        # initiate train, val and test data set
        train_dataset = CifarDataset(train_data["images"], train_data["labels"], data_dict["mean"], data_dict["std"],
                                     self.config)
        val_dataset = CifarDataset(val_data["images"], val_data["labels"], data_dict["mean"], data_dict["std"],
                                   self.config, dataset_type='val')
        test_dataset = CifarDataset(test_data["images"], test_data["labels"], data_dict["mean"], data_dict["std"],
                                    self.config, dataset_type='test')

        # initiate train, val and test data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.config.train_bs, shuffle=True,
                                       num_workers=self.config.workers, drop_last=False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.config.val_bs, num_workers=self.config.workers,
                                     drop_last=False)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.val_bs, num_workers=self.config.workers,
                                      drop_last=False)

        # initiate lr scheduler
        self.scheduler = OneCycleLR(self.optimizer, max_lr=float(self.config.max_lr), pct_start=self.config.pct_start,
                                    div_factor=float(self.config.max_lr) / float(self.config.min_lr),
                                    steps_per_epoch=len(self.train_loader), epochs=self.config.num_epochs)

    def train(self):
        best_loss = np.inf
        for epoch_num in range(self.config.num_epochs):

            logger.info("Learning rate at epoch {} is {:.1E}".format(epoch_num, self.scheduler.get_last_lr()[0]))
            writer.add_scalar("train/lr", self.scheduler.get_last_lr()[0], epoch_num)

            train_loss, train_acc = self.train_one_epoch(epoch_num)
            writer.add_scalar('train/loss', train_loss, epoch_num)
            writer.add_scalar('train/acc', train_acc, epoch_num)

            val_loss, val_acc = self.eval_one_epoch(epoch_num)
            writer.add_scalar('val/loss', val_loss, epoch_num)
            writer.add_scalar('val/acc', val_acc, epoch_num)

            if val_loss < best_loss:
                logger.info(
                    "evaluating at epoch {} for val_acc: {} and val_loss {}".format(epoch_num, val_acc, val_loss))

                best_loss = val_loss

                test_loss, test_acc = self.eval_one_epoch(epoch_num, eval_type="test")
                writer.add_scalar('test/loss', test_loss, epoch_num)
                writer.add_scalar('test/acc', test_acc, epoch_num)

                self.save_checkpoint(self.model.__class__.__name__, self.model,
                                     accuracies=[train_acc, val_acc, test_acc],
                                     losses=[train_loss, val_loss, test_loss],
                                     epoch=epoch_num)

    def train_one_epoch(self, epoch_num):
        """
        train one epoch
        """
        preds = []
        ground_truths = []
        training_loss = 0
        # train for num_epoch
        for batch_idx, item in enumerate(tqdm(self.train_loader)):
            images = item["images"].to(device=self.device)
            labels = item["labels"].to(device=self.device)

            # generate probabilities
            probs = self.model(images)

            # get loss
            loss = self.criterion(probs, labels)
            training_loss += loss
            # clear gradient
            self.optimizer.zero_grad()
            # compute gradient backward
            loss.backward()
            # optimize params
            self.optimizer.step()
            # append preds
            preds.extend(torch.max(probs, dim=-1)[1].tolist())
            ground_truths.extend(labels.tolist())

            # step lr scheduler
            self.scheduler.step()
        acc = accuracy_score(ground_truths, preds)
        training_loss /= len(self.train_loader)
        logger.info("train accuracy and loss at the end of {} epoch : {}, {}".format(epoch_num, acc, training_loss))
        return training_loss, acc

    def eval_one_epoch(self, epoch_num, eval_type="val"):
        """
        evaluate or test on one epoch
        """
        with torch.no_grad():
            loader = self.val_loader if eval_type == "val" else self.test_loader
            preds = []
            ground_truths = []
            loss = 0
            for item in tqdm(loader):
                current_bs = item["images"].shape[0]
                # merge ncrops and batches
                images = item["images"].view(-1, item["images"].shape[-3], item["images"].shape[-2],
                                             item["images"].shape[-1]).to(device=self.device)
                labels = item["labels"].view(-1).to(device=self.device)
                # generate probabilities
                logits = self.model(images)
                # append preds
                logit_crop_avg = logits.view(current_bs, 5, -1).mean(1)
                labels_avg = labels.view(current_bs, 5, -1)[:, 0, 0]
                preds.extend(torch.max(logit_crop_avg, dim=-1)[1].tolist())
                ground_truths.extend(labels_avg.tolist())
                # get loss
                loss += self.criterion(logits, labels)
            acc = accuracy_score(ground_truths, preds)
            loss /= len(loader)
            logger.info(f"{eval_type} accuracy and loss at the end of {epoch_num} epoch : {acc}, {loss}")
        return loss, acc

    def save_checkpoint(self, model_type, model, accuracies, losses, epoch):
        """
        Save a checkpoint
        """
        train_acc, val_acc, test_acc = accuracies
        train_loss, val_loss, test_loss = losses
        checkpoint_dict = dict(epoch=epoch, model_state_dict=model.state_dict(),
                               train_acc=train_acc, train_loss=train_loss,
                               val_acc=val_acc, val_loss=val_loss,
                               test_acc=val_acc, test_loss=val_loss)
        checkpoint_file = '{}_best_model.pth'.format(model_type)

        checkpoint_path = os.path.join(self.run_dir, checkpoint_file)

        torch.save(checkpoint_dict, checkpoint_path)

        logger.info('Saved a checkpoint at test acc {:.3f} and loss {:.3f} to {}'.format(test_acc, test_loss,
                                                                                         checkpoint_path))


if __name__ == "__main__":
    config_path = './config/config.yml'
    cifar_config = Config(config_path)
    base_run_dir = os.path.join(cifar_config.run_dir, "train")
    os.makedirs(base_run_dir, exist_ok=True)
    current_run_idx = len(os.listdir(base_run_dir))
    current_run_dir = os.path.join(base_run_dir, "experiment_" + str(current_run_idx))
    os.makedirs(current_run_dir, exist_ok=False)

    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S', level=logging.INFO, filename=os.path.join(current_run_dir,
                                                                                               "train.log"))

    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())

    logger.info(f"Placing all current run related documents in: {current_run_dir}")
    seed_everything(cifar_config.seed)

    # add tensorboard summary writer for plotting
    writer = SummaryWriter(current_run_dir)

    # move a copy of config yml to the experiment directory
    shutil.copy(config_path, os.path.join(current_run_dir, os.path.basename(config_path) + ".backup"))
    trainer = Trainer(current_run_dir, cifar_config)
    trainer.train()
