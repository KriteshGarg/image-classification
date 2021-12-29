# Image Classification

The task of assigning an input image one label from a fixed set of categories. This is one 
of the core problems in Computer Vision that, despite its simplicity, has a large variety 
of practical applications.

## Results:
* LR finder :
  * Please go through 'lr_finder.log' file and 'loss_wrt_lr.png' image to understand max lr analysis
* Please go to './classification/code/image-classification/runs/train/experiment_0/' for training results:
  * 'train.log': Logs of training process 
  * tf events file : Recording of train, test and val. loss, acc and lr 
  * 'Resnet18_best_model.pth': Best model generated during training with test acc 0.908 and loss 0.369
  * 'config.yaml.backup': Just a back up of config file taken at run time for future refrence
* Please go to './classification/code/image-classification/runs/visualize/experiment_0/' for grad_cam results
  * 'images': for grad_cam view of wrong labels 
  * visualize.log: for logs of visualization

## Installation

#### Install Anaconda 3 (use link https://www.anaconda.com/products/individual)
    conda env create -f environment.yml
    conda activate image_classification

## Config :

* data_dir: "../../data/resnet/cifar-10-batches-py"
* Training Params
  * device: "cuda" # device to be used: "cuda" or "cpu" 
  * run_dir: "./runs" # base directory to store run results
  * num_epochs: 200 # number of epochs for training
  * train_bs: 512 # training batch size
  * val_bs: 512 # validation and test batch size
  * val_ratio: 0.05 # training and validation split for cifar10 dataset from training data
  * max_lr: 1e-3 # maximum lr allowed for one cycle scheduler 
  * min_lr: 1e-5 # minimum lr allowed  for one cycle scheduler
  * pct_start: 0.2 # The percentage of the cycle (in number of steps) spent increasing the learning rate.
  * workers: 8 # num of workers used by torch dataloader
  * seed: 10 # seed for consistent results across runs

* Transforms/Augmentations Used
  * pad: 4 # padding 
  * pad_mode: "reflect" # Type of padding. Should be: constant, edge, reflect or symmetric
  * image_size: 32 # input image size 
  * horizontal_flip_prob: 0.5 # probability for horizontal flip
  * cutout_size: 8 # size of cutout 
  * cutout_prob: 0.5 # probabilty of cutout prob
  * cutout_color: [0, 0, 0] # color to be filled in the cutout color
  
* visualize
  * checkpoint_path: "./runs/train/experiment_0/Resnet18_best_model.pth"
  * mean: [0.49141567945480347, 0.48216670751571655, 0.44657519459724426]
  * std: [0.2469293624162674, 0.24336205422878265, 0.2614584267139435]

## Steps to run 
      
#### Activate environment
    conda activate image_classification
    cd image-classification

#### Finding LR 
    python lr_finder.py 

#### Training : modify "./config/config.yml" if required 
    python train.py
      
#### Visualize     
    python visualize_grad_cam.py