# Imports

## Add the modules to the system path
import os
import sys
import argparse
sys.path.append(os.path.join(".."))

import logging

class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


logging.basicConfig(filename='logging.log',  # Log filename
                    filemode='a',  # Append mode, so logs are not overwritten
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Include timestamp
                    level=logging.INFO,  # Logging level
                    datefmt='%Y-%m-%d %H:%M:%S')  # Timestamp formatlog_file = open('logfile.log', 'w', buffering=1)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Set logging level for console
logging.getLogger('').addHandler(console_handler)

# Redirect stdout and stderr to logging
sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

## test

## Libs
from random import shuffle
import glob
import tifffile
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True

## Own modules
import utils
from train import Trainer



## parser

# Check if the script is running on the server by looking for the environment variable
if os.getenv('RUNNING_ON_SERVER') == 'true':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Process data directory.')
    parser.add_argument('--data_dir', type=str, help='Path to the data directory')
    parser.add_argument('--project_name', type=str, help='Name of the project')
    parser.add_argument('--train_continue', type=str, default='off', choices=['on', 'off'],
                        help='Flag to continue training: "on" or "off" (default: "off")')
    parser.add_argument('--load_epoch', type=int, default=1, 
                        help='Epoch number from which to continue training (default: 1)')


    # Parse arguments
    args = parser.parse_args()

    # Now you can use args.data_dir as the path to your data
    data_dir = args.data_dir
    project_name = args.project_name 
    train_continue = args.train_continue
    load_epoch = args.load_epoch
    project_dir = os.path.join('/g', 'prevedel', 'members', 'Rauscher', 'projects', 'standard_N2V')

    print(f"Using data directory: {data_dir}")
    print(f"Project name: {project_name}")
    print(f"Train continue: {train_continue}")
    print(f"Load epoch: {load_epoch}")
else:

    data_dir = os.path.join('Z:\\', 'members', 'Rauscher', 'data', 'OCT-data-3')
    project_name = 'test-1'
    train_continue = 'off'
    load_epoch = 1
    project_dir = os.path.join('Z:\\', 'members', 'Rauscher', 'projects', 'standard_N2V')
    
    print(f"Not running on server, using default data directory: {data_dir}")
    print(f"Default project name: {project_name}")


path_results = os.path.join(project_dir, project_name, 'results-1')
path_train_dataset = os.path.join(data_dir, 'train')
path_val_dataset = os.path.join(data_dir, 'validation')

# Create all the other paths based on the results folder

# Make a folder to store results
res_folder = os.path.join(path_results, 'training_results')
os.makedirs(res_folder, exist_ok=True)

# Make a folder to store the log files
log_folder = os.path.join(path_results, 'log_files')
os.makedirs(log_folder, exist_ok=True)

log_train_folder = os.path.join(log_folder, 'train')
os.makedirs(log_train_folder, exist_ok=True)

log_val_folder = os.path.join(log_folder, 'val')
os.makedirs(log_val_folder, exist_ok=True)

# Make a folder for the checkpoints
checkpoint_folder = os.path.join(path_results, 'checkpoints')
os.makedirs(checkpoint_folder, exist_ok=True)


# List all folders in the results folder to make sure all folder exists
output_files = os.listdir(path_results)
print("*****Output Folder*****")
print("List of all folder in the results path:")
print(output_files)
print("***********************")


## Load image stack as dataset 
print(path_train_dataset)
filenames_train = glob.glob(os.path.join(path_train_dataset, "*.TIFF"))
filenames_val = glob.glob(os.path.join(path_val_dataset, "*.TIFF"))
print("On following file will be trained:  ", filenames_train[0])
print("On following file will be validated:  ", filenames_val[0])

file_train = tifffile.imread(filenames_train[0])
file_val = tifffile.imread(filenames_val[0])

# file_train = file_train[:,0].squeeze()
# file_val = file_val[:,0].squeeze()


# Select the blindspot parameters #
percent_blindpixel = 2
picking_radius = 5
#*********************************#

# Select the training parameters #
# Z x Y x X
input_size = [16, 64, 64]

# #Training-to-#Validation ratio
train_val_fraction = 0.5

# Training epochs
epoch = 300

# Batch size
batch_size = 4

# Logger frequencies
display_freq = 50
model_storing_freq = 50
#*********************************#

# Parameter dictionary
parameter_dict= {}
# paths
# In case norm-factors are stored somewhere, not necessary
parameter_dict['dir_norm_factors'] = os.path.join("no_norm_factors_stored")
parameter_dict['dir_checkpoint'] = checkpoint_folder
parameter_dict['dir_log'] = log_folder
parameter_dict['dir_result'] = res_folder
# training state
parameter_dict['train_continue'] = train_continue
# hyperparameters
parameter_dict['num_epoch'] = epoch
# batch size
parameter_dict['batch_size'] = batch_size
# adam optimizer
parameter_dict['lr'] = 0.001
parameter_dict['optim'] = 'adam'
parameter_dict['beta1'] = 0.5
parameter_dict['beta2'] = 0.999
# colormap
parameter_dict['cmap'] = 'gray'
# size of the input patches
parameter_dict['ny'] = input_size[2]
parameter_dict['nx'] = input_size[1]
parameter_dict['nz'] = input_size[0]
# channel dimension
parameter_dict['nch'] = 1

# augmentation data for the N2V augmenter
parameter_dict['perc_pixel'] = percent_blindpixel
parameter_dict['n2v_neighborhood_radius'] = picking_radius
parameter_dict['structN2Vmask'] = None
# logger parameter
parameter_dict['num_freq_disp'] = display_freq
parameter_dict['num_freq_save'] = model_storing_freq
# datasets
parameter_dict['train_dataset'] = [file_train]
parameter_dict['val_dataset'] = [file_val[:-int(train_val_fraction*len(file_train))]]


# Generate Trainer
trainer = Trainer(parameter_dict)
# Start training
print("*****Start of Training*****")
trainer.train()
print("*****End of Training*******")