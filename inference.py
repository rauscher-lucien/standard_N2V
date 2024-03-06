# Imports

## Add the modules to the system path
import os
import sys
sys.path.append(os.path.join(".."))

## Libs
import numpy as np
import glob
import tifffile
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
import matplotlib.pyplot as plt

## Own modules
import utils
from transformations import PercentileNormalize3D, PercentileDenormalize3D, ZCrop3D, ToTensor3D, ToNumpy3D
from dataset import InferenceDataset3D
from network import Noise2NoiseUNet3D

project_name = 'test-1'
# my_folder = os.path.join('/g', 'prevedel', 'members', 'Rauscher')
my_folder = os.path.join('Z:', 'members', 'Rauscher')
project_dir = os.path.join(my_folder, 'projects', 'standard_N2V')
path_results = os.path.join(project_dir, project_name, 'results-1')

path_dataset =  os.path.join(my_folder, 'data', 'OCT-data-3')
#********************************************************#

# Create a folder for the inference based on the results folder

# Make a folder to store the inference
inference_folder = os.path.join(path_results, 'inference_results')
os.makedirs(inference_folder, exist_ok=True)

# Define path to the checkpoint folder
checkpoint_folder = os.path.join(path_results, 'checkpoints')

## Load image stack for inference
filenames = glob.glob(os.path.join(path_dataset, "*.TIFF"))
print("Following file will be denoised:  ", filenames[0])

file = tifffile.imread(filenames[0])


# Select the inference parameters #

# Z size of the input substack (should minimum the same as in training)
z_input_size = 16
# Shows the overlap
stride_to_size_ratio=2
# Crop top and bottom z-slices to prevent artefacts
z_crop_width = (2, 2)

#********************************#


# datatype of the original data
data_type = file.dtype
print("The data type of the raw data is:   ", data_type)

# calculate the norm. factors
print("\nThe norm. factors are: ")
min_img, max_img = utils.calc_normfactors(file)

# check if GPU is accessable
if torch.cuda.is_available():
    print("\nGPU will be used.")
    device = torch.device("cuda:0")
else:
    print("\nCPU will be used.")
    device = torch.device("cpu")


## Data handling
# Use the right back-conversation
if data_type == np.uint16:
    norm_func = utils.NormFloat2UInt16(percent=1.0)
elif data_type == np.int16:
    norm_func = utils.NormFloat2Int16(percent=1.0)
else:
    norm_func = utils.NormFloat2UInt8(percent=1.0)

## Transformation
transform = transforms.Compose([PercentileNormalize3D(mi=min_img, ma=max_img),
                                ToTensor3D()
                                ])
transform_inv = transforms.Compose([ToNumpy3D(),
                                    ZCrop3D(z_crop_width),
                                    PercentileDenormalize3D(mi=min_img, ma=max_img),
                                    norm_func
                                   ])
## Inference Dataset and Dataloader
inference_dataset = InferenceDataset3D(file, z_crop_width, z_input_size, stride_to_size_ratio, transform=transform)
inference_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, shuffle=False, num_workers=0)


## Initialize the inference data and for sanity check the input data
size_prediction = inference_dataset.get_output_size()

input_data = np.empty(size_prediction, dtype=np.float64)
inference_data = np.empty(size_prediction, dtype=np.float64)


## Load pretrained model
net = Noise2NoiseUNet3D(in_channels = 1,
                        out_channels = 1,
                        f_maps = 8,
                        final_sigmoid = True).to(device)

net, st_epoch = utils.load(checkpoint_folder, net, device)


## Apply the model
with torch.no_grad():
    net.eval()

    print(f'Number of batches: {len(inference_folder)}')

    # Going through whole dataset
    for batch, data in enumerate(inference_loader):

        input, current_index, norm_factors = data
        cropped_current_index = current_index.squeeze()
        cropped_norm_factors = norm_factors.squeeze() 
        
        input = input.to(device)
        # forward net
        output = net(input)
        ## transform data back
        inference_data[cropped_current_index] += cropped_norm_factors.numpy()[..., None, None] * transform_inv(output).squeeze()
        input_data[cropped_current_index] += cropped_norm_factors.numpy()[..., None, None] * transform_inv(input).squeeze()

# Crop the data to its original size
z_begin, z_end = inference_dataset.get_cropping_indices()
inference_data = inference_data[z_begin:z_end]
input_data = input_data[z_begin:z_end]


plt.figure(figsize=(10,30))

ind = file.shape[0]//2

plt.subplot(131)
plt.title("Raw (Original)")
plt.imshow(file[ind], cmap="gray")

plt.subplot(132)
plt.title("Raw (Input Data)")
plt.imshow(input_data[ind], cmap="gray")

plt.subplot(133)
plt.title("Denoised")
plt.imshow(inference_data[ind], cmap="gray")

plt.tight_layout()
plt.show()