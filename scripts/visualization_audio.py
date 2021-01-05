import sys
sys.path.append('.')

import os
import numpy as np
import torch
import h5py  # to read .mat files
from scipy.fftpack import idct

# Dataset
dataset_name = 'CSR-1-WSJ-0'
if dataset_name == 'CSR-1-WSJ-0':
    from python.dataset.csr1_wjs0_dataset import speech_list, read_dataset

# Settings
dataset_type = 'test'

dataset_size = 'subset'
#dataset_size = 'complete'

# Labels
# labels = 'labels'
labels = 'vad_labels'

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:0"
device = torch.device(cuda_device if cuda else "cpu")

# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'

## IBM
quantile_fraction = 0.999
quantile_weight = 0.999

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

# Data directories
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
classif_dir = os.path.join('models', classif_name + '.pt')
classif_data_dir = 'data/' + dataset_size + '/models/' + classif_name + '/'
output_data_dir = os.path.join('data', dataset_size, 'models', classif_name + '/')

def main():
    mat_file_path = os.path.join("data/complete/matlab_raw", self.data[i]) + ".mat"
    with h5py.File(mat_file_path, 'r') as f:
        for key, value in f.items():
            matlab_frames_list_per_user = np.array(value)

    video_frames = torch.FloatTensor(matlab_frames_list_per_user.shape[0], 3, 67, 67)

    for frame in range(matlab_frames_list_per_user.shape[0]):
        data_frame = matlab_frames_list_per_user[frame]  # data frame will be shortened to "df" below
        reshaped_df = data_frame.reshape(67, 67)
        idct_df = idct(idct(reshaped_df).T).T
        epsilon = 1e-8  # for numerical stability in the operation below:
        normalized_df = idct_df / (matlab_frames_list_per_user.flatten().max() + epsilon) * 255.0
        rotated_df = np.rot90(normalized_df, 3)
        rgb_rotated_df = np.stack((rotated_df,) * 3, axis=0)
        video_frames[frame] = torch.from_numpy(rgb_rotated_df)

if __name__ == '__main__':
    main()