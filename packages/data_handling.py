import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py as h5 # to read .mat files
from scipy.fftpack import idct
import math


base_path = "data/complete/matlab_raw/"


class VideoFrames(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        self.index = np.arange(len(self.data))
 
    def __getitem__(self, i):
        mat_file_path = os.path.join("data/complete/matlab_raw", self.data[i]) + ".mat"
        with h5.File(mat_file_path, 'r') as f:
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

        start_point = np.random.randint(matlab_frames_list_per_user.shape[0] - self.seq_length)
        video_frames = video_frames[start_point:(start_point + self.seq_length)]
        labels = np.load("{}{}.npy".format("data/complete/labels/", self.data[i]))[start_point + self.seq_length]
        return video_frames, labels

    def __len__(self):
        return len(self.data)

class HDF5SpectrogramLabeledFrames(Dataset):
    def __init__(self, output_h5_dir, dataset_type, rdcc_nbytes, rdcc_nslots):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.output_h5_dir = output_h5_dir
        self.dataset_type = dataset_type
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        with h5.File(self.output_h5_dir, 'r') as file:
            self.dataset_len = file["X_" + dataset_type].shape[-1]

    def open_hdf5(self):
        #We are using 400Mb of chunk_cache_mem here ("rdcc_nbytes" and "rdcc_nslots")
        self.f = h5.File(self.output_h5_dir, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
        
        # Faster to open datasets once, rather than at every call of __getitem__
        self.data = self.f['X_' + self.dataset_type]
        self.labels = self.f['Y_' + self.dataset_type]

    def __getitem__(self, i):
        # Open hdf5 here if num_workers > 0
        if not hasattr(self, 'f'):
            self.open_hdf5()
        return self.data[:,i], self.labels[:,i]

    def __len__(self):
        return self.dataset_len

    def __del__(self): 
        if hasattr(self, 'f'):
            self.f.close()


class HDF5SequenceSpectrogramLabeledFrames(Dataset):
    def __init__(self, output_h5_dir, dataset_type, rdcc_nbytes, rdcc_nslots, seq_length):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.output_h5_dir = output_h5_dir
        self.dataset_type = dataset_type
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        self.seq_length = seq_length
        with h5.File(self.output_h5_dir, 'r') as file:
            self.dataset_len = file["X_" + dataset_type].shape[-1]

    def open_hdf5(self):
        #We are using 400Mb of chunk_cache_mem here ("rdcc_nbytes" and "rdcc_nslots")
        self.f = h5.File(self.output_h5_dir, 'r', rdcc_nbytes=self.rdcc_nbytes, rdcc_nslots=self.rdcc_nslots)
        
        # Faster to open datasets once, rather than at every call of __getitem__
        self.data = self.f['X_' + self.dataset_type]
        self.labels = self.f['Y_' + self.dataset_type]

    def __getitem__(self, i):
        # Open hdf5 here if num_workers > 0
        if not hasattr(self, 'f'):
            self.open_hdf5()
        
        # i = i * self.seq_length

        # data = torch.zeros(self.seq_length, 1)
        # target = torch.zeros(self.seq_length,1)

        # data[i] = torch.tensor(self.data[:,:,][0])
        # target[i] = torch.tensor(self.oudataframe.iloc[idx+i][1])

        # start_point = np.random.randint(matlab_frames_list_per_user.shape[0] - self.seq_length)
        # video_frames = video_frames[start_point:(start_point + self.seq_length)]
        if i < self.seq_length:
            # return smaller sequence
            data = np.array(self.data[...,:i+1])
            labels = np.array(self.labels[...,i:i+1])
            length = data.shape[-1]
            # return torch.Tensor(self.data[...,:i+1]), torch.Tensor(self.labels[...,i:i+1]) # Take only the last label
            return torch.Tensor(data), torch.Tensor(labels), length #, length # Take only the last label
        else:
            # return full sequence
            data = np.array(self.data[...,i+1-self.seq_length:i+1])
            labels = np.array(self.labels[...,i:i+1])
            length = data.shape[-1]
            # return torch.Tensor(self.data[...,i+1-self.seq_length:i+1]), torch.Tensor(self.labels[...,i:i+1]) # Take only the last label
            return torch.Tensor(data), torch.Tensor(labels), length #, length # Take only the last label

    def __len__(self):
        return self.dataset_len
        # return math.ceil(self.dataset_len / self.seq_length)

    def __del__(self): 
        if hasattr(self, 'f'):
            self.f.close()