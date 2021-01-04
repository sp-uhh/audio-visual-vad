import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py  # to read .mat files
from scipy.fftpack import idct


base_path = "data/complete/matlab_raw/"


class VideoFrames(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
        self.index = np.arange(len(self.data))
 
    def __getitem__(self, i):
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

        start_point = np.random.randint(matlab_frames_list_per_user.shape[0] - self.seq_length)
        video_frames = video_frames[start_point:(start_point + self.seq_length)]
        labels = np.load("{}{}.npy".format("data/complete/labels/", self.data[i]))[start_point + self.seq_length]
        return video_frames, labels

    def __len__(self):
        return len(self.data)