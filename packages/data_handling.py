import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py as h5 # to read .mat files
from scipy.fftpack import idct
import math
import torch
import torchaudio
from packages.processing.stft import stft_pytorch

# Parameters
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list, proc_noisy_clean_pair_dict

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
        #TODO: change and move to the next sequence without overlapping (or overlapping by half)
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

class HDF5WholeSequenceSpectrogramLabeledFrames(Dataset):
    def __init__(self, output_h5_dir, dataset_type, rdcc_nbytes, rdcc_nslots, seq_length):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.output_h5_dir = output_h5_dir
        self.dataset_type = dataset_type
        self.rdcc_nbytes = rdcc_nbytes
        self.rdcc_nslots = rdcc_nslots
        self.seq_length = seq_length
        with h5.File(self.output_h5_dir, 'r') as file:
            dataset_len = file["X_" + dataset_type].shape[-1]
            self.dataset_len = math.ceil(dataset_len / seq_length)

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
        
        i = i * self.seq_length

        # data = torch.zeros(self.seq_length, 1)
        # target = torch.zeros(self.seq_length,1)

        # data[i] = torch.tensor(self.data[:,:,][0])
        # target[i] = torch.tensor(self.oudataframe.iloc[idx+i][1])

        # start_point = np.random.randint(matlab_frames_list_per_user.shape[0] - self.seq_length)
        # video_frames = video_frames[start_point:(start_point + self.seq_length)]
        
        #TODO: change and move to the next sequence without overlapping (or overlapping by half)
        # return full sequence
        data = np.array(self.data[...,i:i+self.seq_length])
        labels = np.array(self.labels[...,i:i+self.seq_length])
        length = data.shape[-1]
        # return torch.Tensor(self.data[...,i+1-self.seq_length:i+1]), torch.Tensor(self.labels[...,i:i+1]) # Take only the last label
        return torch.Tensor(data), torch.Tensor(labels), length #, length # Take only the last label

    def __len__(self):
        return self.dataset_len
        # return math.ceil(self.dataset_len / self.seq_length)

    def __del__(self): 
        if hasattr(self, 'f'):
            self.f.close()

class WavWholeSequenceSpectrogramLabeledFrames(Dataset):
    def __init__(self, input_video_dir, dataset_type, labels='vad_labels', upsampled=False):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.dataset_type = dataset_type        
        self.input_video_dir = input_video_dir

        # Create file list
        self.mat_file_paths = video_list(input_video_dir=input_video_dir,
                                dataset_type=dataset_type,
                                labels=labels,
                                upsampled=upsampled)
        
        self.dataset_len = len(self.mat_file_paths) # total number of utterances

    def __getitem__(self, i):
        # select utterance
        h5_file_path = self.input_video_dir + self.mat_file_paths[i]

        # Open HDF5 file
        with h5.File(h5_file_path, 'r') as file:
            data = np.array(file["X"][:])
            labels = np.array(file["Y"][:])
            length = data.shape[-1]
        
        return torch.Tensor(data), torch.Tensor(labels), length #, length # Take only the last label

    def __len__(self):
        return self.dataset_len

class NoisyWavWholeSequenceSpectrogramLabeledFrames(Dataset):
    def __init__(self,
                 input_video_dir, dataset_type,
                 dataset_size, labels='vad_labels',
                 fs=16000, wlen_sec=64e-3, win='hann', hop_percent=0.25,
                 center=True, pad_mode='reflect', pad_at_end=True):
        # Do not load hdf5 in __init__ if num_workers > 0
        self.input_video_dir = input_video_dir
        self.dataset_type = dataset_type
        self.dataset_size = dataset_size        
        self.labels = labels

        # STFT parameters
        self.fs = fs
        self.wlen_sec = wlen_sec
        self.win = win
        self.hop_percent = hop_percent
        self.center = center
        self.pad_mode = pad_mode
        self.pad_at_end = pad_at_end

        # Dict mapping noisy speech to clean speech
        self.noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=input_video_dir,
                                                dataset_type=dataset_type,
                                                dataset_size=dataset_size,
                                                labels=labels)

        # Convert dict to tuples
        self.noisy_clean_pair_paths = list(self.noisy_clean_pair_paths.items())

        self.dataset_len = len(self.noisy_clean_pair_paths) # total number of utterances

    def __getitem__(self, i):
        # select utterance
        (proc_noisy_file_path, clean_file_path) = self.noisy_clean_pair_paths[i]
        
        # Read noisy audio
        noisy_speech, fs_noisy_speech = torchaudio.load(self.input_video_dir + proc_noisy_file_path)
        noisy_speech = noisy_speech[0] # 1channel

        # Normalize audio
        noisy_speech = noisy_speech / (torch.max(torch.abs(noisy_speech)))

        # TF representation (PyTorch)
        noisy_speech_tf = stft_pytorch(noisy_speech,
                fs=self.fs,
                wlen_sec=self.wlen_sec,
                win=self.win, 
                hop_percent=self.hop_percent,
                center=self.center,
                pad_mode=self.pad_mode,
                pad_at_end=self.pad_at_end) # shape = (freq_bins, frames)

        # Power spectrogram
        data = noisy_speech_tf[...,0]**2 + noisy_speech_tf[...,1]**2
       
        # Read label
        output_h5_file = clean_file_path

        with h5.File(output_h5_file, 'r') as file:
            label = np.array(file["Y"][:])
            label = torch.Tensor(label)

        # Reduce frames of video or label
        if label.shape[-1] < data.shape[-1]:
            data = data[...,:label.shape[-1]]
        if label.shape[-1] > data.shape[-1]:
            data = label[...,:data.shape[-1]]
        
        # Sequence length
        length = data.shape[-1]
        
        return data, label, length #, length # Take only the last label

    def __len__(self):
        return self.dataset_len