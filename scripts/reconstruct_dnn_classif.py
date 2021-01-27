import sys
sys.path.append('.')

import os
import numpy as np
import torch
from torch import nn
import time
import soundfile as sf
from tqdm import tqdm
import librosa
import h5py as h5

from packages.processing.stft import stft
from packages.models.Video_Net import DeepVAD_video
from packages.models.utils import f1_loss
#from utils import count_parameters

# Dataset
# dataset_size = 'subset'
dataset_size = 'complete'

dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list

dataset_type = 'test'
# labels = 'labels'
labels = 'vad_labels'
upsampled = True

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:1"
device = torch.device(cuda_device if cuda else "cpu")

# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
# hop_percent = math.floor((1 / (wlen_sec * visual_frame_rate)) * 1e4) / 1e4  # hop size as a percentage of the window length
hop_percent = 0.25 # hop size as a percentage of the window length
win = 'hann' # type of window
center = False # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect' # This argument is ignored if center = False
pad_at_end = True # pad audio file at end to match same size after stft + istft
dtype = 'complex64'

## Video
visual_frame_rate_i = 30 # initial visual frames per second
visual_frame_rate_o = 1 / (wlen_sec * hop_percent)
width = 67
height = 67
crf = 0 #set the constant rate factor to 0, which is lossless

## IBM
quantile_fraction = 0.999
quantile_weight = 0.999

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Classifier
# if labels == 'labels':
    # classif_name = 'classif_batchnorm_before_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_59.58'
    # x_dim = 513 # frequency bins (spectrogram)
    # y_dim = 513
    # h_dim_cl = [128, 128]
    # batch_norm = True
    # std_norm = False
    # eps = 1e-8

if labels == 'vad_labels':
    classif_name = 'Video_Classifier_upsampled_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_005_vloss_0.32'
    # x_dim = 513 # frequency bins (spectrogram)
    # y_dim = 1
    # h_dim_cl = [128, 128]
    lstm_layers = 2
    lstm_hidden_size = 1024
    std_norm = True
    batch_norm = False
    eps = 1e-8

# Data directories
input_video_dir = os.path.join('data',dataset_size,'processed/')
input_speech_dir = os.path.join('data',dataset_size,'raw/')
classif_dir = os.path.join('models', classif_name + '.pt')
classif_data_dir = os.path.join('data', dataset_size, 'models', classif_name + '/')
output_h5_dir = input_video_dir + os.path.join(dataset_name + '_statistics_upsampled' + '.h5')

# Data normalization
if std_norm:
    print('Load mean and std')
    with h5.File(output_h5_dir, 'r') as file:
        mean = file['X_train_mean'][:]
        std = file['X_train_std'][:]

    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

def main():

    # Log file
    file = open('output.log','w') 
    print('Torch version: {}'.format(torch.__version__))
    print('Device: %s' % (device))
    if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())

    classifier = DeepVAD_video(lstm_layers, lstm_hidden_size)
    classifier.load_state_dict(torch.load(classif_dir, map_location=cuda_device))
    if cuda: classifier = classifier.to(device)

    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    # Create file list
    mat_file_paths = video_list(input_video_dir=input_video_dir,
                            dataset_type=dataset_type,
                            upsampled=upsampled)

    audio_file_paths = speech_list(input_speech_dir=input_speech_dir,
                            dataset_type=dataset_type)

    #Init variables to compute f1-score
    total_accuracy, total_precision, total_recall, total_f1_score = (0., 0., 0., 0.)

    #TODO: paralllelize over 4 GPUs
    for i, (mat_file_path, audio_file_paths) in tqdm(enumerate(zip(mat_file_paths, audio_file_paths))):
        
        # select utterance
        h5_file_path = input_video_dir + mat_file_path

        # Open HDF5 file
        with h5.File(h5_file_path, 'r') as file:
            x = np.array(file["X"][:])
            y = np.array(file["Y"][:])
            length = [x.shape[-1]]
        
        x = torch.Tensor(x).to(device)
        y = torch.Tensor(y).to(device)

        # Transpose to match PyTorch
        x = x.T # (frames,channels,width,height)

        # Normalize power spectrogram
        if std_norm:
            x -= mean.T
            x /= (std + eps).T

        # Classify
        y_hat_soft = classifier(x, length)
        #TODO: make it stateful
        y_hat_soft = torch.sigmoid(y_hat_soft)
        y_hat_hard = (y_hat_soft > 0.5).int()
        
        # F1-score
        accuracy, precision, recall, f1_score = f1_loss(y_hat_hard=torch.flatten(y_hat_hard), y=torch.flatten(y), epsilon=eps)
        total_accuracy += accuracy
        total_precision += precision
        total_recall+= recall
        total_f1_score += f1_score

        # Output file
        output_path = classif_data_dir + mat_file_path
        output_path = os.path.splitext(output_path)[0]

        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))

        # save y_hat_soft and y_hat_hard as numpy array
        y_hat_soft = y_hat_soft.cpu().numpy()
        y_hat_hard = y_hat_hard.cpu().numpy()

        np.save(output_path + '_y_hat_soft.npy', y_hat_soft)
        np.save(output_path + '_y_hat_hard.npy', y_hat_hard)
    
    # Compute total accuracy, precision, recall, F1-score
    n_utt = len(mat_file_paths)
    total_accuracy /= n_utt
    total_precision /= n_utt
    total_recall /= n_utt
    total_f1_score /= n_utt

    print("[Test]       Accuracy: {:.2f}    Precision: {:.2f}    \n"
    "Recall: {:.2f}     F1_score: {:.2f}".format(total_accuracy, total_precision, total_recall, total_f1_score))

if __name__ == '__main__':
    main()