import sys
sys.path.append('.')

import os
import numpy as np
import torch, torchaudio
from torch import nn
import time
from tqdm import tqdm
import h5py as h5 # to read .mat files
import torch.multiprocessing as multiprocessing
import math

from packages.models.utils import f1_loss
from packages.processing.stft import stft_pytorch
from packages.models.Video_Net import DeepVAD_video
#from utils import count_parameters

from packages.visualization import display_multiple_signals

# Parameters
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import proc_video_audio_pair_dict

# Settings
dataset_type = 'test'
# dataset_type = 'validation'

# dataset_size = 'subset'
dataset_size = 'complete'

# Labels
labels = 'vad_labels'
# labels = 'ibm_labels'
upsampled = False
dct = False
norm_video = True

# System
cuda = torch.cuda.is_available()
cuda_device = "cuda:4"
device = torch.device(cuda_device if cuda else "cpu")

## Video
visual_frame_rate_i = 30 # initial visual frames per second
width = 67
height = 67
crf = 0 #set the constant rate factor to 0, which is lossless

# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = math.floor((1 / (wlen_sec * visual_frame_rate_i)) * 1e4) / 1e4  # hop size as a percentage of the window length
# hop_percent = 0.25 # hop size as a percentage of the window length
win = 'hann' # type of window
center = False # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect' # This argument is ignored if center = False
pad_at_end = True # pad audio file at end to match same size after stft + istft
dtype = 'complex64'

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Classifier
if labels == 'vad_labels':
    # classif_name = 'Video_Classifier_vad_loss_eps_upsampled_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_007_vloss_4.48'
    # x_dim = 513
    # y_dim = 1
    # lstm_layers = 2
    # lstm_hidden_size = 1024
    # batch_norm=False
    # std_norm =True
    # eps = 1e-8

    # classif_name = 'Video_Classifier_vad_loss_eps_upsampled_align_shuffle_nopretrain_nonorm_batch64_noseqlength_end_epoch_100/Video_Net_epoch_012_vloss_4.53'
    # x_dim = 513
    # y_dim = 1
    # lstm_layers = 2
    # lstm_hidden_size = 1024
    # batch_norm=False
    # std_norm =False
    # eps = 1e-8

    # classif_name = 'Video_Classifier_vad_dct_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_003_vloss_4.76'
    # x_dim = 513
    # y_dim = 1
    # lstm_layers = 2
    # lstm_hidden_size = 1024
    # batch_norm=False
    # std_norm =True
    # eps = 1e-8

    # # classif_name = 'Video_Classifier_vad_dct_align_shuffle_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_003_vloss_4.24'
    # classif_name = 'Video_Classifier_vad_pixel_lstm_512_nopretrain_normimage_batch64_noseqlength_end_epoch_100/'
    # x_dim = 513
    # y_dim = 1
    # lstm_layers = 2
    # # lstm_hidden_size = 1024
    # lstm_hidden_size = 512
    # batch_norm=False
    # std_norm =True
    # eps = 1e-8

    # classif_name = 'Video_Classifier_vad_full_dct_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_002_vloss_5.52'
    # x_dim = 513
    # y_dim = 1
    # lstm_layers = 2
    # lstm_hidden_size = 1024
    # batch_norm=False
    # std_norm =True
    # eps = 1e-8

    # classif_name = 'Video_Classifier_vad_full_dct_align_shuffle_nopretrain_nonorm_batch64_noseqlength_end_epoch_100/Video_Net_epoch_004_vloss_5.47'
    # x_dim = 513
    # y_dim = 1
    # lstm_layers = 2
    # lstm_hidden_size = 1024
    # batch_norm=False
    # std_norm =False
    # eps = 1e-8

    # classif_name = 'Video_Classifier_vad_loss_eps_upsampled_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_007_vloss_4.48'
    # x_dim = 513
    # y_dim = 1
    # lstm_layers = 2
    # lstm_hidden_size = 1024
    # batch_norm=False
    # std_norm =True
    # eps = 1e-8

    # classif_name = 'Video_Classifier_vad_resnet_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_004_vloss_4.47'
    # x_dim = 513
    # y_dim = 1
    # lstm_layers = 2
    # lstm_hidden_size = 1024
    # batch_norm=False
    # std_norm =True
    # eps = 1e-8

    # classif_name = 'Video_Classifier_vad_resnet_normvideo_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_006_vloss_4.38'
    # classif_name = 'Video_Classifier_vad_resnet_normvideo_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_002_vloss_4.36'
    # classif_name = 'Video_Classifier_vad_resnet_normvideo2_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_001_vloss_4.55'
    # classif_name = 'Video_Classifier_vad_resnet_normvideo2_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_006_vloss_4.25'
    # classif_name = 'Video_Classifier_vad_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_004_vloss_4.34'
    # classif_name = 'Video_Classifier_vad_resnet_normvideo4_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_003_vloss_4.16'
    classif_name = 'Video_Classifier_vad_resnet_normvideo4_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_006_vloss_4.31'
    x_dim = 513
    y_dim = 1
    lstm_layers = 2
    lstm_hidden_size = 1024
    batch_norm=False
    std_norm =True
    eps = 1e-8

if labels == 'ibm_labels':
    classif_name = 'Video_Classifier_ibm_nodropout_normdataset_batch64_noseqlength_end_epoch_100/'
    x_dim = 513
    y_dim = 513
    lstm_layers = 2
    lstm_hidden_size = 1024
    batch_norm=False
    std_norm =True
    eps = 1e-8

# GPU Multiprocessing
# nb_devices = torch.cuda.device_count()
nb_devices = 4
nb_process_per_device = 1

# Data directories
processed_data_dir = os.path.join('data',dataset_size,'processed/')
classif_dir = os.path.join('models', classif_name + '.pt')
classif_data_dir = 'data/' + dataset_size + '/models/' + classif_name + '/'
output_data_dir = os.path.join('data', dataset_size, 'models', classif_name + '/')

def process_utt(classifier, mean, std, video_file_path, audio_file_path, device):

    # Read target
    h5_file_path = processed_data_dir + audio_file_path

    with h5.File(h5_file_path, 'r') as file:
        y = np.array(file["Y"][:])
        y = torch.LongTensor(y) # Convert y to Tensor for f1-score

    # Read video
    h5_file_path = processed_data_dir + video_file_path

    # Read files
    with h5.File(h5_file_path, 'r') as file:
        x = np.array(file["X"][:])
        x = torch.Tensor(x)

    # Set to device
    x = x.to(device)

    # Transpose to match PyTorch
    lengths = [x.shape[-1]]
    x = x.unsqueeze(0).transpose(0, -1).squeeze() # (frames, height, width)

    # Normalize power spectrogram
    if std_norm:
        x -= mean.T
        x /= (std + eps).T

    # Add dimension
    x = x[None]

    # Classify
    y_hat_soft = classifier(x, lengths)
    y_hat_soft = y_hat_soft[...,0]  # Reduce last dimension for librosa.display
    y_hat_soft = y_hat_soft.cpu()
    y_hat_soft = torch.sigmoid(y_hat_soft.detach())
    y_hat_hard = (y_hat_soft > 0.5).int()

    # save y_hat_soft / y_hat_hard as .pt
    output_path = classif_data_dir + video_file_path
    output_path = os.path.splitext(output_path)[0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save(y_hat_hard, output_path + '_y_hat_hard.pt')
    torch.save(y_hat_soft, output_path + '_y_hat_soft.pt')

def process_sublist(device, sublist, classifier):

    if cuda: classifier = classifier.to(device)

    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    # Data normalization
    # output_h5_dir = processed_data_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'pixel' + '_statistics.h5')
    # output_h5_dir = processed_data_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'dct' + '_statistics.h5')
    # output_h5_dir = processed_data_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'pixel_dct' + '_statistics.h5')
    # output_h5_dir = processed_data_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_statistics.h5')
    output_h5_dir = processed_data_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'normvideo' + '_statistics.h5')

    with h5.File(output_h5_dir, 'r') as file:
        mean = file['X_train_mean'][:]
        std = file['X_train_std'][:]

    mean = torch.tensor(mean).to(device)
    std = torch.tensor(std).to(device)

    for (video_file_path, audio_file_path) in sublist:
        process_utt(classifier, mean, std, video_file_path, audio_file_path, device)

def main():
    file = open('output.log','w')

    print('Torch version: {}'.format(torch.__version__))
    print('Device: %s' % (device))
    if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())

    # Start context for GPU multiprocessing
    ctx = multiprocessing.get_context('spawn')

    classifier = DeepVAD_video(lstm_layers, lstm_hidden_size, y_dim)
    classifier.load_state_dict(torch.load(classif_dir, map_location=cuda_device))
    # if cuda: classifier = classifier.to(device)

    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    # Dict mapping video to clean speech
    video_file_paths, audio_file_paths = proc_video_audio_pair_dict(input_video_dir=processed_data_dir,
                                            dataset_type=dataset_type,
                                            labels=labels,
                                            upsampled=upsampled,
                                            dct=dct,
                                            norm_video=norm_video)
    # Convert dict to tuples
    video_clean_pair_paths = list(zip(video_file_paths, audio_file_paths))

    # Split list in nb_devices * nb_processes_per_device
    b = np.array_split(video_clean_pair_paths, nb_devices*nb_process_per_device)

    # Assign each list to a process
    b = [(4 + i%nb_devices, sublist, classifier) for i, sublist in enumerate(b)]

    print('Start evaluation')
    # start = time.time()
    t1 = time.perf_counter()

    with ctx.Pool(processes=nb_process_per_device*nb_devices) as multi_pool:
        multi_pool.starmap(process_sublist, b)

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()