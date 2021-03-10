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

from packages.models.utils import f1_loss
from packages.processing.stft import stft_pytorch
from packages.models.Audio_Net import DeepVAD_audio
#from utils import count_parameters

from packages.visualization import display_multiple_signals

# Parameters
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import proc_noisy_clean_pair_dict

# Settings
dataset_type = 'test'
# dataset_type = 'train'

# dataset_size = 'subset'
dataset_size = 'complete'

# Labels
labels = 'vad_labels'
# labels = 'ibm_labels'

# System 
cuda = torch.cuda.is_available()
cuda_device = "cuda:4"
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

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Classifier
if labels == 'vad_labels':
    classif_name = 'Audio_Classifier_vad_loss_eps_upsampled_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_100/Video_Net_epoch_009_vloss_16.64'
    x_dim = 513 
    y_dim = 1
    lstm_layers = 2
    lstm_hidden_size = 1024 
    batch_norm=False
    std_norm =True
    eps = 1e-8

if labels == 'ibm_labels':
    classif_name = 'Audio_Classifier_ibm_normdataset_batch16_noseqlength_end_epoch_100/Video_Net_epoch_006_vloss_9.22'
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

def process_utt(classifier, mean, std, proc_noisy_file_path, clean_file_path, device):

    # Extract input SNR and noise type
    snr_db = int(proc_noisy_file_path.split('/')[3])
    noise_type = proc_noisy_file_path.split('/')[2]

    # Read target
    h5_file_path = processed_data_dir + clean_file_path

    with h5.File(h5_file_path, 'r') as file:
        y = np.array(file["Y"][:])
        y = torch.LongTensor(y) # Convert y to Tensor for f1-score

    # Read files
    x_t, fs_x = torchaudio.load(processed_data_dir + proc_noisy_file_path)
    x_t = x_t[0] # 1channel

    # x = x/np.max(x)
    T_orig = len(x_t)

    # Normalize audio
    norm_max = torch.max(torch.abs(x_t))
    x_t = x_t / norm_max
    
    # TF representation (PyTorch)
    # Input should be (frames, freq_bins)
    x_tf = stft_pytorch(x_t,
            fs=fs,
            wlen_sec=wlen_sec,
            win=win, 
            hop_percent=hop_percent,
            center=center,
            pad_mode=pad_mode,
            pad_at_end=pad_at_end) # shape = (freq_bins, frames)

    # Power spectrogram
    x = x_tf[...,0]**2 + x_tf[...,1]**2

    # Reduce frames of audio
    if y.shape[-1] < x.shape[-1]:
        x = x[...,:y.shape[-1]]

    # Apply log
    x = torch.log(x + eps)

    # Set to device
    x = x.to(device)

    # Transpose to match PyTorch
    lengths = [x.shape[-1]]
    x = x.T # (frames, freq_bins)

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
    output_path = classif_data_dir + proc_noisy_file_path
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
    if std_norm:
        # # Load mean and variance
        # mean = np.load(os.path.dirname(classif_dir) + '/' + 'trainset_mean.npy')
        # std = np.load(os.path.dirname(classif_dir) + '/' + 'trainset_std.npy')
        
        output_h5_dir = processed_data_dir + os.path.join(dataset_name, 'Noisy', dataset_name + '_' + 'power_spec' + '_statistics.h5')
        with h5.File(output_h5_dir, 'r') as file:
            mean = file['X_train_mean'][:]
            std = file['X_train_std'][:]

        mean = torch.tensor(mean).to(device)
        std = torch.tensor(std).to(device)
    
    for (proc_noisy_file_path, clean_file_path) in sublist:
        process_utt(classifier, mean, std, proc_noisy_file_path, clean_file_path, device)

def main():
    file = open('output.log','w') 

    print('Torch version: {}'.format(torch.__version__))
    print('Device: %s' % (device))
    if torch.cuda.device_count() >= 1: print("Number GPUs: ", torch.cuda.device_count())

    # Start context for GPU multiprocessing
    ctx = multiprocessing.get_context('spawn')

    classifier = DeepVAD_audio(lstm_layers, lstm_hidden_size, y_dim)
    classifier.load_state_dict(torch.load(classif_dir, map_location=cuda_device))
    # if cuda: classifier = classifier.to(device)

    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    # Dict mapping noisy speech to clean speech
    noisy_clean_pair_paths = proc_noisy_clean_pair_dict(input_speech_dir=processed_data_dir,
                                            dataset_type=dataset_type,
                                            dataset_size=dataset_size,
                                            labels=labels)

    # Convert dict to tuples
    noisy_clean_pair_paths = list(noisy_clean_pair_paths.items())

    # Split list in nb_devices * nb_processes_per_device
    b = np.array_split(noisy_clean_pair_paths, nb_devices*nb_process_per_device)
    
    # Assign each list to a process
    b = [(4 + i%nb_devices, sublist, classifier) for i, sublist in enumerate(b)]

    print('Start evaluation')
    # start = time.time()
    t1 = time.perf_counter()

    with ctx.Pool(processes=nb_process_per_device*nb_devices) as multi_pool:
        multi_pool.starmap(process_sublist, b)

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

    # for i, (proc_noisy_file_path, clean_file_path) in tqdm(enumerate(noisy_clean_pair_paths)):
        
    #     # Extract input SNR and noise type
    #     snr_db = int(proc_noisy_file_path.split('/')[3])
    #     noise_type = proc_noisy_file_path.split('/')[2]

    #     # Read target
    #     h5_file_path = processed_data_dir + clean_file_path

    #     with h5.File(h5_file_path, 'r') as file:
    #         y = np.array(file["Y"][:])
    #         y = torch.LongTensor(y) # Convert y to Tensor for f1-score

    #     # Read files
    #     x_t, fs_x = torchaudio.load(processed_data_dir + proc_noisy_file_path)
    #     x_t = x_t[0] # 1channel

    #     # x = x/np.max(x)
    #     T_orig = len(x_t)

    #     # Normalize audio
    #     norm_max = torch.max(torch.abs(x_t))
    #     x_t = x_t / norm_max
        
    #     # TF representation (PyTorch)
    #     # Input should be (frames, freq_bins)
    #     x_tf = stft_pytorch(x_t,
    #             fs=fs,
    #             wlen_sec=wlen_sec,
    #             win=win, 
    #             hop_percent=hop_percent,
    #             center=center,
    #             pad_mode=pad_mode,
    #             pad_at_end=pad_at_end) # shape = (freq_bins, frames)

    #     # Power spectrogram
    #     x = x_tf[...,0]**2 + x_tf[...,1]**2

    #     # Reduce frames of audio
    #     if y.shape[-1] < x.shape[-1]:
    #         x = x[...,:y.shape[-1]]

    #     # Apply log
    #     x = torch.log(x + eps)

    #     # Set to device
    #     x = x.to(device)

    #     # Transpose to match PyTorch
    #     lengths = [x.shape[-1]]
    #     x = x.T # (frames, freq_bins)

    #     # Normalize power spectrogram
    #     if std_norm:
    #         x -= mean.T
    #         x /= (std + eps).T

    #     # Add dimension
    #     x = x[None]

    #     # Classify
    #     y_hat_soft = classifier(x, lengths)
    #     y_hat_soft = y_hat_soft[...,0]  # Reduce last dimension for librosa.display
    #     y_hat_soft = y_hat_soft.cpu()
    #     y_hat_soft = torch.sigmoid(y_hat_soft.detach())
    #     y_hat_hard = (y_hat_soft > 0.5).int()

    #     # save y_hat_soft / y_hat_hard as .pt
    #     output_path = classif_data_dir + proc_noisy_file_path
    #     output_path = os.path.splitext(output_path)[0]

    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)

    #     torch.save(y_hat_hard, output_path + '_y_hat_hard.pt')
    #     torch.save(y_hat_soft, output_path + '_y_hat_soft.pt')

if __name__ == '__main__':
    main()