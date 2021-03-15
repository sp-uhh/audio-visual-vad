import sys
sys.path.append('.')

import os
import numpy as np
import torch, torchaudio
import h5py  # to read .mat files
import math
from tqdm import tqdm
import cv2
import ffmpeg
import time
import concurrent.futures # for multiprocessing
import soundfile as sf

from packages.processing.stft import stft, stft_pytorch
from packages.processing.target import clean_speech_VAD, clean_speech_IBM, \
                                    noise_robust_clean_speech_IBM # because clean audio is very noisy
from packages.visualization import display_wav_spectro_mask

import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_theme()

# Dataset
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list

# Settings
# dataset_type = 'test'
# dataset_type = 'validation'
dataset_type = 'train'

dataset_size = 'subset'
# dataset_size = 'complete'

# Labels
# labels = 'ibm_labels'
labels = 'vad_labels'

# Parameters
## STFT
# visual_frame_rate = 29.970030  # initial visual frames per second
visual_frame_rate = 30 # initial visual frames per second
fs = int(16e3) # audio sampling rate
wlen_sec = 0.064  # window length in seconds
hop_percent = math.floor((1 / (wlen_sec * visual_frame_rate)) * 1e4) / 1e4  # hop size as a percentage of the window length
# hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann'  # type of window function (to perform filtering in the time domain)
center = False  # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect'  # This argument is ignored if center = False
pad_at_end = True  # pad audio file at end to match same size after stft + istft
dtype = 'complex64' # STFT data type

## Noise robust VAD
vad_threshold = 1.70

## Noise robust IBM
eps = 1e-8
ibm_threshold = 50 # Hard threshold
# ibm_threshold = 65 # Soft threshold

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Classifier
# classif_name = 'oracle_classif'
classif_name = 'oracle_classif_ibm'
# classif_name = 'oracle_classif_ibm_noise_robust'

# Data directories
input_video_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
classif_dir = os.path.join('models', classif_name + '.pt')
classif_data_dir = 'data/' + dataset_size + '/models/' + classif_name + '/'
output_data_dir = os.path.join('data', dataset_size, 'models', classif_name + '/')

#############################################################################################

def process_audio(args):
    # Separate args
    mat_file_path, audio_file_path  = args[0], args[1]

    # Read clean speech
    s_t, fs_s = sf.read(input_video_dir + audio_file_path, samplerate=None)
    s_t_torch, fs_s = torchaudio.load(input_video_dir + audio_file_path)
    s_t_torch = s_t_torch[0] # 1channel

    # Normalize audio
    s_t = s_t/(np.max(np.abs(s_t)))
    s_t_torch = s_t_torch / (torch.max(torch.abs(s_t_torch)))

    # TF reprepsentation (Librosa)
    s_tf = stft(s_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win, 
                hop_percent=hop_percent,
                center=center,
                pad_mode=pad_mode,
                pad_at_end=pad_at_end,
                dtype=dtype) # shape = (freq_bins, frames)

    # TF representation (torchaudio.stft)
    s_tf_torch = stft_pytorch(s_t_torch,
            fs=fs,
            wlen_sec=wlen_sec,
            win=win, 
            hop_percent=hop_percent,
            center=center,
            pad_mode=pad_mode,
            pad_at_end=pad_at_end) # shape = (freq_bins, frames)

    # TF representation (torchaudio.transform.Spectrogram)
    # #TODO: Sometimes stft / istft shortens the ouput due to window size
    # # so you need to pad the end with hopsamp zeros
    # if pad_at_end:
    #     utt_len = len(x) / fs
    #     if math.ceil(utt_len / wlen_sec / hop_percent) != int(utt_len / wlen_sec / hop_percent):
    #         x_ = torch.nn.functional.pad(x, (0,hopsamp), mode='constant')
    #     else:
    #         x_ = x
    #TODO: implement own stft_transform to put center=False
    # s_tf_torch_transform = stft_transform(s_t_torch)

    # compare TF representation
    s_tf_torch = s_tf_torch[...,0].numpy() + 1j * s_tf_torch[...,1].numpy()

    # np.testing.assert_allclose(s_tf_torch, s_tf)

    if labels == 'vad_labels':

        # Compute vad
        speech_vad = clean_speech_VAD(s_t_torch.numpy(),
                                      fs=fs,
                                      wlen_sec=wlen_sec,
                                      hop_percent=hop_percent,
                                      center=center,
                                      pad_mode=pad_mode,
                                      pad_at_end=pad_at_end,
                                      vad_threshold=vad_threshold)
        
        label = speech_vad

    if labels == 'ibm_labels':
        # binary mask
        # speech_ibm = clean_speech_IBM(s_tf_torch,
        #                               eps=eps,
        #                               ibm_threshold=ibm_threshold)

        speech_ibm = noise_robust_clean_speech_IBM(s_t_torch.numpy(),
                                                   s_tf_torch,
                                                   fs=fs,
                                                   wlen_sec=wlen_sec,
                                                   hop_percent=hop_percent,
                                                   center=center,
                                                   pad_mode=pad_mode,
                                                   pad_at_end=pad_at_end,
                                                   vad_threshold=vad_threshold,
                                                   eps=eps,
                                                   ibm_threshold=ibm_threshold)
        label = speech_ibm

    # Plot figure
    fig = display_wav_spectro_mask(x=s_t, x_tf=s_tf_torch, x_ibm=label,
                        fs=fs, vmin=vmin, vmax=vmax,
                        wlen_sec=wlen_sec, hop_percent=hop_percent,
                        xticks_sec=xticks_sec, fontsize=fontsize)
    
    # # put all metrics in the title of the figure
    # title = "Cumulated power = {:,} dB \n" \
    #     "".format(sorted_power.sum())
    # fig.suptitle(title, fontsize=40)

    # Save figure
    output_path = classif_data_dir + mat_file_path
    output_path = os.path.splitext(output_path)[0]
    print(output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # fig.savefig(output_path + '_hard_' + labels + '_torch_transform.png')
    # fig.savefig(output_path + '_hard_' + labels + '_threshold.png')
    fig.savefig(output_path + '_hard_' + labels + '_30fps.png')
    # fig.savefig(output_path + '_hard_' + labels + '_threshold_50.png')
    # fig.savefig(output_path + '_hard_' + labels + '_threshold_11.png')

    #TODO: put in visualization.py
    # # Plot histograms of energy
    # plt.figure(figsize=(20,25))
    # ax = sns.histplot(sorted_power, binwidth=10)
    # ax.figure.savefig(output_path + '_hist.png')

    # Close figure
    plt.close()

def main():

    # global stft_transform
    global vad

    # Create file list
    mat_file_paths = video_list(input_video_dir=input_video_dir,
                             dataset_type=dataset_type)
    
    audio_file_paths, output_audio_file_paths = speech_list(input_speech_dir=input_video_dir,
                             dataset_type=dataset_type)
    
    # # STFT (torchaudio.transforms.Spectrogram)
    # nfft = int(wlen_sec * fs) # STFT window length in samples
    # hopsamp = int(hop_percent * nfft) # hop size in samples

    # stft_transform = torchaudio.transforms.Spectrogram(n_fft=nfft,
    #                                   win_length=None,
    #                                   hop_length=hopsamp,
    #                                   pad=nfft//2,
    #                                   window_fn=torch.hann_window,
    #                                   power=None)
    
    t1 = time.perf_counter()

    args = [[mat_file_path, audio_file_path]
                    for mat_file_path, audio_file_path in zip(mat_file_paths, audio_file_paths)]

    # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
    #     executor.map(process_audio, args)

    for i, args in enumerate(zip(mat_file_paths, audio_file_paths)):
        process_audio(args)
    
    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()