import sys
sys.path.append('.')

import os
import numpy as np
import torch
import soundfile as sf
import librosa
import json
import matplotlib.pyplot as plt
import concurrent.futures # for multiprocessing
import time
import tempfile
import ffmpeg
import skvideo.io
import cv2
import h5py as h5

from packages.metrics import energy_ratios, compute_stats
from pystoi import stoi
from pesq import pesq
#from uhh_sp.evaluation import polqa

from packages.processing.stft import stft, istft
from packages.visualization import display_multiple_signals
from packages.models.utils import f1_loss

# Dataset
dataset_size = 'subset'
# dataset_size = 'complete'

dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list

dataset_type = 'test'
# labels = 'labels'
labels = 'vad_labels'
upsampled = True

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

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Stats
confidence = 0.95 # confidence interval

# if labels == 'labels':
    # M2
    # model_name = 'M2_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_085_vloss_417.69'
    # model_name = 'M2_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_098_vloss_414.57'

    # classifier
    # classif_name = 'classif_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_57.53'
    # classif_name = 'classif_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_073_vloss_56.43'
    # classif_name = 'oracle_classif'
    # classif_name = 'timo_classif'

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
classif_data_dir = os.path.join('data', dataset_size, 'models', classif_name + '/')

####################################################

def compute_metrics_utt(args):
    # Separate args
    mat_file_path, audio_file_path = args[0], args[1]

    # select utterance
    h5_file_path = input_video_dir + mat_file_path

    # Open HDF5 file
    with h5.File(h5_file_path, 'r') as file:
        x_video = np.array(file["X"][:])
        y = np.array(file["Y"][:])
        length = x_video.shape[-1]
    
    # Output file
    output_path = classif_data_dir + mat_file_path
    output_path = os.path.splitext(output_path)[0]

    # Load y_hat_hard / y_hat_soft
    y_hat_hard = np.load(output_path + '_y_hat_hard.npy')
    y_hat_soft = np.load(output_path + '_y_hat_soft.npy')

    ## F1 score
    accuracy, precision, recall, f1_score = f1_loss(y_hat_hard=torch.flatten(torch.Tensor(y_hat_hard)), y=torch.flatten(torch.Tensor(y)), epsilon=eps)

    # make video with audio with target
    # Create temporary file for video without audio
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        out = skvideo.io.FFmpegWriter(tmp.name,
                inputdict={'-r': str(visual_frame_rate_o),
                        '-s':'{}x{}'.format(width,height)},
                outputdict={'-filter:v': 'fps=fps={}'.format(visual_frame_rate_o),
                            '-c:v': 'libx264',
                            '-crf': str(crf),
                            '-preset': 'veryslow'}
        )

        # Write video
        for j, x_video_frame in enumerate(x_video.T):
            # Add label on the video
            if y[...,j] == 1:
                x_video_frame.T[-9:,-9:] = 255 # white square
            out.writeFrame(x_video_frame.T)
            
        # close out the video writer
        out.close()

        # Add the audio using ffmpeg-python
        video = ffmpeg.input(tmp.name)
        audio = ffmpeg.input(input_speech_dir + audio_file_path)
        out = ffmpeg.output(video, audio, os.path.splitext(output_path)[0] + '_oracle_audio.mp4', vcodec='copy', acodec='aac', strict='experimental')
        out = out.overwrite_output()
        out.run()

    # make video with audio with pred
    # Create temporary file for video without audio
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        out = skvideo.io.FFmpegWriter(tmp.name,
                inputdict={'-r': str(visual_frame_rate_o),
                        '-s':'{}x{}'.format(width,height)},
                outputdict={'-filter:v': 'fps=fps={}'.format(visual_frame_rate_o),
                            '-c:v': 'libx264',
                            '-crf': str(crf),
                            '-preset': 'veryslow'}
        )

        # Write video
        for j, x_video_frame in enumerate(x_video.T):
            # Add label on the video
            if y_hat_hard[...,j] == 1:
                x_video_frame.T[-9:,-9:] = 255 # white square
            out.writeFrame(x_video_frame.T)
        
        # close out the video writer
        out.close()

        # Add the audio using ffmpeg-python
        video = ffmpeg.input(tmp.name)
        audio = ffmpeg.input(input_speech_dir + audio_file_path)
        out = ffmpeg.output(video, audio, os.path.splitext(output_path)[0] + '_pred_audio.mp4', vcodec='copy', acodec='aac', strict='experimental')
        out = out.overwrite_output()
        out.run()

    #TODO: make video with y_hat_soft

    # Read files
    s_t, fs_s = sf.read(input_speech_dir + audio_file_path) # clean speech

    # TF representation
    s_tf = stft(s_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win, 
                hop_percent=hop_percent,
                center=center,
                pad_mode=pad_mode,
                pad_at_end=pad_at_end,
                dtype=dtype) # shape = (freq_bins, frames)

    # Reduce size if larger than number of video frames
    if s_tf.shape[-1] > x_video.shape[-1]:
        s_tf = s_tf[...,:x_video.shape[-1]]

    # plots of target
    ## mixture signal (wav + spectro)
    ## target signal (wav + spectro + mask)
    ## estimated signal (wav + spectro + mask)

    signal_list = [
        [s_t, s_tf, y], # clean speech
        [None, None, y_hat_hard],
        [None, None, y_hat_soft]
    ]

    fig = display_multiple_signals(signal_list,
                        fs=fs, vmin=vmin, vmax=vmax,
                        wlen_sec=wlen_sec, hop_percent=hop_percent,
                        xticks_sec=xticks_sec, fontsize=fontsize)
    
    # put all metrics in the title of the figure
    #TODO: modify
    title = "Input SNR = {:.1f} dB \n" \
        "F1-score = {:.3f} \n".format(accuracy, precision, recall,f1_score)

    fig.suptitle(title, fontsize=40)

    # Save figure
    fig.savefig(output_path + '_fig.png')

    # Clear figure
    plt.close()

    metrics = [accuracy, precision, recall, f1_score]
    
    return metrics

def main():

    # Create file list
    mat_file_paths = video_list(input_video_dir=input_video_dir,
                            dataset_type=dataset_type,
                            upsampled=upsampled)

    audio_file_paths = speech_list(input_speech_dir=input_speech_dir,
                            dataset_type=dataset_type)

    # Fuse both list
    args = [[mat_file_path, audio_file_path] for mat_file_path, audio_file_path in zip(mat_file_paths, audio_file_paths)]

    t1 = time.perf_counter()

    all_metrics = []
    for arg in args:
        metrics = compute_metrics_utt(arg)
        all_metrics.append(metrics)

    # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
    #     all_metrics = executor.map(compute_metrics_utt, args)
    
    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

    # Transform generator to list
    # all_metrics = list(all_metrics)
    metrics_keys = ['Accuracy', 'Precision', 'Recall', 'F1-score']

    # Compute & save stats
    compute_stats(metrics_keys=metrics_keys,
                  all_metrics=all_metrics,
                  model_data_dir=classif_data_dir,
                  confidence=confidence)

if __name__ == '__main__':
    main()