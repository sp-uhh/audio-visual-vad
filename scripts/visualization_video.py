import sys
sys.path.append('.')

import os
import numpy as np
import torch
import h5py  # to read .mat files
import math
from tqdm import tqdm
import cv2
import ffmpeg
import time
import concurrent.futures # for multiprocessing
import soundfile as sf

from packages.processing.stft import stft
from packages.processing.video import preprocess_ntcd_matlab

# Dataset
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list
    from packages.processing.target import noise_robust_clean_speech_VAD # because clean audio is very noisy


# Settings
dataset_type = 'test'

dataset_size = 'subset'
#dataset_size = 'complete'

# Labels
# labels = 'ibm_labels'
labels = 'vad_labels'

# System 


# Parameters
## STFT
# visual_frame_rate = 29.970030  # initial visual frames per second
visual_frame_rate = 30 # initial visual frames per second
fs = int(16e3) # audio sampling rate
wlen_sec = 0.064  # window length in seconds
hop_percent = math.floor((1 / (wlen_sec * visual_frame_rate)) * 1e4) / 1e4  # hop size as a percentage of the window length
win = 'hann'  # type of window function (to perform filtering in the time domain)
center = False  # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect'  # This argument is ignored if center = False
pad_at_end = True  # pad audio file at end to match same size after stft + istft
dtype = 'complex64' # STFT data type

## IBM
quantile_fraction_begin = 0.93
quantile_fraction_end = 0.999
quantile_weight = 0.999

## Video
# epsilon = 1e-8  # for numerical stability in the operation below:

## Save video
# fps = 60
# width = 1920
width = 67
height = 67
# height = 1080
# crf = 17

## Classifier
classif_name = 'oracle_classif'

# Data directories
input_video_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
classif_dir = os.path.join('models', classif_name + '.pt')
classif_data_dir = 'data/' + dataset_size + '/models/' + classif_name + '/'
output_data_dir = os.path.join('data', dataset_size, 'models', classif_name + '/')

#############################################################################################

def process_video(args):
    # Separate args
    mat_file_path, audio_file_path  = args[0], args[1]

    # Read clean speech
    speech, fs_speech = sf.read(input_video_dir + audio_file_path, samplerate=None)

    # Normalize audio
    speech = speech/(np.max(np.abs(speech)))

    # TF reprepsentation
    speech_tf = stft(speech,
                     fs=fs,
                     wlen_sec=wlen_sec,
                     win=win, 
                     hop_percent=hop_percent,
                     center=center,
                     pad_mode=pad_mode,
                     pad_at_end=pad_at_end,
                     dtype=dtype) # shape = (freq_bins, frames)

    # Compute vad
    speech_vad = noise_robust_clean_speech_VAD(speech_tf,
                                        quantile_fraction_begin=quantile_fraction_begin,
                                        quantile_fraction_end=quantile_fraction_end,
                                        quantile_weight=quantile_weight)
    # Reduce dims
    y_hat_hard = speech_vad[0]

    # Read video
    with h5py.File(input_video_dir + mat_file_path, 'r') as f:
        for key, value in f.items():
            matlab_frames_list_per_user = np.array(value)

    # Process video
    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    video_filename = classif_data_dir + mat_file_path
    video_filename = os.path.splitext(video_filename)[0] + '.mp4'
    
    if not os.path.exists(os.path.dirname(video_filename)):
        os.makedirs(os.path.dirname(video_filename))

    out = cv2.VideoWriter(video_filename, fourcc, visual_frame_rate, (width, height))
    
    # TODO: in parallel ?
    for frame in range(matlab_frames_list_per_user.shape[0]):
        preprocess_ntcd_matlab(video_writer=out,
                                matlab_frames=matlab_frames_list_per_user,
                                frame=frame,
                                width=width,
                                height=height,
                                y_hat_hard=y_hat_hard)
            
    # close out the video writer
    out.release()

    # Add the audio using ffmpeg-python
    video = ffmpeg.input(video_filename)
    audio = ffmpeg.input(input_video_dir + audio_file_path)
    out = ffmpeg.output(video, audio, os.path.splitext(video_filename)[0] + '_audio.mp4', vcodec='copy', acodec='aac', strict='experimental')
    out = out.overwrite_output()
    out.run()

def main():

    # Create file list
    mat_file_paths = video_list(input_video_dir=input_video_dir,
                             dataset_type=dataset_type)
    
    audio_file_paths = speech_list(input_speech_dir=input_video_dir,
                             dataset_type=dataset_type)

    t1 = time.perf_counter()

    args = [[mat_file_path, audio_file_path]
                    for mat_file_path, audio_file_path in zip(mat_file_paths, audio_file_paths)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        executor.map(process_video, args)

    # # TODO: in parallel
    # for i, args in enumerate(zip(mat_file_paths, audio_file_paths)):
    #     process_video(args)
    
    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()