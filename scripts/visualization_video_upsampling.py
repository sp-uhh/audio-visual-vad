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
import subprocess as sp
import skvideo.io

from packages.processing.stft import stft
from packages.processing.video import preprocess_ntcd_matlab
from packages.processing.target import noise_robust_clean_speech_VAD # because clean audio is very noisy

# Dataset
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list

# Settings
dataset_type = 'test'
# dataset_type = 'validation'
# dataset_type = 'train'

dataset_size = 'subset'
#dataset_size = 'complete'

# Labels
# labels = 'ibm_labels'
labels = 'vad_labels'

# Parameters
## STFT
fs = int(16e3) # audio sampling rate
wlen_sec = 0.064  # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann'  # type of window function (to perform filtering in the time domain)
center = False  # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect'  # This argument is ignored if center = False
pad_at_end = True  # pad audio file at end to match same size after stft + istft
dtype = 'complex64' # STFT data type
visual_frame_rate_i = 30 # initial visual frames per second
visual_frame_rate_o = 1 / (wlen_sec * hop_percent)

## Noise robust VAD
quantile_fraction_begin = 0.93
quantile_fraction_end = 0.999
quantile_weight = 0.999

## Video
width = 67
height = 67
crf = 0 #set the constant rate factor to 0, which is lossless

## Multiprocessing
num_processes = os.cpu_count() + 4
# num_processes = 1

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

    # # Compute vad
    # speech_vad = noise_robust_clean_speech_VAD(speech_tf,
    #                                     quantile_fraction_begin=quantile_fraction_begin,
    #                                     quantile_fraction_end=quantile_fraction_end,
    #                                     quantile_weight=quantile_weight)
    # # Reduce dims
    # y_hat_hard = speech_vad[0]

    # Read video
    with h5py.File(input_video_dir + mat_file_path, 'r') as f:
        for key, value in f.items():
            matlab_frames_list_per_user = np.array(value)

    # Process video
    video_filename = classif_data_dir + mat_file_path
    # video_filename = os.path.splitext(video_filename)[0] + '_skvideo.mp4'
    video_filename = os.path.splitext(video_filename)[0] + '_skvideo_upsampled_2.mp4'
    
    if not os.path.exists(os.path.dirname(video_filename)):
        os.makedirs(os.path.dirname(video_filename))
    
    out = skvideo.io.FFmpegWriter(video_filename,
                inputdict={'-r': str(visual_frame_rate_i),
                           '-s':'{}x{}'.format(width,height)},
                outputdict={'-filter:v': 'fps=fps={}'.format(visual_frame_rate_o),
                            '-c:v': 'libx264',
                            '-crf': str(crf),
                            '-preset': 'veryslow'}
    )

    for frame in range(matlab_frames_list_per_user.shape[0]):
        rgb_rotated_df = preprocess_ntcd_matlab(matlab_frames=matlab_frames_list_per_user,
                               frame=frame,
                               width=width,
                               height=height,
                               y_hat_hard=None)
        out.writeFrame(rgb_rotated_df)
            
    # close out the video writer
    out.close()

    # Add the audio using ffmpeg-python
    video = ffmpeg.input(video_filename)
    audio = ffmpeg.input(input_video_dir + audio_file_path)
    out = ffmpeg.output(video, audio, os.path.splitext(video_filename)[0] + '_audio.mp4', vcodec='copy', acodec='aac', strict='experimental')
    out = out.overwrite_output()
    out.run()

    # Check length of new video w.r.t spectrogram
    cap = cv2.VideoCapture(video_filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frameCount  and ret):
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    assert(speech_tf.shape[-1] == buf.shape[0])

def main():

    # Create file list
    mat_file_paths = video_list(input_video_dir=input_video_dir,
                             dataset_type=dataset_type)
    
    audio_file_paths = speech_list(input_speech_dir=input_video_dir,
                             dataset_type=dataset_type)

    t1 = time.perf_counter()

    # args = [[mat_file_path, audio_file_path]
    #                 for mat_file_path, audio_file_path in zip(mat_file_paths, audio_file_paths)]

    # with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #     executor.map(process_video, args)

    for arg in zip(mat_file_paths, audio_file_paths):
        process_video(arg)

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()