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
import tempfile
import skvideo.io
from scipy.fftpack import idct

from packages.processing.stft import stft
from packages.processing.video import preprocess_ntcd_matlab
from packages.processing.target import clean_speech_VAD # because clean audio is very noisy

# Dataset
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list

# Settings
dataset_type = 'test'
# dataset_type = 'validation'
# dataset_type = 'train'

# dataset_size = 'subset'
dataset_size = 'complete'

# Labels
# labels = 'ibm_labels'
labels = 'vad_labels'

## Video
visual_frame_rate_i = 30 # initial visual frames per second
width = 67
height = 67
crf = 0 #set the constant rate factor to 0, which is lossless

## STFT
fs = int(16e3) # audio sampling rate
wlen_sec = 0.064  # window length in seconds
hop_percent = math.floor((1 / (wlen_sec * visual_frame_rate_i)) * 1e4) / 1e4  # hop size as a percentage of the window length
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

## Multiprocessing
num_processes = os.cpu_count() + 4
# num_processes = 1

## Classifier
# classif_name = 'oracle_classif'
classif_name = 'oracle_classif_normvideo'

# Data directories
input_video_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
classif_dir = os.path.join('models', classif_name + '.pt')
classif_data_dir = 'data/' + dataset_size + '/models/' + classif_name + '/'
output_data_dir = os.path.join('data', dataset_size, 'models', classif_name + '/')

#############################################################################################

def process_video(args):
    # Separate args
    input_clean_file_path, mat_file_path = args[0], args[1]

    # Read clean speech
    speech, fs_speech = sf.read(input_video_dir + input_clean_file_path, samplerate=None)

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
    speech_vad = clean_speech_VAD(speech,
                                    fs=fs,
                                    wlen_sec=wlen_sec,
                                    hop_percent=hop_percent,
                                    center=center,
                                    pad_mode=pad_mode,
                                    pad_at_end=pad_at_end,
                                    vad_threshold=vad_threshold)

    # Reduce dims
    y_hat_hard = speech_vad[0]

    # Read video
    with h5py.File(input_video_dir + mat_file_path, 'r') as f:
        for key, value in f.items():
            matlab_frames_list_per_user = np.array(value)

    # Process video
    # Create temporary file for video upsampling
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:

        # initialize video writer
        out = skvideo.io.FFmpegWriter(tmp.name,
                    inputdict={'-r': str(visual_frame_rate_i),
                            '-s':'{}x{}'.format(width,height)},
                    # outputdict={'-filter:v': 'fps=fps={}'.format(visual_frame_rate_o),
                    outputdict={'-r': str(visual_frame_rate_i),
                                '-c:v': 'libx264',
                                '-crf': str(crf),
                                '-preset': 'veryslow'}
        )
        A = np.zeros_like(matlab_frames_list_per_user)
        A = A.reshape(-1, width, height)
        # Write and upsample video
        for frame in range(matlab_frames_list_per_user.shape[0]):
            # rgb_rotated_df = preprocess_ntcd_matlab(matlab_frames=matlab_frames_list_per_user,
            #                     frame=frame,
            #                     width=width,
            #                     height=height,
            #                     y_hat_hard=y_hat_hard)

            data_frame = matlab_frames_list_per_user[frame]  # data frame will be shortened to "df" below
            reshaped_df = data_frame.reshape(width, height)
            idct_df = idct(idct(reshaped_df).T).T
            A[frame] = idct_df

        for frame in range(A.shape[0]):
            #TODO: take the largest max - min differences among the frames
            #TODO: divide by idct
            data_frame = A[frame]
            normalized_df = (data_frame - A.min()) / (A.max(axis=(-2,-1)) - A.min(axis=(-2,-1))).max() * 255.0
            rotated_df = np.rot90(normalized_df, 3)

            # Add label on the video
            if y_hat_hard is not None and y_hat_hard[frame] == 1:
                rotated_df[-9:,-9:] = 9*[255] # white square

            # Duplicate channel to visualize video
            rgb_rotated_df = cv2.merge([rotated_df] * 3)           
            out.writeFrame(rgb_rotated_df)
                
        # close out the video writer
        out.close()

        # Process video
        video_filename = classif_data_dir + mat_file_path
        video_filename = os.path.splitext(video_filename)[0]

        os.makedirs(os.path.dirname(video_filename), exist_ok=True)

        # Add the audio using ffmpeg-python
        video = ffmpeg.input(tmp.name)
        audio = ffmpeg.input(input_video_dir + input_clean_file_path)
        out = ffmpeg.output(video, audio, os.path.splitext(video_filename)[0] + '_audio.mp4', vcodec='copy', acodec='aac', strict='experimental')
        out = out.overwrite_output()
        out.run()

def main():

    # Create file list
    mat_file_paths = video_list(input_video_dir=input_video_dir,
                             dataset_type=dataset_type)
    
    input_clean_file_paths, \
        output_clean_file_paths = speech_list(input_speech_dir=input_video_dir,
                            dataset_type=dataset_type)

    t1 = time.perf_counter()

    args = [[input_clean_file_path, mat_file_path]
                    for input_clean_file_path, mat_file_path in zip(input_clean_file_paths, mat_file_paths)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(process_video, args)

    # for arg in zip(input_clean_file_paths, mat_file_paths):
    #     process_video(arg)

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()