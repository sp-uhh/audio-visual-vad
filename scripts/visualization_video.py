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

from packages.processing.stft import stft
from packages.processing.video import preprocess_ntcd_matlab, preprocess_ntcd_matlab_nowriter
from packages.processing.target import noise_robust_clean_speech_VAD # because clean audio is very noisy

# Dataset
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit import video_list, speech_list

# Settings
dataset_type = 'test'

dataset_size = 'subset'
#dataset_size = 'complete'

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
win = 'hann'  # type of window function (to perform filtering in the time domain)
center = False  # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect'  # This argument is ignored if center = False
pad_at_end = True  # pad audio file at end to match same size after stft + istft
dtype = 'complex64' # STFT data type

## Noise robust VAD
quantile_fraction_begin = 0.93
quantile_fraction_end = 0.999
quantile_weight = 0.999

## Video
# epsilon = 1e-8  # for numerical stability in the operation below:
# fps = 60
# width = 1920
width = 67
height = 67
# height = 1080
# crf = 17

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

def read_audio_video_only(mat_file_path, audio_file_path):

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
    
    return matlab_frames_list_per_user, y_hat_hard


def process_video_multiprocessing(group_number):

    # Split video frames per num_processes
    frame_count = matlab_frames_list_per_user.shape[0]
    # frame_jump_unit =  frame_count// num_processes
    frame_jump_unit = math.ceil(frame_count/num_processes)

    if frame_count > group_number * frame_jump_unit + frame_jump_unit:
        current_frame_jump_unit = frame_jump_unit
    else:
        current_frame_jump_unit = frame_count - group_number * frame_jump_unit

    # # Process video
    # # initialize video writer
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    # video_filename = classif_data_dir + mat_file_path
    # video_filename = os.path.splitext(video_filename)[0] + '_{}.mp4'.format(group_number)
    
    # if not os.path.exists(os.path.dirname(video_filename)):
    #     os.makedirs(os.path.dirname(video_filename))

    # out = cv2.VideoWriter(video_filename, fourcc, visual_frame_rate, (width, height))

    proc_frames = 0

    video_chunk = np.zeros((current_frame_jump_unit, width, height, 3), dtype='uint8')

    while proc_frames < current_frame_jump_unit:
        frame = proc_frames + frame_jump_unit * group_number

        # preprocess_ntcd_matlab(video_writer=out,
        #             matlab_frames=matlab_frames_list_per_user,
        #             frame=frame,
        #             width=width,
        #             height=height,
        #             y_hat_hard=y_hat_hard)

        video_chunk[proc_frames] = preprocess_ntcd_matlab_nowriter(
                    matlab_frames=matlab_frames_list_per_user,
                    frame=frame,
                    width=width,
                    height=height,
                    y_hat_hard=y_hat_hard)

        proc_frames += 1

    # # Release resources
    # out.release()
    return video_chunk

def combine_output_files(num_processes):

    video_filename = classif_data_dir + mat_file_path
    video_filename = os.path.splitext(video_filename)[0]

    # Create a list of output files and store the file names in a txt file
    list_of_output_files = [video_filename + '_{}.mp4'.format(i) for i in range(num_processes)]
    with open("list_of_output_files.txt", "w") as f:
        for t in list_of_output_files:
            f.write("file {} \n".format(t))

    # use ffmpeg to combine the video output files
    ffmpeg_cmd = "ffmpeg -y -loglevel error -f concat -safe 0 -i list_of_output_files.txt -vcodec copy " + video_filename + '.mp4'
    # ffmpeg_cmd = "ffmpeg -y -i list_of_output_files.txt -vcodec copy " + video_filename + '.mp4'
    sp.Popen(ffmpeg_cmd, shell=True).wait()

    # Remove the temperory output files
    for f in list_of_output_files:
        os.remove(f)
    os.remove("list_of_output_files.txt")

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

    for i, args in enumerate(zip(mat_file_paths, audio_file_paths)):
        process_video(args)
    
    # args = [i for i in range(num_processes)]

    # global mat_file_path
    # global audio_file_path
    # global matlab_frames_list_per_user
    # global y_hat_hard

    # for (a, b) in zip(mat_file_paths, audio_file_paths):

    #     mat_file_path = a
    #     audio_file_path = b

    #     matlab_frames_list_per_user, y_hat_hard = read_audio_video_only(mat_file_path=mat_file_path,
    #                                                                     audio_file_path=audio_file_path)

    #     # process_video_multiprocessing(2)

    #     # Parallel the execution of a function across multiple input values
    #     with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
    #         video = executor.map(process_video_multiprocessing, args)
    #         # executor.map(process_video_multiprocessing, args)

    #     # Concat video chunks
    #     video = np.concatenate(list(video), axis=0)

    #     # Process video
    #     # initialize video writer
    #     fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        
    #     video_filename = classif_data_dir + mat_file_path
    #     video_filename = os.path.splitext(video_filename)[0] + '.mp4'
        
    #     if not os.path.exists(os.path.dirname(video_filename)):
    #         os.makedirs(os.path.dirname(video_filename))

    #     out = cv2.VideoWriter(video_filename, fourcc, visual_frame_rate, (width, height))
    #     for frame in video:
    #         out.write(frame)

    #     out.release()

    #     # Add the audio using ffmpeg-python
    #     video = ffmpeg.input(video_filename)
    #     audio = ffmpeg.input(input_video_dir + audio_file_path)
    #     out = ffmpeg.output(video, audio, os.path.splitext(video_filename)[0] + '_audio.mp4', vcodec='copy', acodec='aac', strict='experimental')
    #     out = out.overwrite_output()
    #     out.run()

        # combine_output_files(num_processes)
        # video_writer.write(rgb_rotated_df)

    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

if __name__ == '__main__':
    main()