import os
import math
import numpy as np
import torch
import librosa 

labels_path = "data/complete/labels/"
target_path ="data/complete/AV_lips/"
audio_path = "data/complete/Clean/volunteers/"

global_frame_rate = 29.970030  # frames per second
fps = 30
wlen_sec = 0.064  # window length in seconds
hop_percent = math.floor((1 / (wlen_sec * global_frame_rate)) * 1e4) / 1e4  # hop size as a percentage of the window length
win = 'hann'  # type of window function (to perform filtering in the time domain)
center = False  # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect'  # This argument is ignored if center = False
pad_at_end = True  # pad audio file at end to match same size after stft + istft

# Ideal binary mask
quantile_fraction = 0.93
quantile_fraction_end = 0.99
quantile_weight = 0.999

# Other parameters:
sampling_rate = 16000
nr_of_audiosamples_per_videosample = int(wlen_sec * hop_percent * sampling_rate)
nr_input_frames = 2


def clean_speech_VAD(observations, quantile_fraction=0.93, quantile_weight=0.999):
    power = abs(observations * observations.conj())
    power = power.sum(axis=0) # sum energy of all frequencies
    sorted_power = np.sort(power, axis=None)[::-1]
    lorenz_function = np.cumsum(sorted_power) / np.sum(sorted_power)
    threshold = np.min(sorted_power[lorenz_function < quantile_fraction])
    vad = power > threshold
    vad = 0.5 + quantile_weight * (vad - 0.5)
    vad = np.round(vad) # to have either 0 or 1 values
    if vad.dtype != 'float32':
        vad = np.float32(vad) # convert to float32
    return vad

def stft(x, fs=16e3, wlen_sec=50e-3, win='hann', hop_percent=0.25, center=True,
         pad_mode='reflect', pad_at_end=True, dtype='complex64'):
    if wlen_sec * fs != int(wlen_sec * fs):
        raise ValueError("wlen_sample of STFT is not an integer.")
    nfft = int(wlen_sec * fs) # STFT window length in samples
    hopsamp = int(hop_percent * nfft) # hop size in samples
    if pad_at_end:
        utt_len = len(x) / fs
        if math.ceil(utt_len / wlen_sec / hop_percent) != int(utt_len / wlen_sec / hop_percent):
            x = np.pad(x, (0,hopsamp), mode='constant')
    Sxx = librosa.core.stft(y=x,n_fft=nfft, hop_length=hopsamp, win_length=None, window=win,
                            center=center, pad_mode=pad_mode, dtype=dtype)
    return Sxx

subsets = sorted(os.listdir(target_path))
for subset in subsets:
	base_video_path = os.path.join(target_path, subset)
	speaker_folder_list = sorted(os.listdir(base_video_path))
	for _, speaker_folder in enumerate(speaker_folder_list):
		video_folder_path = os.path.join(base_video_path, speaker_folder)
		raw_video_files = sorted([x for x in os.listdir(video_folder_path) if '.mp4' in x])
		for _, video_file in enumerate(raw_video_files):
			clean_audio, Fs = librosa.load("{}{}/straightcam/{}.wav".format(audio_path, speaker_folder,
			video_file[:-4]), sr=sampling_rate)
			clean_audio_tf = stft(clean_audio, fs=sampling_rate, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent, 
				center=center, pad_mode=pad_mode, pad_at_end=pad_at_end, dtype='complex64')
			vad_labels = clean_speech_VAD(clean_audio_tf, quantile_fraction=quantile_fraction, quantile_weight=quantile_weight)
			vad_labels_end = clean_speech_VAD(clean_audio_tf, quantile_fraction=quantile_fraction_end, quantile_weight=quantile_weight)
			indices = np.nonzero(vad_labels)
			indices_end = np.nonzero(vad_labels_end)
			vad_labels[indices[0][0]:indices_end[0][-1]] = (indices_end[0][-1]-indices[0][0])*[1]
			np.save('{}{}/{}/{}.npy'.format(labels_path, subset, speaker_folder, video_file[:-4]), vad_labels)
