import sys
sys.path.append('.')

import numpy as np
import soundfile as sf
import os
from tqdm import tqdm
import math
import h5py as h5

from packages.processing.stft import stft
from packages.processing.target import noise_robust_clean_speech_IBM, noise_robust_clean_speech_VAD

# Parameters
dataset_name = 'ntcd_timit'
if dataset_name == 'ntcd_timit':
    from packages.dataset.ntcd_timit_dataset import speech_list, write_dataset

## Dataset
dataset_types = ['train', 'validation']

dataset_size = 'subset'
# dataset_size = 'complete'
output_data_folder = 'export'

labels = 'vad_labels'
# labels = 'ibm_labels'

input_speech_dir = os.path.join('data', dataset_size, 'raw/')
output_wav_dir = os.path.join('data', dataset_size, 'processed/')
output_dataset_file = os.path.join('data', dataset_size, output_data_folder, dataset_name + '_' + labels + '.h5')


## STFT
video_frame_rate = 29.970030  # frames per second
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
#hop_percent = math.floor((1 / (wlen_sec * video_frame_rate)) * 1e4) / 1e4  # hop size as a percentage of the window length
hop_percent = 0.25
win = 'hann' # type of window
center = False # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect' # This argument is ignored if center = False
pad_at_end = True # pad audio file at end to match same size after stft + istft
dtype = 'complex64'

## Noise robust VAD
vad_quantile_fraction_begin = 0.93
vad_quantile_fraction_end = 0.99
vad_quantile_weight = 0.999

## Noise robust IBM
ibm_quantile_fraction = 0.999
ibm_quantile_weight = 0.999

# HDF5 parameters
rdcc_nbytes = 1024**2*400 # The number of bytes to use for the chunk cache
                          # Default is 1 Mb
                          # Here we are using 400Mb of chunk_cache_mem here
rdcc_nslots = 1e5 # The number of slots in the cache's hash table
                  # Default is 521
                  # ideally 100 x number of chunks that can be fit in rdcc_nbytes
                  # (see https://docs.h5py.org/en/stable/high/file.html?highlight=rdcc#chunk-cache)
                  # for compression 'zlf' --> 1e4 - 1e7
                  # for compression 32001 --> 1e4
X_shape = (513, 0)
X_maxshape = (513, None)
X_chunks = (513, 1)

Y_shape = (513, 0)
Y_maxshape = (513, None)
Y_chunks = (513, 1)
compression = 'lzf'

def main():
    
    if not os.path.exists(os.path.dirname(output_dataset_file)):
        os.makedirs(os.path.dirname(output_dataset_file))

    with h5.File(output_dataset_file, 'a', rdcc_nbytes=rdcc_nbytes, rdcc_nslots=rdcc_nslots) as f:    
            
        # STFT attributes
        f.attrs['fs'] = fs
        f.attrs['wlen_sec'] = wlen_sec
        f.attrs['hop_percent'] = hop_percent
        f.attrs['win'] = win
        f.attrs['dtype'] = dtype

        # label attributes
        f.attrs['vad_quantile_fraction_begin'] = vad_quantile_fraction_begin
        f.attrs['vad_quantile_fraction_end'] = vad_quantile_fraction_end
        if labels == 'vad_labels':
            f.attrs['vad_quantile_weight'] = vad_quantile_weight
        if labels == 'ibm_labels':
            f.attrs['ibm_quantile_fraction'] = ibm_quantile_fraction
            f.attrs['ibm_quantile_weight'] = ibm_quantile_weight

        # HDF5 attributes
        f.attrs['X_chunks'] = X_chunks
        f.attrs['Y_chunks'] = Y_chunks
        f.attrs['compression'] = compression

        for dataset_type in dataset_types:

            # Create file list
            file_paths = speech_list(input_speech_dir=input_speech_dir,
                                    dataset_type=dataset_type)

            # Delete datasets if already exists
            if 'X_' + dataset_type in f:
                del f['X_' + dataset_type]
                del f['Y_' + dataset_type]
            
            # Exact shape of dataset is unknown in advance unfortunately
            # Faster writing if you know the shape in advance
            # Size of chunks corresponds to one spectrogram frame
            f.create_dataset('X_' + dataset_type, shape=X_shape, dtype='float32', maxshape=X_maxshape, chunks=X_chunks, compression=compression)
            f.create_dataset('Y_' + dataset_type, shape=Y_shape, dtype='float32', maxshape=Y_maxshape, chunks=Y_chunks, compression=compression)

            # Store dataset in variables for faster I/O
            fx = f['X_' + dataset_type]
            fy = f['Y_' + dataset_type]

            # Compute mean, std of the train set
            if dataset_type == 'train':
                # VAR = E[X**2] - E[X]**2
                channels_sum, channels_squared_sum = 0., 0.

            for i, file_path in tqdm(enumerate(file_paths)):

                speech, fs_speech = sf.read(input_speech_dir + file_path, samplerate=None)

                # Normalize audio
                speech = speech/(np.max(np.abs(speech)))

                if fs != fs_speech:
                    raise ValueError('Unexpected sampling rate')

                if dataset_size == 'subset':
                    # Save .wav files, just to check if it working
                    output_path = output_wav_dir + file_path
                    output_path = os.path.splitext(output_path)[0]

                    if not os.path.exists(os.path.dirname(output_path)):
                        os.makedirs(os.path.dirname(output_path))
                    
                    sf.write(output_path + '_s.wav', speech, fs)
                    # sf.write(output_path + '_x.wav', mixture, fs)

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
                spectrogram = np.power(abs(speech_tf), 2)

                # Store spectrogram in dataset
                fx.resize((fx.shape[1] + spectrogram.shape[1]), axis = 1)
                fx[:,-spectrogram.shape[1]:] = spectrogram


                if labels == 'vad_labels':
                    # vad
                    speech_vad = noise_robust_clean_speech_VAD(speech_tf,
                                                        quantile_fraction_begin=vad_quantile_fraction_begin,
                                                        quantile_fraction_end=vad_quantile_fraction_end,
                                                        quantile_weight=vad_quantile_weight)                    
                    label = speech_vad

                if labels == 'ibm_labels':
                    # binary mask
                    speech_ibm = noise_robust_clean_speech_IBM(speech_tf,
                                                        vad_quantile_fraction_begin=vad_quantile_fraction_begin,
                                                        vad_quantile_fraction_end=vad_quantile_fraction_end,
                                                        ibm_quantile_fraction=ibm_quantile_fraction,
                                                        quantile_weight=ibm_quantile_weight)
                    label = speech_ibm

                # Store spectrogram in label
                fy.resize((fy.shape[1] + label.shape[1]), axis = 1)
                fy[:,-label.shape[1]:] = label

                # Compute mean, std
                if dataset_type == 'train':
                    # VAR = E[X**2] - E[X]**2
                    channels_sum += np.sum(spectrogram, axis=-1)
                    channels_squared_sum += np.sum(spectrogram**2, axis=-1)
            
            # Compute and save mean, std
            if dataset_type == 'train':
                print('Compute mean and std')
                #NB: compute the empirical std (!= regular std)
                n_samples = fx.shape[1]
                mean = channels_sum / n_samples
                std = np.sqrt((1/(n_samples - 1))*(channels_squared_sum - n_samples * mean**2))
                
                # Delete datasets if already exists
                if 'X_' + dataset_type + '_mean' in f:
                    del f['X_' + dataset_type + '_mean']
                    del f['X_' + dataset_type + '_std']

                f.create_dataset('X_' + dataset_type + '_mean', shape=X_chunks, dtype='float32', maxshape=X_chunks, chunks=None, compression=compression)
                f.create_dataset('X_' + dataset_type + '_std', shape=X_chunks, dtype='float32', maxshape=X_chunks, chunks=None, compression=compression)
                
                f['X_' + dataset_type + '_mean'][:] = mean[..., None] # Add axis to fit chunks shape
                f['X_' + dataset_type + '_std'][:] = std[..., None] # Add axis to fit chunks shape
                print('Mean and std saved in HDF5.')

if __name__ == '__main__':
    main()