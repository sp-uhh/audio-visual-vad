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

from python.processing.stft import stft, istft
from python.processing.target import clean_speech_IBM, clean_speech_VAD

from python.metrics import energy_ratios, compute_stats
from pystoi import stoi
from pesq import pesq
#from uhh_sp.evaluation import polqa
from sklearn.metrics import f1_score

from python.visualization import display_multiple_signals

# Dataset
dataset_name = 'CSR-1-WSJ-0'
if dataset_name == 'CSR-1-WSJ-0':
    from python.dataset.csr1_wjs0_dataset import speech_list, read_dataset

# Settings
dataset_type = 'test'

# dataset_size = 'subset'
dataset_size = 'complete'

# Labels
# labels = 'labels'
labels = 'vad_labels'

# Parameters
## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
hop_percent = 0.25  # hop size as a percentage of the window length
win = 'hann' # type of window
dtype = 'complex64'
eps = 1e-8

## Ideal binary mask
quantile_fraction = 0.999
quantile_weight = 0.999

## Plot spectrograms
vmin = -40 # in dB
vmax = 20 # in dB
xticks_sec = 2.0 # in seconds
fontsize = 30

## Stats
confidence = 0.95 # confidence interval

if labels == 'labels':
    # M2
    # model_name = 'M2_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_085_vloss_417.69'
    model_name = 'M2_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_098_vloss_414.57'

    # classifier
    # classif_name = 'classif_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_57.53'
    # classif_name = 'classif_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_073_vloss_56.43'
    classif_name = 'oracle_classif'
    # classif_name = 'timo_classif'

if labels == 'vad_labels':
    # M2
    # model_name = 'M2_VAD_hdim_128_128_zdim_032_end_epoch_100/M2_epoch_085_vloss_465.98'
    # model_name = 'M2_VAD_quantile_0.999_hdim_128_128_zdim_032_end_epoch_200/M2_epoch_085_vloss_487.80'
    # model_name = 'M2_VAD_quantile_0.999_hdim_128_128_zdim_032_end_epoch_200/M2_epoch_087_vloss_482.93'
    model_name = 'M2_VAD_quantile_0.999_hdim_128_128_zdim_032_end_epoch_200/M2_epoch_087_vloss_482.93'

    # classifier
    # classif_name = 'classif_VAD_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_096_vloss_0.21'
    classif_name = 'classif_VAD_qf0.999_normdataset_hdim_128_128_end_epoch_100/Classifier_epoch_090_vloss_0.23'
    # classif_name = 'oracle_classif'
    # classif_name = 'ones_classif'
    # classif_name = 'zeros_classif'
    # classif_name = 'timo_vad_classif'


# Data directories
input_speech_dir = os.path.join('data',dataset_size,'raw/')
processed_data_dir = os.path.join('data',dataset_size,'processed/')
model_data_dir = os.path.join('data', dataset_size, 'models_wsj0', model_name, classif_name + '/') # Directory where estimated data is stored


def compute_metrics_utt(args):
    # Separate args
    file_path, snr_db = args[0], args[1]

    # Open HDF5 file
    with h5.File(h5_file_path, 'r') as file:
        x = np.array(file["X"][:])
        y = np.array(file["Y"][:])
        length = data.shape[-1]
    
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)

    #TODO: make video with audio with target
    # Create temporary file for video without audio
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        out = skvideo.io.FFmpegWriter(output_path + '_target.mp4',
                inputdict={'-r': str(visual_frame_rate_o),
                        '-s':'{}x{}'.format(width,height)},
                outputdict={'-filter:v': 'fps=fps={}'.format(visual_frame_rate_o),
                            '-c:v': 'libx264',
                            '-crf': str(crf),
                            '-preset': 'veryslow'}
        )

        # Write video
        #TODO: control the shape
        for frame in range(data.T):
            # Add label on the video
            if y[...,frame] == 1:
                frame.T[-9:,-9:] = 9*[255] # white square
            out.writeFrame(frame.T)
            
        # close out the video writer
        out.close()

        # Add the audio using ffmpeg-python
        video = ffmpeg.input(output_path)
        audio = ffmpeg.input(input_speech_dir + audio_file_path)
        out = ffmpeg.output(video, audio, os.path.splitext(output_path)[0] + '_audio.mp4', vcodec='copy', acodec='aac', strict='experimental')
        out = out.overwrite_output()
        out.run()

    #TODO: make video with audio with pred
    # Create temporary file for video without audio
    with tempfile.NamedTemporaryFile(suffix='.mp4') as tmp:
        out = skvideo.io.FFmpegWriter(output_path + '_pred.mp4',
                inputdict={'-r': str(visual_frame_rate_o),
                        '-s':'{}x{}'.format(width,height)},
                outputdict={'-filter:v': 'fps=fps={}'.format(visual_frame_rate_o),
                            '-c:v': 'libx264',
                            '-crf': str(crf),
                            '-preset': 'veryslow'}
        )
    
        # Write and upsample video
        for frame in range(data.T):
            # Add label on the video
            if y_hat_hard[...,frame] == 1:
                frame.T[-9:,-9:] = 9*[255] # white square
            out.writeFrame(frame.T)
        
        # close out the video writer
        out.close()

        # Add the audio using ffmpeg-python
        video = ffmpeg.input(output_path)
        audio = ffmpeg.input(input_speech_dir + audio_file_path)
        out = ffmpeg.output(video, audio, os.path.splitext(output_path)[0] + '_audio.mp4', vcodec='copy', acodec='aac', strict='experimental')
        out = out.overwrite_output()
        out.run()

    # Read files
    s_t, fs_s = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_s.wav') # clean speech
    n_t, fs_n = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_n.wav') # noise
    x_t, fs_x = sf.read(processed_data_dir + os.path.splitext(file_path)[0] + '_x.wav') # mixture
    s_hat_t, fs_s_hat = sf.read(model_data_dir + os.path.splitext(file_path)[0] + '_s_est.wav') # est. speech

    ## F1 score
    # ideal binary mask
    y_hat_hard = torch.load(model_data_dir + os.path.splitext(file_path)[0] + '_ibm_hard_est.pt', map_location=lambda storage, location: storage) # shape = (frames, freq_bins)
    y_hat_hard = (y_hat_hard > 0.5).T # Transpose to match target y, shape = (freq_bins, frames)

    # F1-score
    f1_score, tp, tn, fp, fn = f1_loss(y_hat_hard=torch.flatten(y_hat_hard), y=torch.flatten(y), epsilon=eps)

    # TF representation
    s_tf = stft(s_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype) # shape = (freq_bins, frames)

    if labels == 'labels':
        y = clean_speech_IBM(s_tf,
                                quantile_fraction=quantile_fraction,
                                quantile_weight=quantile_weight)
    if labels == 'vad_labels':
        y = clean_speech_VAD(s_tf,
                        quantile_fraction=quantile_fraction,
                        quantile_weight=quantile_weight)

    f1score_s_hat = f1_score(y.flatten(), y_hat_hard.flatten(), average="binary")

    # plots of target / estimation
    # TF representation
    x_tf = stft(x_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype) # shape = (freq_bins, frames)

    s_hat_tf = stft(s_hat_t,
                fs=fs,
                wlen_sec=wlen_sec,
                win=win,
                hop_percent=hop_percent,
                dtype=dtype) # shape = (freq_bins, frames)                 

    #TODO: make TF representation
    ## mixture signal (wav + spectro)
    ## target signal (wav + spectro + mask)
    ## estimated signal (wav + spectro + mask)
    y_hat_hard = y_hat_hard.cpu().numpy()
    y_hat_hard = y_hat_hard.T  # Transpose to match librosa.display
    x_tf = x_tf.T # Transpose to match librosa.display

    signal_list = [
        [x_t, x_tf, None], # mixture: (waveform, tf_signal, no mask)
        [s_t, s_tf, y], # clean speech
        [None, None, y_hat_hard]
        #[None, None, y_hat_soft]
    ]

    fig = display_multiple_signals(signal_list,
                        fs=fs, vmin=vmin, vmax=vmax,
                        wlen_sec=wlen_sec, hop_percent=hop_percent,
                        xticks_sec=xticks_sec, fontsize=fontsize)
    
    # put all metrics in the title of the figure
    title = "Input SNR = {:.1f} dB \n" \
        "STOI = {:.2f}, " \
        "PESQ = {:.2f} \n" \
        "F1-score = {:.3f} \n".format(snr_db, si_sdr, si_sir, si_sar, stoi_s_hat, pesq_s_hat, f1score_s_hat)

    fig.suptitle(title, fontsize=40)

    # Save figure
    fig.savefig(model_data_dir + os.path.splitext(file_path)[0] + '_fig.png')

    # Clear figure
    plt.close()

    metrics = [si_sdr, si_sir, si_sar, stoi_s_hat, pesq_s_hat, f1score_s_hat]
    
    return metrics

def main():
    # Load input SNR
    all_snr_db = read_dataset(processed_data_dir, dataset_type, 'snr_db')
    all_snr_db = np.array(all_snr_db)

    # Create file list
    file_paths = speech_list(input_speech_dir=input_speech_dir,
                                dataset_type=dataset_type)

    # Fuse both list
    args = [[file_path, snr_db] for file_path, snr_db in zip(file_paths, all_snr_db)]

    t1 = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
        all_metrics = executor.map(compute_metrics_utt, args)
    
    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

    # Transform generator to list
    all_metrics = list(all_metrics)
    metrics_keys = ['SI-SDR', 'SI-SIR', 'SI-SAR', 'STOI', 'PESQ', 'F1-SCORE']

    # Compute & save stats
    compute_stats(metrics_keys=metrics_keys,
                  all_metrics=all_metrics,
                  all_snr_db=all_snr_db,
                  model_data_dir=model_data_dir,
                  confidence=confidence)

if __name__ == '__main__':
    main()