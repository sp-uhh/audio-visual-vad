import sys
sys.path.append('.')

import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import h5py as h5
import math

from torch.utils.data import DataLoader

from packages.data_handling import AudioVisualSequenceLabeledFrames, AudioVisualSequenceWavLabeledFrames
from packages.models.AV_Net import DeepVAD_AV
from packages.models.Video_Net import DeepVAD_video
from packages.models.utils import binary_cross_entropy, f1_loss
from packages.utils import count_parameters, collate_many2many_AV, collate_many2many_AV_waveform
from packages.processing.stft import stft_pytorch

# Dataset
# dataset_size = 'subset'
dataset_size = 'complete'

dataset_name = 'ntcd_timit'

# Labels
labels = 'vad_labels'
# labels = 'ibm_labels'
upsampled = True

# ## Video
# visual_frame_rate_i = 30 # initial visual frames per second

## STFT
fs = int(16e3) # Sampling rate
wlen_sec = 64e-3 # window length in seconds
# hop_percent = math.floor((1 / (wlen_sec * visual_frame_rate_i)) * 1e4) / 1e4  # hop size as a percentage of the window length
hop_percent = 0.25 # hop size as a percentage of the window length
win = 'hann' # type of window
center = False # see https://librosa.org/doc/0.7.2/_modules/librosa/core/spectrum.html#stft
pad_mode = 'reflect' # This argument is ignored if center = False
pad_at_end = True # pad audio file at end to match same size after stft + istft

# System
cuda = torch.cuda.is_available()
cuda_device = "cuda:0"
device = torch.device(cuda_device if cuda else "cpu")
num_workers = 16
pin_memory = True
non_blocking = True
rdcc_nbytes = 1024**2*400  # The number of bytes to use for the chunk cache
                           # Default is 1 Mb
                           # Here we are using 400Mb of chunk_cache_mem here
rdcc_nslots = 1e5 # The number of slots in the cache's hash table
                  # Default is 521
                  # ideally 100 x number of chunks that can be fit in rdcc_nbytes
                  # (see https://docs.h5py.org/en/stable/high/file.html?highlight=rdcc#chunk-cache)

# Deep Generative Model
x_dim = 513
if labels == 'vad_labels':
    y_dim = 1
if labels == 'ibm_labels':
    y_dim = 513
# h_dim = [128, 128]
lstm_layers = 2
lstm_hidden_size = 1024
use_mcb=True
batch_norm=False
std_norm =True
eps = 1e-8

# Training
# batch_size = 32
batch_size = 16
# batch_size = 2
learning_rate = 1e-4
# weight_decay = 1e-4
# momentum = 0.9
log_interval = 1
start_epoch = 1
end_epoch = 100

if labels == 'vad_labels':
    # model_name = 'AV_Classifier_vad_upsampled_align_shuffle_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_mcb_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_video_batchnorm_mcb_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_video_dropout_mcb_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_mcb_1024_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_mcb_1024_ssr_relu_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_mcb_512_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_mcb_512_ssr_relu_nopretrain_normdataset_batch32_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_mcb_512_initial_eps_nopretrain_normdataset_batch32_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_mcb_256_nopretrain_normdataset_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_noeps_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_mcb_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_mcb_batchnorm_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_batchnorm_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_pretrained_mcb_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_cleanspeech_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_mcb_cleanspeech_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_cleanspeech_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_frozenResNet_cleanspeech_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_vad_frozenResNet_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    model_name = 'AV_Classifier_vad_frozenResNet_mcb_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_{:03d}'.format(end_epoch)

    pretrained_model_name = 'Video_Classifier_vad_noeps_upsampled_resnet_normvideo3_nopretrain_normimage_batch64_noseqlength_end_epoch_100/Video_Net_epoch_007_vloss_4.51'

if labels == 'ibm_labels':
    # model_name = 'AV_Classifier_mcb_ibm_normdataset_batch16_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'AV_Classifier_batchnorm_resnet_ibm_normdataset_batch16_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    model_name = 'AV_Classifier_ibm_novideo_batchnorm_resnet_ibm_normdataset_batch16_noseqlength_end_epoch_{:03d}'.format(end_epoch)
    # model_name = 'dummy_AV_Classifier_ibm_normdataset_batch16_noseqlength_end_epoch_{:03d}'.format(end_epoch)

# Data directories
input_video_dir = os.path.join('data', dataset_size, 'processed/')
# audio_h5_dir = input_video_dir + os.path.join(dataset_name, 'Noisy', dataset_name + '_' + 'power_spec' + '_statistics.h5')
audio_h5_dir = input_video_dir + os.path.join(dataset_name, 'Noisy', dataset_name + '_' + 'log_power_spec_upsampled' + '_statistics.h5')
# audio_h5_dir = input_video_dir + os.path.join(dataset_name, 'Clean', dataset_name + '_' + 'log_power_spec' + '_statistics.h5')
# audio_h5_dir = input_video_dir + os.path.join(dataset_name, 'Clean', dataset_name + '_' + 'log_power_spec_upsampled' + '_statistics.h5')
# video_h5_dir = input_video_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'pixel' + '_statistics.h5')
video_h5_dir = input_video_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'upsampled' + '_statistics.h5')
# video_h5_dir = input_video_dir + os.path.join(dataset_name, 'matlab_raw', dataset_name + '_' + 'normvideo' + '_statistics.h5')
pretrained_classif_dir = os.path.join('models', pretrained_model_name + '.pt')

#####################################################################################################

print('Load data')
train_dataset = AudioVisualSequenceLabeledFrames(input_video_dir=input_video_dir, dataset_type='train',
                                                              dataset_size=dataset_size, labels=labels, upsampled=upsampled,
                                                              fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent,
                                                              center=center, pad_mode=pad_mode, pad_at_end=pad_at_end, eps=eps)
valid_dataset = AudioVisualSequenceLabeledFrames(input_video_dir=input_video_dir, dataset_type='validation',
                                                              dataset_size=dataset_size, labels=labels, upsampled=upsampled,
                                                              fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent,
                                                              center=center, pad_mode=pad_mode, pad_at_end=pad_at_end, eps=eps)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None,
                        batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory,
                        drop_last=False, timeout=0, worker_init_fn=None, collate_fn=collate_many2many_AV)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=None,
                        batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory,
                        drop_last=False, timeout=0, worker_init_fn=None, collate_fn=collate_many2many_AV)

# train_dataset = AudioVisualSequenceWavLabeledFrames(input_video_dir=input_video_dir, dataset_type='train',
#                                                               dataset_size=dataset_size, labels=labels,
#                                                               fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent,
#                                                               center=center, pad_mode=pad_mode, pad_at_end=pad_at_end, eps=eps)
# valid_dataset = AudioVisualSequenceWavLabeledFrames(input_video_dir=input_video_dir, dataset_type='validation',
#                                                               dataset_size=dataset_size, labels=labels,
#                                                               fs=fs, wlen_sec=wlen_sec, win=win, hop_percent=hop_percent,
#                                                               center=center, pad_mode=pad_mode, pad_at_end=pad_at_end, eps=eps)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None,
#                         batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory,
#                         drop_last=False, timeout=0, worker_init_fn=None, collate_fn=collate_many2many_AV_waveform)
# valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=None,
#                         batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory,
#                         drop_last=False, timeout=0, worker_init_fn=None, collate_fn=collate_many2many_AV_waveform)


print('- Number of training samples: {}'.format(len(train_dataset)))
print('- Number of validation samples: {}'.format(len(valid_dataset)))

print('- Number of training batches: {}'.format(len(train_loader)))
print('- Number of validation batches: {}'.format(len(valid_loader)))

def main():
    print('Create model')
    model = DeepVAD_AV(lstm_layers, lstm_hidden_size, y_dim, use_mcb, eps)
    
    
    print('Load pretrained model')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(pretrained_classif_dir)
    
    # 1. Load Resnet and filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'features' in k}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    model.load_state_dict(model_dict)



    if cuda: model = model.to(device)

    model = nn.parallel.DataParallel(model, device_ids=[0,1,2,3])
    # model = nn.parallel.DataParallel(model, device_ids=[4,5,6,7])

    if cuda: model = model.to(device)

    # nfft = int(wlen_sec * fs) # STFT window length in samples
    # window = torch.hann_window(window_length=nfft).to(device)

    # Create model folder
    model_dir = os.path.join('models', model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if std_norm:
        # Audio
        print('Load audio mean and std')
        with h5.File(audio_h5_dir, 'r') as file:
            audio_mean = file['X_train_mean'][:]
            audio_std = file['X_train_std'][:]

        # Save mean and std
        np.save(model_dir + '/' + 'trainset_audio_mean.npy', audio_mean)
        np.save(model_dir + '/' + 'trainset_audio_std.npy', audio_std)

        audio_mean = torch.tensor(audio_mean).to(device)
        audio_std = torch.tensor(audio_std).to(device)

        # Video
        print('Load video mean and std')
        with h5.File(video_h5_dir, 'r') as file:
            video_mean = file['X_train_mean'][:]
            video_std = file['X_train_std'][:]

        # Save mean and std
        np.save(model_dir + '/' + 'trainset_video_mean.npy', video_mean)
        np.save(model_dir + '/' + 'trainset_video_std.npy', video_std)

        video_mean = torch.tensor(video_mean).to(device)
        video_std = torch.tensor(video_std).to(device)

    # Start log file
    file = open(model_dir + '/' +'output_batch.log','w')
    file = open(model_dir + '/' +'output_epoch.log','w')

    # Optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # criterion = nn.CrossEntropyLoss().to(device) # Ignore padding in the loss

    print('Freeze ResNet')
    for name, child in model.module.named_children():
        if name == 'features':
            for param in child.parameters():
                param.requires_grad = False

    t = len(train_loader)
    m = len(valid_loader)
    print('- Number of learnable parameters: {}'.format(count_parameters(model)))

    print('Start training')
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss, total_accuracy, total_precision, total_recall, total_f1_score = (0, 0, 0, 0, 0)
        # for batch_idx, (x, y) in enumerate(train_loader):
        for batch_idx, (lengths, x, v, y) in tqdm(enumerate(train_loader)):
            if cuda:
                x, v, y, lengths = x.to(device, non_blocking=non_blocking),\
                                    v.to(device, non_blocking=non_blocking),\
                                        y.long().to(device, non_blocking=non_blocking),\
                                            lengths.to(device, non_blocking=non_blocking)

            # # TF representation (PyTorch)
            # x = stft_pytorch(x,
            #         fs=fs,
            #         wlen_sec=wlen_sec,
            #         win=window,
            #         hop_percent=hop_percent,
            #         center=center,
            #         pad_mode=pad_mode,
            #         pad_at_end=pad_at_end) # shape = (freq_bins, frames)

            # # Power spectrogram
            # x = x[...,0]**2 + x[...,1]**2

            # # Apply log
            # x = torch.log(x + eps)

            # # Swap x_dim and seq_length axes
            # x = x.transpose(1,-1) # (B,T,*)

            # # Reduce length of x
            # x = x[:,:v.shape[1]]

            # Normalize power spectrogram
            if std_norm:
                x_norm = x - audio_mean.T
                x_norm /= (audio_std + eps).T

                v_norm = v - video_mean.T
                v_norm /= (video_std + eps).T

                y_hat_soft = model(x_norm, v_norm, lengths)
            else:
                y_hat_soft = model(x, v, lengths)

            # y_hat_soft = torch.squeeze(y_hat_soft)
            loss = 0.
            for (length, pred, target) in zip(lengths, y_hat_soft, y):
                # loss += binary_cross_entropy(pred[:length], target[:length])
                loss += binary_cross_entropy(pred[:length], target[:length], eps)
            # loss /= len(lengths)
            # loss = binary_cross_entropy(y_hat_soft, y, eps)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            # _, y_hat_hard = torch.max(y_hat_soft.data, 1)
            y_hat_soft = torch.sigmoid(y_hat_soft.detach())
            y_hat_hard = (y_hat_soft > 0.5).int()
            # y_hat_hard = (y_hat_soft.detach() > 0.5).int()

            # exclude padding from F1-score
            y_hat_hard_batch, y_batch = [], []
            accuracy, precision, recall, f1_score = 0., 0., 0., 0.
            for (length, pred, target) in zip(lengths, y_hat_hard, y):
                acc, prec, rec, f1 = f1_loss(y_hat_hard=torch.flatten(pred[:length]),
                                                                y=torch.flatten(target[:length]),
                                                                epsilon=eps)
                accuracy += acc
                precision += prec
                recall += rec
                f1_score += f1
            accuracy /= len(lengths)
            precision /= len(lengths)
            recall /= len(lengths)
            f1_score /= len(lengths)

            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1_score += f1_score

            # Save to log
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:4d}/{:4d} ({:2d}%)]    Loss: {:.2f}    Accuracy: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1-score.: {:.2f}'\
                    + '').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            loss.item(), accuracy.item(), precision.item(), recall.item(), f1_score.item()),
                    file=open(model_dir + '/' + 'output_batch.log','a'))

        if epoch % 1 == 0:
            model.eval()
            print("Epoch: {}".format(epoch))
            print(("[Train]       Loss: {:.2f}    Accuracy: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1_score: {:.2f}"\
                + '').format(total_loss / t, total_accuracy / t, total_precision / t, total_recall / t, total_f1_score / t))

            # Save to log
            print(("Epoch: {}".format(epoch)), file=open(model_dir + '/' + 'output_epoch.log','a'))
            print(("[Train]       Loss: {:.2f}    Accuracy: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1_score: {:.2f}"\
                + '').format(total_loss / t, total_accuracy / t, total_precision / t, total_recall / t, total_f1_score / t),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            total_loss, total_accuracy, total_precision, total_recall, total_f1_score = (0, 0, 0, 0, 0)

            for batch_idx, (lengths, x, v, y) in tqdm(enumerate(valid_loader)):
                if cuda:
                    x, v, y, lengths = x.to(device, non_blocking=non_blocking),\
                                        v.to(device, non_blocking=non_blocking),\
                                            y.long().to(device, non_blocking=non_blocking),\
                                                lengths.to(device, non_blocking=non_blocking)
                    # x, v, y = x.to(device, non_blocking=non_blocking),\
                                            # y.long().to(device, non_blocking=non_blocking)

                # # TF representation (PyTorch)
                # x = stft_pytorch(x,
                #         fs=fs,
                #         wlen_sec=wlen_sec,
                #         win=window,
                #         hop_percent=hop_percent,
                #         center=center,
                #         pad_mode=pad_mode,
                #         pad_at_end=pad_at_end) # shape = (freq_bins, frames)

                # # Power spectrogram
                # x = x[...,0]**2 + x[...,1]**2

                # # Apply log
                # x = torch.log(x + eps)

                # # Swap x_dim and seq_length axes
                # x = x.transpose(1,-1)

                # # Reduce length of x
                # x = x[:,:v.shape[1]]

                # Normalize power spectrogram
                if std_norm:
                    x_norm = x - audio_mean.T
                    x_norm /= (audio_std + eps).T

                    v_norm = v - video_mean.T
                    v_norm /= (video_std + eps).T

                    y_hat_soft = model(x_norm, v_norm, lengths)
                else:
                    y_hat_soft = model(x, v, lengths)

                # y_hat_soft = torch.squeeze(y_hat_soft)
                loss = 0.
                for (length, pred, target) in zip(lengths, y_hat_soft, y):
                    loss += binary_cross_entropy(pred[:length], target[:length], eps)
                # loss /= len(lengths)

                total_loss += loss.item()
                # _, y_hat_hard = torch.max(y_hat_soft.data, 1)
                y_hat_soft = torch.sigmoid(y_hat_soft.detach())
                y_hat_hard = (y_hat_soft > 0.5).int()
                # y_hat_hard = (y_hat_soft.detach() > 0.5).int()

                # exclude padding from F1-score
                y_hat_hard_batch, y_batch = [], []
                accuracy, precision, recall, f1_score = 0., 0., 0., 0.
                for (length, pred, target) in zip(lengths, y_hat_hard, y):
                    acc, prec, rec, f1 = f1_loss(y_hat_hard=torch.flatten(pred[:length]),
                                                                    y=torch.flatten(target[:length]),
                                                                    epsilon=eps)
                    accuracy += acc
                    precision += prec
                    recall += rec
                    f1_score += f1
                accuracy /= len(lengths)
                precision /= len(lengths)
                recall /= len(lengths)
                f1_score /= len(lengths)

                total_accuracy += accuracy
                total_precision += precision
                total_recall += recall
                total_f1_score += f1_score

            print(("[Validation]  Loss: {:.2f}    Accuracy: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1_score: {:.2f}"\
                + '').format(total_loss / m, total_accuracy / m, total_precision / m, total_recall / m, total_f1_score / m))

            # Save to log
            print(("[Validation]  Loss: {:.2f}    Accuracy: {:.2f}    Precision: {:.2f}    Recall: {:.2f}    F1_score: {:.2f}"\
                + '').format(total_loss / m, total_accuracy / m, total_precision / m, total_recall / m, total_f1_score / m),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            # Save model
            # NB: if using DataParallel, save model as model.module.state_dict()
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), model_dir + '/' + 'Video_Net_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                    epoch, total_loss / m))
            else:
                torch.save(model.state_dict(), model_dir + '/' + 'Video_Net_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                    epoch, total_loss / m))

if __name__ == '__main__':
    main()
