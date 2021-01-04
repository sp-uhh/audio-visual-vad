import os
import sys
import torch
import pickle
import numpy as np
import h5py  # to read .mat files
from scipy.fftpack import idct

from torch.utils.data import DataLoader
from video_net import VideoClassifier
from data_handling import VideoFrames
from python.models.utils import binary_cross_entropy, f1_score
from python.utils import count_parameters
import skvideo.io
from subprocess import call


# System 
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

lstm_layers = 2
lstm_hidden_size = 1024 
eps = 1e-8
batch_size = 1


model_name = 'models/Video_Classifier_end_epoch_100/Video_Net_epoch_067_vloss_0.47.pt'

labels_path = "data/complete/labels/test/"
labels_est_path = "data/complete/labels_estimated/test/"
estimation_path = "data/complete/Estimated/test/"
base_path = "data/complete/matlab_raw/"
target_path ="data/complete/AV_lips/"
audio_path = "data/complete/Clean/volunteers/"

accuracy = []
f1score = []

model = VideoClassifier(lstm_layers, lstm_hidden_size, batch_size)
model.load_state_dict(torch.load(model_name))
if cuda: model = model.cuda()
model.eval()

print("Load data")

test_data_path = "data/complete/matlab_raw/test/"
speaker_folders_test = sorted(os.listdir(test_data_path))
for speaker_folder in speaker_folders_test:
    video_folder_path = os.path.join(test_data_path, speaker_folder)
    video_files = sorted([x for x in os.listdir(video_folder_path) if '.mat' in x])
    for video_file in video_files:
        print("{}/{}".format(speaker_folder, video_file))
        mat_file_path = "data/complete/matlab_raw/test/{}/{}".format(speaker_folder, video_file)
        with h5py.File(mat_file_path, 'r') as f:
            for key, value in f.items():
                matlab_frames_list_per_user = np.array(value)

        video_frames = torch.FloatTensor(matlab_frames_list_per_user.shape[0], 3, 67, 67)

        for frame in range(matlab_frames_list_per_user.shape[0]):
            data_frame = matlab_frames_list_per_user[frame]  # data frame will be shortened to "df" below
            reshaped_df = data_frame.reshape(67, 67)
            idct_df = idct(idct(reshaped_df).T).T
            epsilon = 1e-8  # for numerical stability in the operation below:
            normalized_df = idct_df / (matlab_frames_list_per_user.flatten().max() + epsilon) * 255.0
            rotated_df = np.rot90(normalized_df, 3)
            rgb_rotated_df = np.stack((rotated_df,) * 3, axis=0)
            video_frames[frame] = torch.from_numpy(rgb_rotated_df)

        labels = np.load("{}/{}/{}.npy".format(labels_path, speaker_folder, video_file[:-4]))
        labels = [int(elem) for elem in labels]

        y_hat_soft = model.test(video_frames[None,:,:,:,:].to(device))

        y_hat_hard = []
        for i in range(len(y_hat_soft)):
            _, _y_hat_hard = torch.max(y_hat_soft[i], 0)
            y_hat_hard.append(_y_hat_hard.item())

        labels = labels[:len(y_hat_hard)]

        np.save('{}{}/{}.npy'.format(labels_est_path, speaker_folder, video_file[:-4]), y_hat_hard)

        _accuracy = np.sum(np.array(y_hat_hard) == np.array(labels))/len(y_hat_hard)
        _f1_score, _, _, _, _ = f1_score(y_hat_hard, labels)

        accuracy.append(_accuracy)
        f1score.append(_f1_score)

        writer = skvideo.io.FFmpegWriter("{}{}/{}/{}.mp4".format(base_path, "test", speaker_folder, video_file[:-4]), 
	         inputdict={"-r": '90 '}, outputdict={"-r": '30'})

        for frame in range(matlab_frames_list_per_user.shape[0]):
            data_frame = matlab_frames_list_per_user[frame]  # data frame will be shortened to "df" below
            reshaped_df = data_frame.reshape(67, 67)
            idct_df = idct(idct(reshaped_df).T).T
            epsilon = 1e-8  # for numerical stability in the operation below:
            normalized_df = idct_df / (matlab_frames_list_per_user.flatten().max() + epsilon) * 255.0
            rotated_df = np.rot90(normalized_df, 3)
            if y_hat_hard[frame] == 1:
            	rotated_df[-9:,-9:] = 9*[0.0]
            rgb_rotated_df = np.stack((rotated_df,) * 3, axis=0)
            writer.writeFrame(rgb_rotated_df)
        writer.close()
        call(["ffmpeg", "-y", "-i", "{}{}/{}/{}.mp4".format(base_path, "test", speaker_folder, video_file[:-4]), 
        	"-i", "{}{}/straightcam/{}.wav".format(audio_path, speaker_folder, video_file[:-4]), "-c:v", 
        	"copy", "-c:a", "aac", "{}{}/{}-{:.2f}.mp4".format(estimation_path,  speaker_folder, video_file[:-4], _accuracy)])


total_accuracy = np.mean(accuracy)
total_f1_score = np.mean(f1score)

print("Accuracy: {:.2f}".format(total_accuracy))
print("F1-score: {:.2f}".format(total_f1_score))

