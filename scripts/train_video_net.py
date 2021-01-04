import os
import sys
import torch
import pickle
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader
from video_net import VideoClassifier
from data_handling import VideoFrames
from python.models.utils import binary_cross_entropy, binary_cross_entropy_2classes, f1_loss
from python.utils import count_parameters



# System 
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

lstm_layers = 2
lstm_hidden_size = 1024 
eps = 1e-8
batch_size = 64
seq_length = 15

# Training
learning_rate = 1e-4
weight_decay = 1e-4
momentum = 0.9
log_interval = 1
start_epoch = 1
end_epoch = 100


model_name = 'Video_Classifier_end_epoch_{:03d}'.format(end_epoch)

print('Load data')
train_files = []
train_data_path = "data/complete/matlab_raw/train/"
speaker_folders_train = sorted(os.listdir(train_data_path))
for speaker_folder in speaker_folders_train:
	video_folder_path = os.path.join(train_data_path, speaker_folder)
	video_files = sorted([x for x in os.listdir(video_folder_path) if '.mat' in x])
	for video_file in video_files:
		train_files.append("train/{}/{}".format(speaker_folder, video_file[:-4]))

dataset_train = VideoFrames(train_files, seq_length)

valid_files = []
valid_data_path = "data/complete/matlab_raw/dev/"
speaker_folders_valid = sorted(os.listdir(valid_data_path))
for speaker_folder in speaker_folders_valid:
	video_folder_path = os.path.join(valid_data_path, speaker_folder)
	video_files = sorted([x for x in os.listdir(video_folder_path) if '.mat' in x])
	for video_file in video_files:
		valid_files.append("dev/{}/{}".format(speaker_folder, video_file[:-4]))

dataset_valid = VideoFrames(valid_files, seq_length)

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, drop_last=True)


print('- Number of training samples: {}'.format(len(dataset_train)))
print('- Number of validation samples: {}'.format(len(dataset_valid)))


def main():
    print('Create model')
    model = VideoClassifier(lstm_layers, lstm_hidden_size, batch_size)

    if cuda: model = model.to(device)

    # Create model folder
    model_dir = os.path.join('models', model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Start log file
    file = open(model_dir + '/' +'output_batch.log','w') 
    file = open(model_dir + '/' +'output_epoch.log','w') 

    # Optimizer settings
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss().cuda()


    t = len(train_loader)
    m = len(valid_loader)
    print('- Number of learnable parameters: {}'.format(count_parameters(model)))

    print('Start training')
    for epoch in range(start_epoch, end_epoch):
        model.train()
        total_loss, total_tp, total_tn, total_fp, total_fn = (0, 0, 0, 0, 0)
        for batch_idx, (x, y) in enumerate(train_loader):
            if cuda:
                x, y = x.to(device), y.long().to(device)

            y_hat_soft = model(x)

            # loss = binary_cross_entropy_2classes(y_hat_soft[:, 0], y_hat_soft[:, 1], y, eps)
            loss = criterion(y_hat_soft, y)

            # loss = binary_cross_entropy(y_hat_soft, y, eps)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            _, y_hat_hard = torch.max(y_hat_soft.data, 1)


            # y_hat_hard = (y_hat_soft[:,0] > 0.5).int()

            f1_score, tp, tn, fp, fn = f1_loss(y_hat_hard=torch.flatten(y_hat_hard), y=torch.flatten(y), epsilon=eps)
            total_tp += tp.item()
            total_tn += tn.item()
            total_fp += fp.item()
            total_fn += fn.item()

            # Save to log
            if batch_idx % log_interval == 0:
                print(('Train Epoch: {:2d}   [{:4d}/{:4d} ({:2d}%)]    Loss: {:.2f}    F1-score.: {:.2f}'\
                    + '').format(epoch, batch_idx*len(x), len(train_loader.dataset), int(100.*batch_idx/len(train_loader)),\
                            loss.item(), f1_score.item()), 
                    file=open(model_dir + '/' + 'output_batch.log','a'))

        if epoch % 1 == 0:
            model.eval()

            total_precision = total_tp / (total_tp + total_fp + eps)
            total_recall = total_tp / (total_tp + total_fn + eps) 
            total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall + eps)

            print("Epoch: {}".format(epoch))
            print("[Train]       Loss: {:.2f}    F1_score: {:.2f}".format(total_loss / t, total_f1_score))

            # Save to log
            print(("Epoch: {}".format(epoch)), file=open(model_dir + '/' + 'output_epoch.log','a'))
            print("[Train]       Loss: {:.2f}    F1_score: {:.2f}".format(total_loss / t, total_f1_score),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            total_loss, total_tp, total_tn, total_fp, total_fn = (0, 0, 0, 0, 0)

            for batch_idx, (x, y) in enumerate(valid_loader):

                if cuda:
                    x, y = x.to(device), y.long().to(device)

                y_hat_soft = model(x)
                loss = criterion(y_hat_soft, y)

                total_loss += loss.item()
                _, y_hat_hard = torch.max(y_hat_soft.data, 1)

                f1_score, tp, tn, fp, fn = f1_loss(y_hat_hard=torch.flatten(y_hat_hard), y=torch.flatten(y), epsilon=eps)
                total_tp += tp.item()
                total_tn += tn.item()
                total_fp += fp.item()
                total_fn += fn.item()

            total_precision = total_tp / (total_tp + total_fp + eps)
            total_recall = total_tp / (total_tp + total_fn + eps) 
            total_f1_score = 2 * (total_precision * total_recall) / (total_precision + total_recall + eps)

            print("[Validation]  Loss: {:.2f}    F1_score: {:.2f}".format(total_loss / m, total_f1_score))

            # Save to log
            print("[Validation] Loss: {:.2f}    F1_score: {:.2f}".format(total_loss / m, total_f1_score),
                file=open(model_dir + '/' + 'output_epoch.log','a'))

            # Save model
            torch.save(model.state_dict(), model_dir + '/' + 'Video_Net_epoch_{:03d}_vloss_{:.2f}.pt'.format(
                epoch, total_loss / m))

if __name__ == '__main__':
    main()
