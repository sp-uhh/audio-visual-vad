import sys
sys.path.append('.')

import os
import torch
import pickle
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import h5py as h5

from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
# from video_net import VideoClassifier
# from data_handling import VideoFrames

from packages.data_handling import HDF5SequenceSpectrogramLabeledFrames
from packages.models.Video_Net import DeepVAD_video
from packages.models.utils import binary_cross_entropy, binary_cross_entropy_2classes, f1_loss
from packages.utils import count_parameters, my_collate

# Dataset
# dataset_size = 'subset'
dataset_size = 'complete'

dataset_name = 'ntcd_timit'
data_dir = 'export'
labels = 'vad_labels'

# Distributed Data Parallel
n_machines = 1
ngpus_per_node = 4
world_size = ngpus_per_node * n_machines
rank = 0 # redefined below

# System 
cuda = torch.cuda.is_available()
# cuda_device = "cuda:0"
# device = torch.device(cuda_device if cuda else "cpu")
num_workers = 8 # redefined below
pin_memory = True
non_blocking = True
rdcc_nbytes = 1024**2*400  # The number of bytes to use for the chunk cache
                           # Default is 1 Mb
                           # Here we are using 400Mb of chunk_cache_mem here
rdcc_nslots = 1e5 # The number of slots in the cache's hash table
                  # Default is 521
                  # ideally 100 x number of chunks that can be fit in rdcc_nbytes
                  # (see https://docs.h5py.org/en/stable/high/file.html?highlight=rdcc#chunk-cache)
eps = 1e-8

# Deep Generative Model
x_dim = 513 
if labels == 'noisy_labels':
    y_dim = 513
if labels == 'noisy_vad_labels':
    y_dim = 1
# h_dim = [128, 128]
lstm_layers = 2
lstm_hidden_size = 1024 
seq_length = 15
# seq_length = 5
batch_norm=False
std_norm =True


# Training
batch_size = 64 # redefined below
learning_rate = 1e-4
# weight_decay = 1e-4
# momentum = 0.9
log_interval = 1
start_epoch = 1
end_epoch = 100

if labels == 'vad_labels':
    model_name = 'Video_Classifier_ddp_align_shuffle_normdataset_batch64_seqlength15_end_epoch_{:03d}'.format(end_epoch)

# GPU Multiprocessing
nb_devices = torch.cuda.device_count()
gpu_list = np.arange(nb_devices)

# print('Load data')
# train_files = []
# train_data_path = "data/complete/matlab_raw/train/"
# speaker_folders_train = sorted(os.listdir(train_data_path))
# for speaker_folder in speaker_folders_train:
# 	video_folder_path = os.path.join(train_data_path, speaker_folder)
# 	video_files = sorted([x for x in os.listdir(video_folder_path) if '.mat' in x])
# 	for video_file in video_files:
# 		train_files.append("train/{}/{}".format(speaker_folder, video_file[:-4]))

# dataset_train = VideoFrames(train_files, seq_length)

# valid_files = []
# valid_data_path = "data/complete/matlab_raw/dev/"
# speaker_folders_valid = sorted(os.listdir(valid_data_path))
# for speaker_folder in speaker_folders_valid:
# 	video_folder_path = os.path.join(valid_data_path, speaker_folder)
# 	video_files = sorted([x for x in os.listdir(video_folder_path) if '.mat' in x])
# 	for video_file in video_files:
# 		valid_files.append("dev/{}/{}".format(speaker_folder, video_file[:-4]))

# dataset_valid = VideoFrames(valid_files, seq_length)

# train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
# valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, drop_last=True)


#####################################################################################################

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(gpu, ngpus_per_node):
    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    rank = 0
    batch_size = 64
    num_workers = 16

    rank = rank * ngpus_per_node + gpu

    print(f"Running DDP on rank {rank}.")
    setup(rank, world_size)

    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    batch_size = int(batch_size / ngpus_per_node)
    num_workers = int((num_workers + ngpus_per_node - 1) / ngpus_per_node)

    print('Create model')
    # model = VideoClassifier(lstm_layers, lstm_hidden_size, batch_size)
    model = DeepVAD_video(lstm_layers, lstm_hidden_size, batch_size)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(gpu)
    model.to(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

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

    # DataLoader
    print('Load data')
    output_h5_dir = os.path.join('data', dataset_size, data_dir, dataset_name + '_' + labels + '.h5')

    if std_norm:
        print('Load mean and std')
        # Normalize train_data, valid_data
        # mean = np.mean(np.power(abs(train_data), 2), axis=1)[:, None]
        # std = np.std(np.power(abs(train_data), 2), axis=1, ddof=1)[:, None]
        with h5.File(output_h5_dir, 'r') as file:
            mean = file['X_train_mean'][:]
            std = file['X_train_std'][:]

        # Save mean and std
        np.save(model_dir + '/' + 'trainset_mean.npy', mean)
        np.save(model_dir + '/' + 'trainset_std.npy', std)

        mean = torch.tensor(mean).to(gpu)
        std = torch.tensor(std).to(gpu)

    train_dataset = HDF5SequenceSpectrogramLabeledFrames(output_h5_dir=output_h5_dir,
                                                        dataset_type='train',
                                                        rdcc_nbytes=rdcc_nbytes,
                                                        rdcc_nslots=rdcc_nslots,
                                                        seq_length=seq_length)
    valid_dataset = HDF5SequenceSpectrogramLabeledFrames(output_h5_dir=output_h5_dir,
                                                        dataset_type='validation',
                                                        rdcc_nbytes=rdcc_nbytes,
                                                        rdcc_nslots=rdcc_nslots,
                                                        seq_length=seq_length)

    # Sampler for training
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=train_sampler, 
                            batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                            drop_last=False, timeout=0, worker_init_fn=None, collate_fn=my_collate)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, sampler=None, 
                            batch_sampler=None, num_workers=num_workers, pin_memory=pin_memory, 
                            drop_last=False, timeout=0, worker_init_fn=None, collate_fn=my_collate)

    print('- Number of training samples: {}'.format(len(train_dataset)))
    print('- Number of validation samples: {}'.format(len(valid_dataset)))

    print('- Number of training batches: {}'.format(len(train_loader)))
    print('- Number of validation batches: {}'.format(len(valid_loader)))

    t = len(train_loader)
    m = len(valid_loader)
    print('- Number of learnable parameters: {}'.format(count_parameters(model)))

    print('Start training')
    for epoch in range(start_epoch, end_epoch):
        train_sampler.set_epoch(epoch)

        model.train()

        total_loss, total_tp, total_tn, total_fp, total_fn = (0, 0, 0, 0, 0)
        # for batch_idx, (x, y) in enumerate(train_loader):
        for batch_idx, (lengths, x, y) in tqdm(enumerate(train_loader)):
            if cuda:
                # x, y = x.to(device), y.long().to(device)
                x, y, lengths = x.to(gpu), y.long().to(gpu), lengths.to(gpu)

            # Normalize power spectrogram
            if std_norm:
                x_norm = x - mean.T
                x_norm /= (std + eps).T

                y_hat_soft = model(x_norm, lengths) 
            else:
                y_hat_soft = model(x, lengths)

            # loss = binary_cross_entropy_2classes(y_hat_soft[:, 0], y_hat_soft[:, 1], y, eps)
            y = torch.squeeze(y)
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
            
            # for batch_idx, (x, y) in enumerate(valid_loader):
            for batch_idx, (lengths, x, y) in tqdm(enumerate(valid_loader)):

                if cuda:
                    # x, y = x.to(device), y.long().to(device)
                    x, y, lengths = x.to(gpu), y.long().to(gpu), lengths.to(gpu)

                # y_hat_soft = model(x)
                # Normalize power spectrogram
                if std_norm:
                    x_norm = x - mean.T
                    x_norm /= (std + eps).T

                    y_hat_soft = model(x_norm, lengths) 
                else:
                    y_hat_soft = model(x, lengths)
                
                y = torch.squeeze(y)
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
    
    # Distribued Data Parallel
    cleanup()

def run_demo(fn):
    mp.spawn(
        fn,
        nprocs = ngpus_per_node,
        args = (ngpus_per_node, ),
        # join = True
    )

if __name__ == '__main__':
    # main()
    run_demo(main)
    
