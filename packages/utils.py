import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# a simple custom collate function, just to show the idea
def my_collate(batch):
    lengths = [i[2] for i in batch]   # get the length of each sequence in the batch
    batch_size = len(batch)
    seq_length = max(lengths)
    width, height, channel, _ = batch[0][0].size()
    padded_data = torch.zeros((batch_size, width, height, channel, seq_length))
    target = torch.zeros((batch_size, 1))
    for idx, (sample, length) in enumerate(zip(batch, lengths)):
        # Padd sequence at beginning
        npad = (0, seq_length-length)
        padded_data[idx] = pad(sample[0], npad, mode='constant', value=0.) # pad last dimension
        # npad = (0, 0, 0, seq_length)
        # target[idx] = pad(sample[1], npad, mode='constant', value=0.)
        target[idx] = sample[1]
    # pad_sequence()

    # Put seq_length as 2nd axis using unsqueeze + tranpose
    padded_data = padded_data.unsqueeze(1).transpose(1, -1) # .unsqueeze(location).transpose(location, dim) --> (B, T, *, 1)
    
    # Remove last axis
    padded_data = torch.squeeze(padded_data)

    # Swap channels and width
    padded_data = padded_data.permute(0,1,-1,3,2) # batch,frames,channels,height,width

    # Make dim contiguous
    padded_data = padded_data.contiguous()

    # Make lengths as LongTensor
    lengths = torch.LongTensor(lengths)
    
    return lengths, padded_data, target

def collate_many2many_video(batch):
    # Get dimensions
    lengths = [i[-1] for i in batch]   # get the length of each sequence in the batch
    batch_size = len(batch)
    seq_length = max(lengths)
    # height, width, channel, _ = batch[0][0].size()
    height, width, _ = batch[0][0].size()
    y_dim, _ = batch[0][1].size()

    # padded_data = torch.zeros((batch_size, height, width, channel, seq_length))
    padded_data = torch.zeros((batch_size, height, width, seq_length))
    
    # target = torch.zeros((batch_size, seq_length))
    target = torch.zeros((batch_size, y_dim, seq_length))

    for idx, (sample, length) in enumerate(zip(batch, lengths)):
        # Padd sequence at beginning
        npad = (0, seq_length-length)
        padded_data[idx] = pad(sample[0], npad, mode='constant', value=0.) # pad last dimension
        target[idx] = pad(sample[1], npad, mode='constant', value=0.)

    # Put seq_length as 2nd axis using unsqueeze + tranpose
    padded_data = padded_data.unsqueeze(1).transpose(1, -1) # .unsqueeze(location).transpose(location, dim) --> (B, T, *, 1)
    target = target.unsqueeze(1).transpose(1, -1) # .unsqueeze(location).transpose(location, dim) --> (B, T, *, 1)
    
    # Remove last axis 
    padded_data = padded_data[...,0]
    target = target[...,0]

    # # Swap channels and width
    # padded_data = padded_data.permute(0,1,-1,3,2) # batch,frames,channels,width,height

    # Make dim contiguous
    padded_data = padded_data.contiguous()

    # Make lengths as LongTensor
    lengths = torch.LongTensor(lengths)
    
    return lengths, padded_data, target

def collate_many2many_audio(batch):
    # Get dimensions
    lengths = [i[-1] for i in batch]   # get the length of each sequence in the batch
    batch_size = len(batch)
    seq_length = max(lengths)
    x_dim, _ = batch[0][0].size()
    y_dim, _ = batch[0][1].size()

    padded_data = torch.zeros((batch_size, x_dim, seq_length))
    
    target = torch.zeros((batch_size, y_dim, seq_length))

    for idx, (sample, length) in enumerate(zip(batch, lengths)):
        # Padd sequence at beginning
        npad = (0, seq_length-length)
        padded_data[idx] = pad(sample[0], npad, mode='constant', value=0.) # pad last dimension
        target[idx] = pad(sample[1], npad, mode='constant', value=0.)

    # Put seq_length as 2nd axis using unsqueeze + tranpose
    padded_data = padded_data.unsqueeze(1).transpose(1, -1) # .unsqueeze(location).transpose(location, dim) --> (B, T, *, 1)
    target = target.unsqueeze(1).transpose(1, -1) # .unsqueeze(location).transpose(location, dim) --> (B, T, *, 1)
    
    # Remove last axis 
    padded_data = padded_data[...,0]
    target = target[...,0]

    # Make dim contiguous
    padded_data = padded_data.contiguous()

    # Make lengths as LongTensor
    lengths = torch.LongTensor(lengths)
    
    return lengths, padded_data, target

def collate_many2many_AV(batch):
    # Get dimensions
    lengths = [i[-1] for i in batch]   # get the length of each sequence in the batch
    batch_size = len(batch)
    seq_length = max(lengths)
    x_dim, _ = batch[0][0].size()
    height, width, _ = batch[0][1].size()
    y_dim, _ = batch[0][2].size()

    padded_audio = torch.zeros((batch_size, x_dim, seq_length))
    padded_video = torch.zeros((batch_size, height, width, seq_length))    
    target = torch.zeros((batch_size, y_dim, seq_length))

    for idx, (sample, length) in enumerate(zip(batch, lengths)):
        # Padd sequence at beginning
        npad = (0, seq_length-length)
        padded_audio[idx] = pad(sample[0], npad, mode='constant', value=0.) # pad last dimension
        padded_video[idx] = pad(sample[1], npad, mode='constant', value=0.) # pad last dimension
        target[idx] = pad(sample[2], npad, mode='constant', value=0.)

    # Put seq_length as 2nd axis using unsqueeze + tranpose
    padded_audio = padded_audio.unsqueeze(1).transpose(1, -1) # .unsqueeze(location).transpose(location, dim) --> (B, T, *, 1)
    padded_video = padded_video.unsqueeze(1).transpose(1, -1) # .unsqueeze(location).transpose(location, dim) --> (B, T, *, 1)
    target = target.unsqueeze(1).transpose(1, -1) # .unsqueeze(location).transpose(location, dim) --> (B, T, *, 1)
    
    # Remove last axis 
    padded_audio = padded_audio[...,0]
    padded_video = padded_video[...,0]
    target = target[...,0]

    # Make dim contiguous
    padded_audio = padded_audio.contiguous()
    padded_video = padded_video.contiguous()

    # Make lengths as LongTensor
    lengths = torch.LongTensor(lengths)
    
    return lengths, padded_audio, padded_video, target