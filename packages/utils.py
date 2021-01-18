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