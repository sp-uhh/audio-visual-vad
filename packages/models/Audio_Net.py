import torch
import torch.nn as nn
import torch.nn.functional as F
# from networks.wavenet_autoencoder import wavenet_autoencoder
from torch.autograd import Variable
from packages.models.utils import weights_init_normal

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# AUDIO only network
class DeepVAD_audio(nn.Module):
    def __init__(self, lstm_layers, lstm_hidden_size, y_dim):
        super(DeepVAD_audio, self).__init__()

        num_ftrs = 513

        self.lstm_input_size = num_ftrs
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.y_dim = y_dim

        # import json
        # with open('./params/model_params.json', 'r') as f:
        #     params = json.load(f)

        # self.wavenet_en = wavenet_autoencoder(
        #     **params)  # filter_width, dilations, dilation_channels, residual_channels, skip_channels, quantization_channels, use_bias
       
        # self.lstm_audio = nn.LSTM(input_size=params["en_bottleneck_width"],
        self.lstm_audio = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=False)

        self.vad_audio = nn.Linear(self.lstm_hidden_size, y_dim)
        self.dropout = nn.Dropout(p=0.5)
        # self.bn = torch.nn.BatchNorm1d(params["en_bottleneck_width"], eps=1e-05, momentum=0.1, affine=True)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.named_parameters():
            weights_init_normal(m, mean=mean, std=std)

    def forward(self, x, lengths):
        # x = self.wavenet_en(x) # output shape - Batch X Features X seq len
        # x = self.bn(x)

        # # Reshape to (seq_len, batch, input_size)
        # x = x.view(batch , frames, -1)
        
        total_length = x.size(1) # to make unpacking work with DataParallel
        x = pack_padded_sequence(x, lengths=lengths, enforce_sorted=False, batch_first=True)

        out, _ = self.lstm_audio(x)  # output shape - seq len X Batch X lstm size
        
        # Unpack the feature vector & get last output
        out, lens_unpacked = pad_packed_sequence(out, batch_first=True, total_length=total_length) # to make unpacking work with DataParallel

        # out = self.dropout(out)
        out = self.vad_audio(out)
        return out

    def init_hidden(self,is_train):
        if is_train:
            return (Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda(),
                      Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda())