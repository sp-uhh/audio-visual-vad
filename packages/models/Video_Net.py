import torch
import torch.nn as nn
import torchvision.models as models
# import torch.nn.functional as F
from torch.autograd import Variable
from packages.models.utils import weights_init_normal, method1, method3

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import transforms

# VIDEO only network
class DeepVAD_video(nn.Module):

    def __init__(self, lstm_layers, lstm_hidden_size, y_dim):
        super(DeepVAD_video, self).__init__()

        # resnet = models.resnet18(pretrained=True) # set num_ftrs = 512
        resnet = models.resnet18(pretrained=False) # set num_ftrs = 512
        # resnet = models.resnet34(pretrained=True) # set num_ftrs = 512
        # resnet = models.resnet34(pretrained=False) # set num_ftrs = 512

        num_ftrs = 512
        # num_ftrs = 13467
        # num_ftrs = 4489

        self.lstm_input_size = num_ftrs
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.y_dim = y_dim

        self.features = nn.Sequential(
            *list(resnet.children())[:-1]# drop the last FC layer
        )
        
        # Normalize input data the same way ResNet was pretrained
        self.mean = torch.as_tensor([0.485, 0.456, 0.406])
        self.std = torch.as_tensor([0.229, 0.224, 0.225])

        # self.mlp = nn.Linear(4489, 512)

        self.lstm_video = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=False)

        # self.vad_video = nn.Linear(self.lstm_hidden_size, 2)
        self.vad_video = nn.Linear(self.lstm_hidden_size, y_dim)
        self.dropout = nn.Dropout(p=0.5)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.named_parameters():
            weights_init_normal(m, mean=mean, std=std)

    def forward(self, x, lengths, return_last=False):
        # batch, frames, channels, height, width = x.size()
        batch, frames, height, width = x.size()
        channels = 3
        
        # Duplicate x to fit ResNet
        x = x.unsqueeze(2).repeat(1, 1, channels, 1, 1) # batch, frames, channels, height, width

        # Reshape to (batch * seq_len, channels, height, width)
        x = x.view(batch*frames,channels,height,width)
        # x = x.view(batch*frames,height,width)
        
        # # Normalize input data the same way ResNet was pretrained
        # # Set images in range [0,1]
        # x_max, _ = torch.max(x.view(batch*frames, -1), axis=1) # along frame axis
        # x_min, _ = torch.min(x.view(batch*frames, -1), axis=1) # along frame axis
        # x = (x - x_min[:,None,None,None]) /(x_max[:,None,None,None] - x_min[:,None,None,None])
        
        # x -= self.mean[None,:, None, None].to(x.device)
        # x /= self.std[None,:, None, None].to(x.device)

        x = self.features(x).squeeze() # output shape - Batch X Features X seq len
        # x = self.dropout(x)
        # Reshape to (batch , seq_len, Features)
        x = x.view(batch , frames, -1)

        # x = self.mlp(x)
        # x = torch.tanh(x)

        # Pack the feature vector
        # input_dim must be (batch, seq_len, Features)
        total_length = x.size(1) # to make unpacking work with DataParallel
        x = pack_padded_sequence(x, lengths=lengths, enforce_sorted=False, batch_first=True)

        # out, _ = self.lstm_video(x, h)  # output shape - seq len X Batch X lstm size
        out, _ = self.lstm_video(x)  # output shape - seq len X Batch X lstm size
        
        # Unpack the feature vector & get last output
        ##TODO: change and get the whole sequence
        if return_last:
            out = method3(out, lengths)
            # out = method1(out)
        else:
            out, lens_unpacked = pad_packed_sequence(out, batch_first=True, total_length=total_length) # to make unpacking work with DataParallel

        # out = self.dropout(out)
        out = self.vad_video(out)
        return out

    def init_hidden(self,is_train):
        if is_train:
            return (Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda(),
                      Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda())