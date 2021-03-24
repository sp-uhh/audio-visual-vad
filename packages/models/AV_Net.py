import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
# from networks.wavenet_autoencoder import wavenet_autoencoder
from .compact_bilinear_pooling import CountSketch, CompactBilinearPooling
from torch.autograd import Variable
from .utils import weights_init_normal
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Audio_Visual network
class DeepVAD_AV(nn.Module):
    def __init__(self, lstm_layers, lstm_hidden_size, y_dim, use_mcb=False, eps=1e-8):
        super(DeepVAD_AV, self).__init__()

        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.y_dim = y_dim
        self.dropout = nn.Dropout(p=0.05)
        self.use_mcb = use_mcb
        self.eps = eps

        # video related init

        resnet = models.resnet18(pretrained=False)  # set self.num_video_ftrs = 512

        self.num_video_ftrs = 512
        self.features = nn.Sequential(
            *list(resnet.children())[:-1]# drop the last FC layer
        )

        # self.bn = torch.nn.BatchNorm1d(self.num_video_ftrs, eps=1e-5, momentum=0.1, affine=True)
        self.bn = torch.nn.BatchNorm1d(self.num_video_ftrs, eps=self.eps, momentum=0.1, affine=True)
        
        #audio related init

        self.num_audio_ftrs = 513

        # general init

        if self.use_mcb:
            self.mcb_output_size = 1024
            # self.mcb_output_size = 512
            # self.mcb_output_size = 256
            self.lstm_input_size = self.mcb_output_size
            # self.mcb = CompactBilinearPooling(self.num_audio_ftrs, self.num_video_ftrs, self.mcb_output_size).cuda()
            self.mcb = CompactBilinearPooling(self.num_audio_ftrs, self.num_video_ftrs, self.mcb_output_size)
            # self.mcb_bn = torch.nn.BatchNorm1d(self.mcb_output_size, eps=1e-05, momentum=0.1, affine=True)
            self.mcb_bn = torch.nn.BatchNorm1d(self.mcb_output_size, eps=self.eps, momentum=0.1, affine=True)
        else:
            self.lstm_input_size = self.num_audio_ftrs + self.num_video_ftrs

        self.lstm_merged = nn.LSTM(input_size=self.lstm_input_size ,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=False)

        self.vad_merged = nn.Linear(self.lstm_hidden_size, y_dim)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.named_parameters():
            weights_init_normal(m, mean=mean, std=std)

    def init_hidden(self,is_train):
        if is_train:
            return (Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda())

    def forward(self, audio, video, lengths):

        # # Try w/o video
        # video[:] = 0.

        # Video branch
        batch, frames, height, width = video.size()
        channels = 3
        
        # Duplicate x to fit ResNet
        video = video.unsqueeze(2).repeat(1, 1, channels, 1, 1) # batch, frames, channels, height, width
        
        # Reshape to (batch * seq_len, channels, height, width)
        video = video.view(batch*frames,channels,height,width)
        
        video = self.features(video).squeeze() # output shape - Batch X Features X seq len
        # video = self.dropout(video)
        
        # # Reshape to (seq_len, batch, Features)
        # video = video.permute(1, 0, 2)
        
        # Reshape to (batch , seq_len, Features)
        video = video.view(batch , frames, -1)
        
        # # Batch norm before concatenating
        # video = video.permute(1, 2, 0).contiguous()
        # video = self.bn(video)
        # video = video.permute(2, 0, 1).contiguous()
        
        # Audio branch
        # audio = self.wavenet_en(audio) # output shape - Batch X Features X seq len
        # audio = self.bn(audio)
        # audio = self.dropout(audio)  # output shape - Batch X Features X seq len
        # Reshape to (seq_len, batch, input_size)
        # audio = audio.permute(2, 0, 1)

        # Merging branches
        if self.use_mcb:
            #TODO: modify
            y = self.mcb(audio, video)
            # signed square root
            # y =  torch.mul(torch.sign(y), torch.sqrt(torch.abs(y) + 1e-12)) # or y = torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))
            y =  torch.mul(torch.sign(y), torch.sqrt(torch.abs(y) + self.eps)) # or y = torch.sqrt(F.relu(x)) - torch.sqrt(F.relu(-x))
            # y =  torch.sqrt(F.relu(y)) - torch.sqrt(F.relu(-y))
            # L2 normalization
            y = y / torch.norm(y, p=2).detach()

            y = y.permute(1, 2, 0).contiguous()
            y = self.mcb_bn(y)
            y = y.permute(2, 0, 1).contiguous()

        else:
            y = torch.cat([audio,video],dim=2)
      
        # Merged branch
        total_length = y.size(1) # to make unpacking work with DataParallel
        y = pack_padded_sequence(y, lengths=lengths, enforce_sorted=False, batch_first=True)

        # y = self.dropout(y)
        # out, h = self.lstm_merged(y, h)  # output shape - seq len X Batch X lstm size
        self.lstm_merged.flatten_parameters() # Avoid warning: RNN module weights are not part of single contiguous chunk of memory
        out, _ = self.lstm_merged(y)  # output shape - seq len X Batch X lstm size
        # out = self.dropout(out[-1]) # select last time step. many -> one
        
        # Unpack the feature vector & get last output
        out, lens_unpacked = pad_packed_sequence(out, batch_first=True, total_length=total_length) # to make unpacking work with DataParallel
        
        # out = F.sigmoid(self.vad_merged(out))
        out = self.vad_merged(out)
        return out

