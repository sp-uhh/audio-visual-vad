# import json
# with open('./params/model_params.json', 'r') as f:
#     params = json.load(f)

import torch
import torch.nn as nn
import torchvision.models as models

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

seq_len = 15

class VideoClassifier(nn.Module):

    def __init__(self, lstm_layers, lstm_hidden_size, batch_size):
        super(VideoClassifier, self).__init__()

        resnet = models.resnet18(pretrained=True) 

        self.lstm_input_size = 512
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.batch_size = batch_size
        self.test_batch_size = batch_size
        self.features = nn.Sequential(*list(resnet.children())[:-1]) # drop the last FC layer
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers, bidirectional=False)
        self.vad_video = nn.Linear(self.lstm_hidden_size, 2)
        self.dropout = nn.Dropout(p=0.0)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.named_parameters():
            weights_init_normal(m, mean=mean, std=std)

    def forward(self, x):

        batch, frames, channels, height, width = x.shape

        x = x.view(batch*frames, channels, height,width) # Reshape to (batch * seq_len, channels, height, width)
        x = self.features(x) # output shape - Batch X Features X seq len
        x = self.dropout(x)
        x = x.view(batch, frames, -1) # Reshape to (batch , seq_len, Features)
        x = x.permute(1, 0, 2) # Reshape to (seq_len, batch, Features)

        h0 = torch.randn(self.lstm_layers, self.batch_size, self.lstm_hidden_size).to(device) 
        c0 = torch.randn(self.lstm_layers, self.batch_size, self.lstm_hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))  # output shape - seq len X Batch X lstm size
        out = self.dropout(out[-1])  # select last time step. many -> one
        out = torch.sigmoid(self.vad_video(out))
        # out = self.softmax(self.vad_video(out))
        return out

    def test(self, x):
        batch, frames, channels, height, width = x.shape

        x = x.view(batch*frames, channels, height,width) # Reshape to (batch * seq_len, channels, height, width)
        x = self.features(x) # output shape - Batch X Features X seq len
        x = self.dropout(x)
        x = x.view(batch, frames, -1) # Reshape to (batch , seq_len, Features)
        x = x.permute(1, 0, 2) # Reshape to (seq_len, batch, Features)

        h0 = torch.randn(self.lstm_layers, self.batch_size, self.lstm_hidden_size).to(device) 
        c0 = torch.randn(self.lstm_layers, self.batch_size, self.lstm_hidden_size).to(device)

        y_hat_soft = []
        for i in range(-seq_len, len(x)):
            if i < 0:
                frame = x[0]
            else:
                frame = x[i]
            out, (h0, c0) = self.lstm(frame[None,:,:], (h0, c0))  # output shape - seq len X Batch X lstm size
            out = self.dropout(out[-1])  # select last time step. many -> one
            out = torch.sigmoid(self.vad_video(out))
            y_hat_soft.append(out[0,:])
        return y_hat_soft[seq_len:]



    def init_hidden(self,is_train):
        if is_train:
            return (Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda(),
                      Variable(torch.zeros(self.lstm_layers, self.batch_size, self.lstm_hidden_size)).cuda())
        else:
            return (Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda(),
                    Variable(torch.zeros(self.lstm_layers, self.test_batch_size, self.lstm_hidden_size)).cuda())