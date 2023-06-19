import torch
import torch.nn as nn

# custom raw-domain denoising autoencoder inspired by demucs and convTasNet

class AutoVerb(nn.Module):
    def __init__(self, blocks, inChannels, channelFactor):
        super(AutoVerb, self).__init__()
        start = inChannels
        self.conv = nn.Conv1d(2, inChannels, kernel_size=5, padding=2, stride=1)
        self.prelu = nn.PReLU(inChannels)
        # encoder
        self.hidden= nn.ModuleList()
        # decoder
        self.decode = nn.ModuleList()
        # determines how many channels we add for each block
        self.cFactor = channelFactor
        # experimented with pqmf
        #self.pqmf = PQMF(attenuation=30, n_band=32)
        blocks = blocks
        dil = 1
        for block in range(blocks):
            stride = 4
            self.hidden.append(Encoder(start, stride=stride, dil=dil, mul=self.cFactor))
            # set channel size for next block
            start = start + self.cFactor

        self.bottleneck = nn.Conv1d(start, start, 3, padding=(((3 - 1) // 2) * dil), dilation=dil)
        self.botAct = nn.PReLU(start)
        self.lstm= nn.LSTM(start, start,bidirectional=True,num_layers=2)
        self.linear = nn.Linear(start * 2, start)

        for block in range(blocks):
                stride = 1
                self.decode.append(kBlockUp(start, stride=stride, dil=dil, mul=self.cFactor))
                # set channel size for next block
                start = start - self.cFactor
        self.process = cutBlock(start, 2)

    def forward(self, mix):
        x = mix.clone()
        #x = self.pqmf(x)
        x = self.prelu(self.conv(x))
        features = []
        features.append(x)
        for module in self.hidden:
            x = module(x)
            # print(x.shape)
            # save features for skip connections
            features.append(x)

        x = self.botAct(self.bottleneck(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = x.permute(0,2,1)

        for i, module in enumerate(self.decode):
            index = i + 1
            # print("SHAPE",features[-abs(index)].size(2))
            # print(x.shape)
            # Match dims from encoder to decoder
            x = x[:, :, :features[-abs(index)].size(2)] + features[-abs(index)]
            x = module(x)
        #print(x.shape)
        x = x[:, :, :mix.size(-1)]
        # remove noise, refine synthesis of final waveform.
        out = self.process(x)
        return out


class Encoder(nn.Module):
    def __init__(self, channels, dil=1, stride=1, mul=1, gru=False):
        super(Encoder, self).__init__()
        self.kernel = 7
        self.stride = stride
        self.gru = gru

        self.conv1 = nn.Conv1d(channels,channels + mul, kernel_size=self.kernel, stride=stride,
                               padding=((self.kernel - 1) // 2) * dil, dilation=dil)
        # optionally use GRU in encoder
        if gru:
            self.gru = nn.GRU(channels + mul, channels + mul, 2)


        self.prelu1 =  nn.PReLU(channels + mul)

    def forward(self, x):
        x =  self.prelu1(self.conv1(x))
        if self.gru:
            x = x.permute(0,2,1)
            x, _ = self.gru(x)
            x = x.permute(0,2,1)

        return x


class kBlockUp(nn.Module):
    def __init__(self, channels, dil, stride=1, mul=1):
        super(kBlockUp, self).__init__()
        self.stride = stride
        self.kernel = 7
        self.conv1 = nn.ConvTranspose1d(channels, channels-mul, stride= 4,kernel_size=self.kernel, padding=0)
        self.conv2 = nn.Conv1d(channels-mul, channels - mul, kernel_size=self.kernel, dilation=dil,
                               padding=((self.kernel - 1) // 2) * dil)  # padding=dil)

        self.prelu1 = nn.PReLU(channels - mul)


    def forward(self, x):
        upscaled = (self.conv1(x))
        x = (self.prelu1((self.conv2(upscaled))))

        return x


# block of conv layers to refine synthesis
class cutBlock(nn.Module):
    def __init__(self, channels, out, dil=1, stride=1, mul=1,):
        super(cutBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels // 2, kernel_size=11, stride=stride, padding=((11 - 1) // 2) * dil)
        self.conv2 = nn.Conv1d(channels // 2, out, kernel_size=3, stride=stride, padding=((3 - 1) // 2) * dil)
        self.prelu = nn.PReLU(channels// 2)
    def forward(self, x):
        out = (self.prelu(self.conv1(x)))
        out = (self.conv2(out))

        return out

# Experimental sin activation inspired by SIREN paper, not used for final network

class SinScale(nn.Module):
    def __init__(self, in_features):
        super(SinScale, self).__init__()
        self.scale = nn.Parameter(torch.Tensor(1, in_features))
        nn.init.uniform_(self.scale, a=0.1, b=1.0)  # Initialize scale parameter

    def forward(self, x):
        # learn scaling for each channel
        scaled_x = x.transpose(1,2) * self.scale
        transformed_x = torch.sin(scaled_x)
        transformed_x = transformed_x.transpose(1,2)
        return transformed_x

