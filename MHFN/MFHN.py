import math

import torch
import torch.nn as nn
from single_step_models.model_utils import TCSConv1d, Pad_Pool, Pad_Conv

Fs = 100

class Xception(nn.Module):
    def __init__(self, input_shape, output_shape, kernel_size=40, nb_filters=64, depth=4, use_residual=True, nb_outlayer_channels=5):
        super(Xception, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.timesamples = self.input_shape[0]
        self.nb_channels = self.input_shape[1]

        self.kernel_size = kernel_size
        self.nb_features = nb_filters
        self.depth =  depth
        self.use_residual = use_residual
        self.nb_outlayer_channels = nb_outlayer_channels

        self.conv_blocks = nn.ModuleList([self._module(d) for d in range(self.depth)])
        if self.use_residual:
            self.shortcuts = nn.ModuleList([self._shortcut(d) for d in range(int(self.depth / 3))])
        self.gap_layer = nn.AvgPool1d(kernel_size=2, stride=1)
        self.gap_layer_pad = Pad_Pool(left=0, right=1, value=0)
        self.downsample = nn.MaxPool1d(kernel_size=2)

        self.output_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=self.get_nb_channels_output_layer(),
                out_channels=min(
                    self.get_nb_features_output_layer(), self.nb_outlayer_channels
                ),
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=self.nb_outlayer_channels),
            nn.ReLU(),
            Pad_Pool(left=0, right=1, value=0),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )
        self.out_linear = nn.Sequential(
            nn.Linear(input_shape[0], self.output_shape[1]),
            # nn.Softmax() # this is done in Dice Loss
        )

    def forward(self, x):
        """
        Implements the forward pass of the network
        Modules defined in a class implementing ConvNet are stacked and shortcut connections are
        used if specified.
        """
        #changed from training
        x = x[:, :, 0, :]
        input_res = x # set for the residual shortcut connection
        # Stack the modules and residual connection
        shortcut_cnt = 0
        for d in range(self.depth):
            x = self.conv_blocks[d](x)
            if self.use_residual and d % 3 == 2:
                res = self.shortcuts[shortcut_cnt](input_res)
                shortcut_cnt += 1
                x = torch.add(x, res)
                x = nn.functional.relu(x)
                input_res = x
            x = self.downsample(x)
            input_res = self.downsample(input_res)

        x = self.gap_layer_pad(x)
        x = self.gap_layer(x)
        # output = self.output_layer(x) # Defined in BaseNet
        # output = self.out_linear(output)
        # output = output.permute(0, 2, 1)
        return x

    def _module(self, depth):
        """
        The module of Xception. Consists of a separable convolution followed by batch normalization
        and a ReLu activation function.
        Padding=same
        """
        return nn.Sequential(
            TCSConv1d(mother=self, depth=depth, bias=False),
            nn.BatchNorm1d(num_features=self.nb_features),
            nn.ReLU()
        )

    def _shortcut(self, depth):
        """
        Implements a shortcut with a convolution and batch norm
        This is the same for all models implementing ConvNet, therefore defined here
        Padding before convolution for constant tensor shape, similar to tensorflow.keras padding=same
        """
        return nn.Sequential(
                Pad_Conv(kernel_size=self.kernel_size, value=0),
                nn.Conv1d(in_channels=self.nb_channels if depth==0 else self.nb_features,
                             out_channels=self.nb_features, kernel_size=self.kernel_size),
                nn.BatchNorm1d(num_features=self.nb_features)
            )

    def get_nb_features_output_layer(self):
        """
        Return number of features passed into the output layer of the network
        nb.features has to be defined in a model implementing ConvNet
        """
        return self.timesamples * self.nb_outlayer_channels

    # abstract method
    def get_nb_channels_output_layer(self):
        """
        Return the number of channels that the convolution before output layer should take as input
        to reduce them to 1 channel
        This method has to be implemented by models based on BaseNet to compute the number of hidden
        neurons that the output layer takes as input.
        """
        return self.nb_features


class FeedForward(nn.Module):
    def __init__(self,
                 d_model: int,
                 device: str,
                 d_hidden: int = 512):
        super(FeedForward, self).__init__()

        self.linear_1 = torch.nn.Linear(d_model, d_hidden).to(device)
        self.linear_2 = torch.nn.Linear(d_hidden, d_model).to(device)

    def forward(self, x):
        x = self.linear_1(x)
        x = nn.functional.relu(x)
        x = self.linear_2(x)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        # self.downsample = nn.MaxPool1d(kernel_size=2)
        self.W_q = torch.nn.Linear(d_model, q * h).to(device)
        self.W_k = torch.nn.Linear(d_model, q * h).to(device)
        self.W_v = torch.nn.Linear(d_model, v * h).to(device)

        self.W_o = torch.nn.Linear(v * h, d_model).to(device)

        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.score = None

    def forward(self, x, stage):
        # x = x.permute(0, 2, 1)
        # x = self.downsample(x)
        # x = x.permute(0, 2, 1)
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)

        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        score = score.cuda()
        self.score = score

        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0]).cuda()
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2 ** 32 + 1]).expand_as(score[0]).cuda())

        score = nn.functional.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)

        return self_attention, self.score


class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, device=device, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout).to(device)
        self.layerNormal_1 = torch.nn.LayerNorm(d_model).to(device)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model).to(device)

    def forward(self, x, stage):
        residual = x
        x, score = self.MHA(x, stage)
        x = self.dropout(x)
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score

class MHFN(nn.Module):
    def __init__(self, input_shape, output_shape, temp_list=[24, 60, 60], nb_filters=64, nb_outlayer_channels=5):
        super(MHFN, self).__init__()
        self.eeg_extractor = Xception(input_shape=(input_shape[0], 2), output_shape=(output_shape[0], output_shape[1]))
        self.eog_extractor = Xception(input_shape=(input_shape[0], 1), output_shape=(output_shape[0], output_shape[1]))

        self.hour_embed = nn.Embedding(temp_list[0], 64)
        self.minute_embed = nn.Embedding(temp_list[1], 64)
        self.second_embed = nn.Embedding(temp_list[2], 64)

        # self.adapt_ave_pool = nn.AdaptiveAvgPool1d(1)
        self.channel_wise_output = Encoder(d_model=64,
                                                  d_hidden=64,
                                                  q=8,
                                                  v=8,
                                                  h=8,
                                                  mask=True,
                                                  dropout=0.2,
                                                  device='cpu')
        self.step_embedding = torch.nn.Linear(187, 64)
        self.step_wise_output = Encoder(d_model=64,
                                           d_hidden=64,
                                           q=8,
                                           v=8,
                                           h=8,
                                           dropout=0.2,
                                           device='cpu')

        self.output_layer = nn.Sequential(
            nn.Conv1d(
                in_channels=nb_filters,
                out_channels=min(
                    nb_filters, nb_outlayer_channels
                ),
                kernel_size=1,
                stride=1,
            ),
            nn.BatchNorm1d(num_features=nb_outlayer_channels),
            nn.ReLU(),
            Pad_Pool(left=0, right=1, value=0),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )

        self.out_linear = nn.Sequential(
            nn.Linear(nb_filters, nb_outlayer_channels),
            # nn.Softmax() # this is done in Dice Loss
        )
        self.gate = torch.nn.Linear(187 * 64 + 64 * 64, 2)
        self.output_dropout = nn.Dropout(p=0.5)
        self.output_linear = torch.nn.Linear(187 * 64 + 64 * 64, 5)


    def forward(self, x, time, stage):
        # changed from training
        eeg = x[:, :2, :, :]
        eog = x[:, 2, :, :].unsqueeze(1)
        eeg_fm = self.eeg_extractor(eeg)
        eog_fm = self.eog_extractor(eog)
        fm = eeg_fm + eog_fm
        # output = self.output_layer(fm)
        # output = self.adapt_ave_pool(fm)
        # b, _, _ = output.size()
        # output = output.view(b, -1)# Defined in BaseNet
        # output = self.out_linear(output)
        x_ch = fm.transpose(-1, -2)
        pe = torch.ones_like(x_ch[0]).cuda()
        position = torch.arange(0, 187).unsqueeze(-1).cuda()
        temp = torch.Tensor(range(0, 64, 2)).cuda()
        temp = temp * -(math.log(10000) / 64)
        temp = torch.exp(temp).unsqueeze(0).cuda()
        temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
        pe[:, 0::2] = torch.sin(temp).cuda()
        pe[:, 1::2] = torch.cos(temp).cuda()

        hour = self.hour_embed(time[:, 0].unsqueeze(1))
        min = self.minute_embed(time[:, 1].unsqueeze(1))
        sec = self.second_embed(time[:, 2].unsqueeze(1))
        temporal_emb = hour + min + sec
        # temporal_emb = temporal_emb.unsqueeze(1)
        b, copy_dim, l = x_ch.size()
        temporal_emb = temporal_emb.expand(b, copy_dim, l)

        x_ch = x_ch + pe + temporal_emb

        ch_out, _ = self.channel_wise_output(x_ch, stage)

        step_out = self.step_embedding(fm)
        step_out, _ = self.step_wise_output(step_out, stage)

        ch_out = ch_out.reshape(ch_out.shape[0], -1)
        step_out = step_out.reshape(step_out.shape[0], -1)

        gate = nn.functional.softmax(self.gate(torch.cat([ch_out, step_out], dim=-1)), dim=-1)
        encoding = torch.cat([ch_out * gate[:, 0:1], step_out * gate[:, 1:2]], dim=-1)

        output = self.output_dropout(encoding)
        output = self.output_linear(output)
        return output

if __name__ == "__main__":
    batch, chan, time = 4, 3, Fs * 30
    out_chan, out_width = 1, 1
    model = MHFN(input_shape=(time, chan), output_shape=(out_chan, out_width)).cuda()
    inputs = torch.randn(batch, chan, 1, time).cuda()
    time = torch.Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]).to(torch.int64)
    time = time.cuda()
    print('input:', inputs.size())
    out = model(inputs, time, 'train')
    print('output:', out.size())


    from thop import profile

    flops, params = profile(model, inputs=(inputs, time, 'train'))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')