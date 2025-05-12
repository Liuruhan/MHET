import sys
sys.path.append('/home3/liuruhan/sleep_model/sleep_model/git_code/')

import math

import torch
import torch.nn as nn
import torchvision

from single_step_models.model_utils import Pad_Pool

Fs = 100

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, activation=nn.ReLU()):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_filters,
                out_filters,
                5,
                stride=1,
                padding=2,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
            activation,
            nn.BatchNorm1d(out_filters),
            nn.Conv1d(
                out_filters,
                out_filters,
                5,
                stride=1,
                padding=2,
                dilation=1,
                groups=1,
                bias=True,
                padding_mode="zeros",
            ),
            activation,
            nn.BatchNorm1d(out_filters),
        )

    def forward(self, x):
        return self.block(x)


class UUnit(nn.Module):
    def __init__(
            self, in_channel, out_channel, middle_layer_filter=16, pools=2, depth=3, kernel_size=5
    ):
        super(UUnit, self).__init__()
        # Encoder Part
        self.input_layer = Block(in_channel, middle_layer_filter)
        self.encBlocks = nn.ModuleList(
            [Block(middle_layer_filter, middle_layer_filter) for i in range(depth - 1)]
        )
        self.pools = [nn.MaxPool1d(pools) for i in range(depth - 1)]
        self.middle_layer = Block(middle_layer_filter, middle_layer_filter)
        # Decoder Part
        self.upconv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=pools, mode="linear"),
                    nn.Conv1d(
                        middle_layer_filter, middle_layer_filter, kernel_size=pools, padding=0
                    ),
                )
                for i in range(depth - 1)
            ]
        )
        self.decBlocks = nn.ModuleList(
            [Block(2 * middle_layer_filter, middle_layer_filter) for i in range(depth - 1)]
        )

        self.encoder_output = Block(middle_layer_filter, out_channel)

    def forward(self, x0):
        output = []
        x0 = self.input_layer(x0)
        x = x0

        # Encoding
        for i, block in enumerate(self.encBlocks):
            x = block(x)
            output.append(x)
            x = self.pools[i](x)

        x = self.middle_layer(x)

        # Decoding
        for i in range(len(self.decBlocks)):
            x = self.upconv[i](x)
            residual = output.pop()
            encFeat = self.crop(residual, x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.decBlocks[i](x)

        x = x0 + self.pad(x0, x)
        x = self.encoder_output(x)
        return x

    def pad(self, x0, x1):
        """
        pads x1 to size of x0
        :param x0:
        :param x1:
        :return:
        """
        _, _, seq_length = x0.size()
        _, _, segment_length = x1.size()
        top = math.floor((seq_length - segment_length) / 2)
        bottom = math.ceil((seq_length - segment_length) / 2)
        output = nn.ZeroPad2d(padding=(bottom, top, 0, 0))(x1)
        return output

    def crop(self, encFeatures, x):
        _, c, t = x.size()
        encFeatures = torchvision.transforms.CenterCrop([c, t])(encFeatures)
        return encFeatures


class MSE(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, dilation_rates=[], kernel_size=5):
        super().__init__()
        # print(f'MSE input_channel {input_channels}')

        # print(f'MSE output_channel {output_channels}')
        self.encBlocks = nn.ModuleList(
            [
                nn.Conv1d(
                    input_channels,
                    output_channels,
                    kernel_size=kernel_size,
                    dilation=i,
                    padding=int(i * (kernel_size - 1) / 2),
                )
                for i in dilation_rates
            ]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                output_channels * len(dilation_rates), 2 * output_channels, kernel_size=kernel_size
            ),
            nn.Conv1d(2 * output_channels, output_channels, kernel_size=kernel_size),
            nn.BatchNorm1d(output_channels),
        )

    def forward(self, x):
        enc = []
        for block in self.encBlocks:
            enc.append(block(x))
        output = torch.cat(enc, dim=1)
        output = self.bottleneck(output)
        return output

class SalientSleepNet(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 encChannels=(16, 32, 64, 128),
                 bottleneck=128,
                 decChannels=(64, 32, 16),
                 mse_channels=(4, 2, 1),
                 pools=(2, 2, 2),
                 nb_outlayer_channels=5):
        super(SalientSleepNet, self).__init__()

        self.output_shape = output_shape
        self.output_channel = decChannels[-1]
        self.nb_outlayer_channels = nb_outlayer_channels
        self.timesamples = input_shape[0]
        self.input_channels = input_shape[1]
        self.decChannels = (bottleneck,) + decChannels
        self.dilation_rates = [2, 4, 8]

        # print('input_channel:', self.input_channels, encChannels[0])
        self.input_channel = nn.Conv1d(self.input_channels, encChannels[0], kernel_size=5)
        self.encChannels = encChannels
        self.encBlocks = nn.ModuleList(
            [
                UUnit(self.encChannels[i], self.encChannels[i + 1], encChannels[i])
                for i in range(len(encChannels) - 1)
            ]
        )

        self.pools = [nn.MaxPool1d(pools[i]) for i in range(len(pools))]
        # after each pool the channel output size is
        self.bottleneck = UUnit(encChannels[-1], bottleneck, bottleneck)
        # MSE preserves channel sizes, so we use them like on the encoding channel outputs
        self.mse_channels = mse_channels
        l = len(mse_channels)
        self.mse = nn.ModuleList(
            [
                MSE(self.encChannels[l - i], self.mse_channels[i], self.dilation_rates)
                for i in range(len(mse_channels))
            ]
        )
        self.bottleneck_mse = MSE(bottleneck, bottleneck, self.dilation_rates)
        # decoding sizes

        mse_inputs = self.mse_channels
        self.upconv = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Upsample(scale_factor=pools[i], mode="linear"),
                    nn.Conv1d(
                        self.decChannels[i], self.decChannels[i], kernel_size=pools[i], padding=0
                    ),
                )
                for i in range(len(pools))
            ]
        )
        self.decBlocks = nn.ModuleList(
            [
                UUnit(self.decChannels[i] + mse_inputs[i], self.decChannels[i + 1])
                for i in range(len(self.decChannels) - 1)
            ]
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.channel_att = nn.Sequential(
            nn.Linear(self.output_channel, self.output_channel),
            nn.ReLU(),
            nn.Linear(self.output_channel, self.output_channel),
            nn.Sigmoid(),
        )

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
        # print('linear', self.output_shape)
        self.out_linear = nn.Sequential(
            nn.Linear(input_shape[0], self.output_shape[1]),
            # nn.Softmax() # this is done in Dice Loss
        )

    def forward(self, x):
        # x = x[:, :, 0, :]
        # x = x.permute(0, 2, 1)
        # _, seq_length, _ = x.size()
        # x = x.permute(0, 2, 1)
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        _, _, seq_length = x1.size()
        
        # encoder 1
        x1 = self.input_channel(x1)

        output = []
        for i, block in enumerate(self.encBlocks):
            x1 = block(x1)
            output.append(x1)
            x1 = self.pools[i](x1)
        x1 = self.bottleneck(x1)

        x1 = self.bottleneck_mse(x1)

        # encoder 2
        x2 = self.input_channel(x2)

        output = []
        for i, block in enumerate(self.encBlocks):
            x2 = block(x2)
            output.append(x2)
            x2 = self.pools[i](x2)
        x2 = self.bottleneck(x2)

        x2 = self.bottleneck_mse(x2)
        x = x1 + x2

        # decoding
        for i in range(len(self.decBlocks)):
            # pass the inputs through the upsampler blocks
            x = self.upconv[i](x)
            residual = output.pop()
            residual = self.mse[i](residual)
            encFeat = self.pad(residual, x)
            x = torch.cat([residual, encFeat], dim=1)
            x = self.decBlocks[i](x)

        _, _, segment_length = x.size()
        top = math.floor((seq_length - segment_length) / 2)
        bottom = math.ceil((seq_length - segment_length) / 2)
        x = nn.ZeroPad2d(padding=(bottom, top, 0, 0))(x)

        x = self.output_layer(x)
        x = self.out_linear(x)
        return x

    def pad(self, x0, x1):
        _, _, seq_length = x0.size()
        _, _, segment_length = x1.size()
        top = math.floor((seq_length - segment_length) / 2)
        bottom = math.ceil((seq_length - segment_length) / 2)
        output = nn.ZeroPad2d(padding=(bottom, top, 0, 0))(x1)
        return output

    def crop_seq(self, encFeatures, x):
        _.c, t = encFeatures.size()
        _, _, t = x.size()
        encFeatures = torchvision.transforms.CenterCrop([c, t])(encFeatures)
        return encFeatures

    def get_nb_channels_output_layer(self):
        return self.output_channel

    def get_nb_features_output_layer(self):
        return self.output_channel * self.timesamples


if __name__ == "__main__":
    batch, chan, time, steps = 4, 2, Fs * 30, 20
    out_chan, out_width = 1, 20
    model = SalientSleepNet(input_shape=(time, steps), output_shape=(out_chan, out_width))
    inputs = torch.randn(batch, chan, steps, time)
    print('input:', inputs.size())
    out = model(inputs)
    print('output:', out.size())

    from thop import profile

    flops, params = profile(model, inputs=(inputs,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
