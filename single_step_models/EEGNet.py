import torch
import torch.nn as nn
from single_step_models.model_utils import Pad_Pool, Pad_Conv2d

Fs = 100

class Block_1(nn.Module):
    def __init__(self, kernel_size, F1, D, channels, dropout):
        super(Block_1, self).__init__()
        self.kernel_size = kernel_size
        self.F1 = F1
        self.D = D
        self.channels = channels
        self.dropout_rate = dropout

        self.padconv1 = Pad_Conv2d(kernel=(1, self.kernel_size))
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=self.F1, kernel_size=(1, self.kernel_size), bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(self.F1, False)
        self.depthwise_conv1 = nn.Conv2d(
            in_channels=self.F1,
            out_channels=self.F1 * self.D,
            groups=self.F1,
            kernel_size=(self.channels, 1),
            bias=False,
        )
        self.batchnorm1_2 = nn.BatchNorm2d(self.F1 * self.D)
        self.activation1 = nn.ELU()
        self.padpool1 = Pad_Conv2d(kernel=(1, 16))
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 16), stride=1)
        self.dropout1 = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.padconv1(x)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise_conv1(x)
        x = self.batchnorm1_2(x)
        x = self.activation1(x)
        x = self.padpool1(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        return x

class Block_2(nn.Module):
    def __init__(self, nb_outlayer_channels, F1, D, F2, dropout):
        super(Block_2, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.nb_outlayer_channels = nb_outlayer_channels
        self.dropout_rate = dropout

        self.pad_depthwise2 = Pad_Conv2d(kernel=(1, 64))
        self.depthwise_conv2 = nn.Conv2d(
            in_channels=self.F1 * self.D,
            out_channels=self.F2,
            groups=self.F1 * self.D,
            kernel_size=(1, 64),
            bias=False,
        )
        self.pointwise_conv2 = nn.Conv2d( 
            in_channels=self.F2, out_channels=self.nb_outlayer_channels, kernel_size=1, bias=False
        )
        self.batchnorm2 = nn.BatchNorm2d(self.nb_outlayer_channels, False)
        self.activation2 = nn.ELU()
        self.padpool2 = Pad_Conv2d(kernel=(1, 8))
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=1)
        self.dropout2 = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        x = self.pad_depthwise2(x)
        x = self.depthwise_conv2(x)
        x = self.pointwise_conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        x = self.padpool2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        x = torch.squeeze(x, 2)
        return x

class EEGNet(nn.Module):
    def __init__(self, input_shape, output_shape, kernel_size=32, F1=16, F2=256, D=4, dropout=0.5, nb_outlayer_channels=5):
        super(EEGNet, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.timesamples = input_shape[0]
        self.channels = input_shape[1]
        self.nb_outlayer_channels = nb_outlayer_channels

        self.block_1 = Block_1(kernel_size, F1, D, self.channels, dropout)
        self.block_2 = Block_2(nb_outlayer_channels, F1, D, F2, dropout)
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
        )

    def forward(self, x):
        """
        Implements a forward pass of the eegnet.
        """
        x = x[:, :, 0, :]
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.output_layer(x)
        x = self.out_linear(x)
        return x[:, :, 0]

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
        return self.nb_outlayer_channels  # from depthwise conv 2

if __name__ == "__main__":
    batch, chan, time = 4, 3, Fs * 30
    out_chan, out_width = 1, 1
    model = EEGNet(input_shape=(time, chan), output_shape=(out_chan, out_width))
    inputs = torch.randn(batch, chan, 1, time)
    print('input:', inputs.size())
    out = model(inputs)
    print('output:', out.size())

    from thop import profile

    flops, params = profile(model, inputs=(inputs,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
