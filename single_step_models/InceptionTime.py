import torch
import torch.nn as nn
from single_step_models.model_utils import Pad_Conv, Pad_Pool

Fs = 100

class Inception_module(nn.Module):
    """
    This class implements one inception module descirbed above as torch.nn.Module, which can then be
    stacked into a model by ConvNet
    """

    def __init__(self, kernel_size, nb_features, nb_channels, nb_filters, bottleneck_size, depth):
        super().__init__()
        kernel_size_s = [kernel_size // (2**i) for i in range(3)]
        # Define all the layers and modules we need in the forward pass: first the initial
        # convolution and the parallel maxpooling
        self.pad_conv_in = Pad_Conv(kernel_size=kernel_size)
        # This is the bottleneck convolution
        self.conv_in = nn.Conv1d(
            in_channels=nb_channels if depth == 0 else nb_features,
            out_channels=bottleneck_size,
            kernel_size=kernel_size,
            bias=False,
        )
        self.pad_pool_in = Pad_Pool(left=1, right=1)
        self.maxpool_in = nn.MaxPool1d(kernel_size=3, stride=1)
        # 3 parallel convolutions taking the bottleneck as input
        self.conv1 = nn.Conv1d(
            in_channels=bottleneck_size,
            out_channels=nb_filters,
            kernel_size=kernel_size_s[0],
            bias=False,
        )
        self.pad1 = Pad_Conv(kernel_size=kernel_size_s[0])
        self.conv2 = nn.Conv1d(
            in_channels=bottleneck_size,
            out_channels=nb_filters,
            kernel_size=kernel_size_s[1],
            bias=False,
        )
        self.pad2 = Pad_Conv(kernel_size=kernel_size_s[1])
        self.conv3 = nn.Conv1d(
            in_channels=bottleneck_size,
            out_channels=nb_filters,
            kernel_size=kernel_size_s[2],
            bias=False,
        )
        self.pad3 = Pad_Conv(kernel_size=kernel_size_s[2])
        # and the 4th parallel convolution following the maxpooling, no padding needed since 1x1
        # convolution
        self.conv4 = nn.Conv1d(
            in_channels=nb_channels if depth == 0 else nb_features,
            out_channels=nb_filters,
            kernel_size=1,
            bias=False,
        )
        self.batchnorm = nn.BatchNorm1d(num_features=nb_features)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Implements a forward pass through one inception module
        """
        # Implement the left convolution
        x_left = self.pad_conv_in(x)
        x_left = self.conv_in(x_left)
        # Implement the 3 parallel convolutions afterwards
        x_left1 = self.pad1(x_left)
        x_left1 = self.conv1(x_left1)
        x_left2 = self.pad2(x_left)
        x_left2 = self.conv2(x_left2)
        x_left3 = self.pad1(x_left)
        x_left3 = self.conv1(x_left3)
        # Implement the right maxpooling followed by a conv
        x_right = self.pad_pool_in(x)
        x_right = self.maxpool_in(x_right)
        x_right = self.conv4(x_right)
        # Concatenate the 4 outputs
        x = torch.cat(
            tensors=(x_left1, x_left2, x_left3, x_right), dim=1
        )  # concatenate along the feature dimension
        x = self.batchnorm(x)
        return self.activation(x)

class InceptionTime(nn.Module):
    def __init__(self, input_shape, output_shape, kernel_size=64, nb_filters=16, depth=12, bottleneck_size=16, nb_outlayer_channels=5):
        super(InceptionTime, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.timesamples = self.input_shape[0]
        self.nb_channels = self.input_shape[1]

        self.kernel_size = kernel_size
        self.depth =  depth
        self.nb_filters = nb_filters
        self.nb_outlayer_channels = nb_outlayer_channels

        self.bottleneck_size = bottleneck_size
        self.nb_features = 4 * nb_filters

        self.conv_blocks = nn.ModuleList([self._module(d) for d in range(self.depth)])
        self.gap_layer = nn.AvgPool1d(kernel_size=2, stride=1)
        self.gap_layer_pad = Pad_Pool(left=0, right=1, value=0)


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
        x = x[:, :, 0, :]
        for d in range(self.depth):
            x = self.conv_blocks[d](x)
        x = self.gap_layer_pad(x)
        x = self.gap_layer(x)
        output = self.output_layer(x) # Defined in BaseNet
        output = self.out_linear(output)
        return output[:, :, 0]

    def _module(self, depth):
        return Inception_module(
            self.kernel_size,
            self.nb_features,
            self.nb_channels,
            self.nb_filters,
            self.bottleneck_size,
            depth,
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

if __name__ == "__main__":
    batch, chan, time = 4, 3, Fs * 30
    out_chan, out_width = 1, 1
    model = InceptionTime(input_shape=(time, chan), output_shape=(out_chan, out_width))
    inputs = torch.randn(batch, chan, 1, time)
    print('inputs:', inputs.size())
    out = model(inputs)
    print('output:', out.size())

    from thop import profile

    flops, params = profile(model, inputs=(inputs,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')