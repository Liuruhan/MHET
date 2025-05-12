import torch
import torch.nn as nn
from single_step_models.model_utils import TCSConv1d, Pad_Pool, Pad_Conv

Fs = 100

class Xception(nn.Module):
    def __init__(self, input_shape, output_shape, kernel_size=40, nb_filters=64, depth=6, use_residual=True, nb_outlayer_channels=5):
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
        Implements the forward pass of the network
        Modules defined in a class implementing ConvNet are stacked and shortcut connections are
        used if specified.
        """
        x = x[:, :, 0, :]
        input_res = x 
        shortcut_cnt = 0
        for d in range(self.depth):
            x = self.conv_blocks[d](x)
            if self.use_residual and d % 3 == 2:
                res = self.shortcuts[shortcut_cnt](input_res)
                shortcut_cnt += 1
                x = torch.add(x, res)
                x = nn.functional.relu(x)
                input_res = x
        x = self.gap_layer_pad(x)
        x = self.gap_layer(x)
        output = self.output_layer(x) 
        output = self.out_linear(output)
        return output[:, :, 0]

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

if __name__ == "__main__":
    batch, chan, time = 4, 3, Fs * 30
    out_chan, out_width = 1, 1
    model = Xception(input_shape=(time, chan), output_shape=(out_chan, out_width))
    inputs = torch.randn(batch, chan, 1, time)
    print('input:', inputs.size())
    out = model(inputs)
    print('output:', out.size())

    from thop import profile

    flops, params = profile(model, inputs=(inputs,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')