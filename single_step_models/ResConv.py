import torch
from torch import nn
from single_step_models.model_utils import Pad_Pool, Pad_Conv

Fs = 100

class ResConv(nn.Module):
    def __init__(self, input_shape, output_shape, nb_filters=16, depth=10, nb_outlayer_channels=5, kernel_size=64, use_residual=True, preprocessing=False):
        super(ResConv, self).__init__()
        self.nb_features = nb_filters
        self.nb_outlayer_channels = nb_outlayer_channels

        self.use_residual = use_residual
        self.depth = depth
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.timesamples = self.input_shape[0]
        self.nb_channels = self.input_shape[1]
        self.preprocessing = preprocessing

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
        x = x[:, :, 0, :]
        if self.preprocessing:
            x = self._preprocessing(x)
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
        return nn.Sequential(
            Pad_Conv(kernel_size=self.kernel_size, value=0),
            nn.Conv1d(
                in_channels=self.nb_channels if depth == 0 else self.nb_features,
                out_channels=self.nb_features,
                kernel_size=self.kernel_size,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=self.nb_features),
            nn.ReLU(),
            Pad_Pool(left=0, right=1, value=0),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )

    def _shortcut(self, depth):
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
    model = ResConv(input_shape=(time, chan), output_shape=(out_chan, out_width))
    input = torch.randn(batch, chan, 1, time)
    print('input:', input.size())
    out = model(input)
    print('output:', out.size())

    from thop import profile

    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')

