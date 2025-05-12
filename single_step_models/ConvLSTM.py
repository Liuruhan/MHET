import torch
import torch.nn as nn

from single_step_models.model_utils import Pad_Pool, Pad_Conv

Fs = 100

class ConvLSTM(nn.Module):
    def __init__(self, input_shape, output_shape, nb_filters=16, nb_outlayer_channels=5, depth=12, kernel_size=64, dropout=0.5, use_residual=True, preprocessing=False):
        super(ConvLSTM, self).__init__()

        self.nb_features = nb_filters  # For CNN simply the number of filters
        self.hidden_size = self.nb_features  # LSTM processes CNN output channels
        self.nb_outlayer_channels = nb_outlayer_channels

        self.input_shape = input_shape
        self.output_shape = output_shape

        self.timesamples = input_shape[0]
        self.nb_channels = input_shape[1]
        self.use_residual = use_residual
        self.depth = depth
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.conv_blocks = nn.ModuleList([self._module(d) for d in range(self.depth)])
        if self.use_residual:
            self.shortcuts = nn.ModuleList([self._shortcut(d) for d in range(int(self.depth / 3))])
        self.gap_layer = nn.AvgPool1d(kernel_size=2, stride=1)
        self.gap_layer_pad = Pad_Pool(left=0, right=1, value=0)

        self.scale_temporal = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=3, kernel_size=5100, stride=500),
            nn.BatchNorm1d(num_features=3),
            nn.ReLU(),
            Pad_Pool(left=2, right=2, value=0),
            nn.MaxPool1d(kernel_size=5, stride=1),
        )
        self.scale_temporal = nn.Linear(60000, input_shape[0])

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            dropout=dropout,
            num_layers=3,
            bidirectional=True,
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
        input_res = x  # set for the residual shortcut connection

        # Stack the CNN modules and residual connection
        shortcut_cnt = 0
        for d in range(self.depth):
            x = self.conv_blocks[d](x)
            if self.use_residual and d % 3 == 2:
                res = self.shortcuts[shortcut_cnt](input_res)
                shortcut_cnt += 1
                x = torch.add(x, res)
                x = nn.functional.relu(x)
                input_res = x
        # LSTM PART
        x = self.gap_layer_pad(x)
        x = self.gap_layer(x)
        x = x.permute(2, 0, 1)
        output, (hn, cn) = self.lstm(x)
        output = output.permute(1, 2, 0)
        output = self.output_layer(output)
        output = self.out_linear(output)
        # output = output.permute(0, 2, 1)
        # print(f'ConvLstm output shape {output.size()}')
        return output[:, :, 0]

    def _shortcut(self, depth):
        """
        Implements a shortcut with a convolution and batch norm
        This is the same for all models implementing ConvNet, therefore defined here
        Padding before convolution for constant tensor shape, similar to tensorflow.keras
                padding=same
        """
        return nn.Sequential(
            Pad_Conv(kernel_size=self.kernel_size, value=0),
            nn.Conv1d(
                in_channels=self.nb_channels if depth == 0 else self.nb_features,
                out_channels=self.nb_features,
                kernel_size=self.kernel_size,
            ),
            nn.BatchNorm1d(num_features=self.nb_features),
        )

    def get_nb_channels_output_layer(self):
        return 2 * self.hidden_size

    def get_nb_features_output_layer(self):
        """
        Return number of features passed into the output layer of the network
        nb.features has to be defined in a model implementing ConvNet
        """
        return 2 * self.hidden_size * self.timesamples

    def _module(self, depth):
        """
        The module of CNN is made of a simple convolution with batch normalization and ReLu
                activation. Finally, MaxPooling is used. We use two custom padding modules such that
                keras-like padding='same' is achieved, i.e. tensor shape stays constant when passed through
                the module.
        """
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
            Pad_Pool(left=2, right=2, value=0),
            nn.MaxPool1d(kernel_size=5, stride=1),
        )

if __name__ == "__main__":
    batch, chan, time = 4, 3, Fs * 30
    out_chan, out_width = 1, 1
    model = ConvLSTM(input_shape=(time, chan), output_shape=(out_chan, out_width))
    input = torch.randn(batch, chan, 1, time)
    print('input:', input.size())

    out = model(input)
    print('output:', out.size())

    from thop import profile

    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')