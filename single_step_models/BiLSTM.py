import torch
from torch import nn
from single_step_models.model_utils import Pad_Pool

Fs = 100

class BiLSTM(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size=129, nb_outlayer_channels=5, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.timesamples = input_shape[0]
        self.input_channels = input_shape[1]
        self.nb_outlayer_channels = nb_outlayer_channels
        self.output_shape = output_shape

        self.lstm = nn.LSTM(
            input_size=self.input_channels,
            hidden_size=self.hidden_size,
            dropout=dropout,
            num_layers=5,
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

    def forward(self, x, hidden=None):
        """
        Implements the forward pass of the network
        Modules defined in a class implementing ConvNet are stacked and shortcut connections are
                used if specified.
        """
        x = x[:, :, 0, :]
        x = x.permute(2, 0, 1)
        output, (hn, cn) = self.lstm(x, hidden)
        output = output.permute(1, 2, 0)
        output = self.output_layer(output)
        output = self.out_linear(output)
        output = output.permute(0, 2, 1)
        return output[:, 0, :]

    def get_nb_channels_output_layer(self):
        """
        Return the number of features/channels of the tensor before the output layer
        """
        return 2 * self.hidden_size

    def get_nb_features_output_layer(self):
        """
        Return number of features passed into the output layer of the network
        nb.features has to be defined in a model implementing ConvNet
        """
        return 2 * self.hidden_size * self.timesamples



if __name__ == "__main__":
    batch, chan, time = 4, 3, Fs * 30
    out_chan, out_width = 1, 1
    model = BiLSTM(input_shape=(time, chan), output_shape=(out_chan, out_width))
    input = torch.randn(batch, chan, 1, time)
    print('input:', input.size())
    out = model(input)
    print('output:', out.size())

    from thop import profile

    flops, params = profile(model, inputs=(input,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
