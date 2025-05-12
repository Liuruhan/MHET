import torch
import torch.nn as nn
from collections import OrderedDict


class TinySleepNet(nn.Module):
    def __init__(self, sample_rate=100, channel=3, seq_length=1, n_rnn_units=128):
        super(TinySleepNet, self).__init__()
        self.seq_length = seq_length
        self.n_rnn_units = n_rnn_units
        self.padding_edf = {  # same padding in tensorflow
            'conv1': (22, 22),
            'max_pool1': (2, 2),
            'conv2': (3, 4),
            'max_pool2': (0, 1),
        }
        # self.config = config
        first_filter_size = int(sample_rate / 2.0)  # 100/2 = 50, 与以往使用的Resnet相比，这里的卷积核更大
        first_filter_stride = int(sample_rate / 16.0)  # todo 与论文不同: 16，论文给出的stride是100/4=25
        # n_rnn_units = 128

        self.cnn = nn.Sequential(
            nn.ConstantPad1d(self.padding_edf['conv1'], 0),  # conv1
            nn.Sequential(OrderedDict([
                ('conv1', nn.Conv1d(in_channels=channel, out_channels=128, kernel_size=first_filter_size, stride=first_filter_stride,
                      bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool1'], 0),  # max p 1
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=0.5),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv2
            nn.Sequential(OrderedDict([
                ('conv2',
                 nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv3
            nn.Sequential(OrderedDict([
                ('conv3',nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),

            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['conv2'], 0),  # conv4
            nn.Sequential(OrderedDict([
                ('conv4', nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False))
            ])),
            # nn.BatchNorm1d(128),
            # nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.99),
            nn.BatchNorm1d(num_features=128, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.ConstantPad1d(self.padding_edf['max_pool2'], 0),  # max p 2
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Flatten(),
            nn.Dropout(p=0.5),
        )
        # self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1, dropout=0.5)
        # self.rnn = nn.LSTM(input_size=2048, hidden_size=self.config['n_rnn_units'], num_layers=1)

        self.rnn = nn.LSTM(input_size=2048, hidden_size=n_rnn_units, num_layers=1, batch_first=True)
        self.rnn_dropout = nn.Dropout(p=0.5)  # todo 是否需要这个dropout?
        self.fc = nn.Linear(n_rnn_units, 5)

    def forward(self, x):
        x = x[:, :, 0, :]
        x = self.cnn(x)
        # input of LSTM must be shape(seq_len, batch, input_size)
        # x = x.view(self.config['seq_length'], self.config['batch_size'], -1)
        x = x.view(-1, self.seq_length, 2048)  # batch first == True
        assert x.shape[-1] == 2048
        x, _ = self.rnn(x)
        # x = x.view(-1, self.config['n_rnn_units'])
        x = x.reshape(-1, self.n_rnn_units)
        # rnn output shape(seq_length, batch_size, hidden_size)
        x = self.rnn_dropout(x)
        x = self.fc(x)

        return x



if __name__ == '__main__':
    Fs = 100
    batch, chan, time = 4, 3, Fs * 30
    inputs = torch.randn(batch, chan, 1, time)
    print('inputs:', inputs.size())
    model = TinySleepNet(sample_rate=Fs, channel=chan)

    output = model(inputs)
    print("output:", output.size())

    from thop import profile

    flops, params = profile(model, inputs=(inputs,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
