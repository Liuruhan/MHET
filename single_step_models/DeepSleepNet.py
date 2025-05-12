import torch
import torch.nn as nn

Fs = 100

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()

        out, _ = self.lstm(x, (h0, c0))
        return out

class DeepSleepNet(nn.Module):
    def __init__(self, ch = 1):
        super(DeepSleepNet, self).__init__()
        self.features_s = nn.Sequential(
            nn.Conv1d(ch, 64, 50, 6),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(),
            nn.Conv1d(64, 128, 6),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 6),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 6),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.features_l = nn.Sequential(
            nn.Conv1d(ch, 64, 400, 50),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(),
            nn.Conv1d(64, 128, 8),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.features_seq = nn.Sequential(
            BiLSTM(7168, 512, 2),
        )
        self.res = nn.Linear(7168, 1024)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, 5),
        )

    def forward(self, x):
        x = x[:, :, 0, :]
        x_s = self.features_s(x)
        x_l = self.features_l(x)
        x_s = x_s.flatten(1, 2)
        x_l = x_l.flatten(1, 2)
        x = torch.cat((x_s, x_l), 1)
        x_seq = x.unsqueeze(1)
        x_blstm = self.features_seq(x_seq)
        x_blstm = torch.squeeze(x_blstm, 1)
        x_res = self.res(x)
        x = torch.mul(x_res, x_blstm)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    ch_num = 3
    batch_size = 4
    model = DeepSleepNet(ch=ch_num)
    model = model.cuda()
    inputs = torch.rand(batch_size, ch_num, 1, int(Fs*30))
    inputs = inputs.cuda()
    outputs = model(inputs)
    print('input size:', inputs.size(), 'output size:', outputs.size())

    from thop import profile

    flops, params = profile(model, inputs=(inputs,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


