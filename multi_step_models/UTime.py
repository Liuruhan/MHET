import torch
import torch.nn as nn
Fs = 100

class Utime(nn.Module):
    def __init__(self, ch=1):
        super(Utime, self).__init__()
        self.enc_1 = nn.Sequential(
            nn.Conv1d(ch, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.enc_2 = nn.Sequential(
            nn.MaxPool1d(10),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.enc_3 = nn.Sequential(
            nn.MaxPool1d(8),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.enc_4 = nn.Sequential(
            nn.MaxPool1d(6),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.enc_5 = nn.Sequential(
            nn.MaxPool1d(4),
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Conv1d(256, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        self.decoder_b1 = nn.Sequential(
            nn.Upsample(216),
            nn.Conv1d(256, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.decoder_b2 = nn.Sequential(
            nn.Conv1d(256, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Conv1d(128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(1296),
            nn.Conv1d(128, 64, 6, padding=2), # careful when filter size is even
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder_b3 = nn.Sequential(
            nn.Conv1d(128, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Upsample(10368),
            nn.Conv1d(64, 32, 8, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.decoder_b4 = nn.Sequential(
            nn.Conv1d(64, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Upsample(103680),
            nn.Conv1d(32, 16, 10, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.decoder_b5 = nn.Sequential(
            nn.Conv1d(32, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.Conv1d(16, 16, 5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )

        self.segment_classifier = nn.Sequential(
            nn.Conv1d(16, 5, 1),
            nn.BatchNorm1d(5),
            nn.Tanh(),

            nn.ConstantPad1d(660, 0), # 103680 + 660*2 = 105000
        )

        self.final_conv = nn.Sequential(
            nn.Conv1d(5, 5, 1),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        b, c, s, t = x.size()
        x = x.reshape(b, c, s*t)

        enc_1 = self.enc_1(x)
        enc_2 = self.enc_2(enc_1)
        enc_3 = self.enc_3(enc_2)
        enc_4 = self.enc_4(enc_3)
        enc_5 = self.enc_5(enc_4)

        x = self.decoder_b1(enc_5)
        x = crop_conc(x, enc_4) # the second one is larger - 1296

        x = self.decoder_b2(x)

        x = manually_pad(x, 1296)
        x = crop_conc(x, enc_3)

        x = self.decoder_b3(x)

        x = manually_pad(x, 10368)
        x = crop_conc(x, enc_2)

        x = self.decoder_b4(x)

        x = manually_pad(x, 103680)
        x = crop_conc(x, enc_1)

        x = self.decoder_b5(x) # expected BS, 16, 103680
        x = self.segment_classifier(x)

        x = x.view(-1, 5, 3000, 35)

        x = torch.mean(x, dim=2)
        x = self.final_conv(x)

        return x

def crop_conc(x1, x2):
    # crop x1 to comply with x2's dim and concatenate them
    # skip-connection implementation
    crop_x2 = x2[:,:,:x1.size()[2]]
    return torch.cat((x1, crop_x2), 1)


def manually_pad(x, dim):
    # e.g. x's dim is [bs, 128, 1295] required dim = 1296
    tmp = torch.zeros(x.size()[0], x.size()[1], dim, device=x.device)
    tmp[:, :, :x.size()[2]] = x[:, :, :x.size()[2]]
    return tmp

if __name__ == '__main__':
    batch, chan, time, steps = 4, 3, Fs * 30, 35
    out_chan, out_width = 1, 1
    model = Utime(ch=chan)
    inputs = torch.randn(batch, chan, steps, time)
    print('input:', inputs.size())
    output = model(inputs)
    print('outputs:', output.size())

    from thop import profile

    flops, params = profile(model, inputs=(inputs,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    