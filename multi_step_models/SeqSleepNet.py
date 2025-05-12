import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from multi_step_models.utils import *
Fs = 100

class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, device):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.device = device

    def forward(self, x):
        h0 = torch.randn(self.num_layers*2, x.size(0), self.hidden_size)#.cuda()
        if self.device == 'cuda':
            h0 = h0.cuda()
        out, _ = self.gru(x, h0)
        return out

class Parabit(nn.Module):
    def __init__(self, seq_len, dim, class_num):
        super(Parabit, self).__init__()
        self.bits = []
        self.seq_len = seq_len
        for i in range(seq_len):
            bit = nn.Linear(dim, class_num)
            bit_name = 'seq_at_%d' % i
            setattr(self, bit_name, bit)
            self.bits.append(getattr(self, bit_name))

    def forward(self, x):
        bit_fcs = []
        for i in range(self.seq_len):
            xx = x[:,i,:]
            fc = self.bits[i]
            yy = fc(xx)
            yy = yy.unsqueeze(1)
            bit_fcs.append(yy)
        torch_bits = torch.cat(bit_fcs, 1) # bs, seq_len, class_num
        return torch_bits

class SeqSleepNet(nn.Module):
    def __init__(self, ch_num, device='cuda', filterbanks=torch.from_numpy(lin_tri_filter_shape(32, 256, 100, 0, 50)).to(torch.float), seq_len=128, class_num=5):
        super(SeqSleepNet, self).__init__()
        self.seq_len = seq_len
        self.ch_num = ch_num
        self.class_num = class_num
        self.filterbanks = filterbanks
        self.device = device
        if self.device == 'cuda':
            self.filterbanks = filterbanks.cuda()

        self.filterweight = Parameter(torch.randn(ch_num, 129, 32))
        self.epoch_rnn = BiGRU(32, 64, 1, self.device)
        if self.device == 'cuda':
            self.epoch_rnn = self.epoch_rnn.cuda()
        self.attweight_w  = Parameter(torch.randn(128, 64))
        self.attweight_b  = Parameter(torch.randn(64))
        self.attweight_u  = Parameter(torch.randn(64))

        self.seq_rnn      = BiGRU(64*2, 64, 1, self.device)
        self.cls          = Parabit(self.seq_len, 64*2, self.class_num)

    def forward(self, x):
        # x     : [bs, seq_len, 29, 129]
        # return: [bs, seq_len, class_num]

        # torch.mul -> element-wise dot;  torch.matmul -> matrix multiplication
        x = x.permute(0, 2, 1, 3)
        x = preprocessing(x, self.device)
        filterbank = torch.mul(self.filterweight, self.filterbanks)
        tmp = []
        for i in range(self.ch_num):
            xx = x[:,:,i,:,:]
            xx = torch.reshape(xx, [-1,129])
            fb = filterbank[i,:,:]
            c = torch.matmul(xx,fb)
            c = c.unsqueeze(0)
            tmp.append(c)
        x = torch.cat(tmp)
        x = x.mean(0)
        x = torch.reshape(x, [-1, 29, 32])
        x = self.epoch_rnn(x)

        v = torch.tanh(torch.matmul(torch.reshape(x, [-1, 128]), self.attweight_w) + torch.reshape(self.attweight_b, [1, -1]))
        vu= torch.matmul(v, torch.reshape(self.attweight_u, [-1, 1]))
        exps = torch.reshape(torch.exp(vu), [-1, 29])
        x = torch.sum(torch.mul(x, torch.reshape(exps, [-1, 29, 1])), 1)

        x = torch.reshape(x, [-1, self.seq_len, 64*2])
        x = self.seq_rnn(x)
        x = self.cls(x)
        x = x.permute(0, 2, 1)
        return x

if __name__ == '__main__':
    batch, chan, times, steps = 4, 3, Fs * 30, 128
    out_chan, out_width = 1, 1
    class_num = 5

    model = SeqSleepNet(ch_num=chan, device='cuda')

    model = model.cuda()
    inputs = torch.rand(batch, chan, steps, times)
    print('preprocessing inputs:', inputs.shape)
    inputs = inputs.cuda()
    outputs = model(inputs)
    print('outputs:', outputs.size())

    from thop import profile

    flops, params = profile(model, inputs=(inputs,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')