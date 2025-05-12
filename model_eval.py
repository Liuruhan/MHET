import os
import re
import argparse
import torch
import torch.nn as nn
from torchnet import meter
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import classification_report

from single_step_models.BiLSTM import BiLSTM
from single_step_models.EEGNet import EEGNet
from single_step_models.DeepSleepNet import DeepSleepNet
from single_step_models.TinySleepNet import TinySleepNet
from single_step_models.ResConv import ResConv
from single_step_models.InceptionTime import InceptionTime
from single_step_models.Xception import Xception
from single_step_models.ConvLSTM import ConvLSTM

from multi_step_models.UTime import Utime
from multi_step_models.SalientSleepNet import SalientSleepNet
from multi_step_models.SeqSleepNet import SeqSleepNet

from LMHT.LMHT import LMHT

from sleep_dataset import BasicDataset
from dataloaderX import DataPrefetcher, DataLoaderX

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='B',
                        help='input ba tch size for training (default : 64)', dest='batch_size')
    parser.add_argument('-tb', '--test-batch-size', type=int, default=16, metavar='TB',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('-nclasses', '--nclasses', type=int, default=5, metavar='nclasses',
                        help='the class of sleep stages (W, N1, N2, N3, R)')
    parser.add_argument('-c', '--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-d', '--device', type=str, default='0,1,2', metavar='device',
                        help='CUDA device ID')
    parser.add_argument('-ml', '--model-list', type=list, default=['lmht'], metavar='model_list',
                        help='training model list')
    parser.add_argument('-mm', '--multi-model', type=bool, default=False, metavar='mm',
                        help='is the multi-step model?')
    parser.add_argument('-ms', '--multi-step', type=int, default=128, metavar='ms',
                        help='the steps of multi-step model')
    parser.add_argument('-valdir', '--valdir', type=str, default='./sleepEDF/test/', metavar='valdir',
                        help='the path of validation dataset')
    parser.add_argument('-dir', '--cp-dir', type=str, default='./model_result/checkpoints/lmhtCP_epoch3.pth', metavar='cp_dir',
                        help='the path of saving models')
    return parser.parse_args()

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)

def val_net(args):
    val = BasicDataset(args.valdir, args.nclasses)
    val_loader = DataLoaderX(val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    test_files = np.array(os.listdir(args.valdir))
    test_files = np.expand_dims(test_files, axis=1)

    if args.model_name == 'bilstm':
        model = BiLSTM(input_shape=(3000, 3), output_shape=(1, 1))
    elif args.model_name == 'eegnet':
        model = EEGNet(input_shape=(3000, 3), output_shape=(1, 1))
    elif args.model_name == 'deepsleepnet':
        model = DeepSleepNet(ch=3)
    elif args.model_name == 'tinysleepnet':
        model = TinySleepNet(sample_rate=100, channel=3)
    elif args.model_name == 'resconv':
        model = ResConv(input_shape=(3000, 3), output_shape=(1, 1))
    elif args.model_name == 'inceptiontime':
        model = InceptionTime(input_shape=(3000, 3), output_shape=(1, 1))
    elif args.model_name == 'xception':
        model = Xception(input_shape=(3000, 3), output_shape=(1, 1))
    elif args.model_name == 'convlstm':
        model = ConvLSTM(input_shape=(3000, 3), output_shape=(1, 1))
    # multiple models
    elif args.model_name == 'salientsleepnet':
        model = SalientSleepNet(input_shape=(3000, 20), output_shape=(1, 20))
    elif args.model_name == 'utime':
        model = Utime(ch=3)
    elif args.model_name == 'seqsleepnet':
        model = SeqSleepNet(ch_num=2, device='cuda')

    # our model
    elif args.model_name == 'lmht':
        model = LMHT(input_shape=(3000, 3), output_shape=(1, 1))

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()

    model.load_state_dict(torch.load(args.cp_dir))
    model.eval()
    test_loss = 0
    confusion_matrix = meter.ConfusionMeter(args.nclasses)
    out_np = np.zeros((args.batch_size*len(val_loader), args.nclasses))
    tgt_np = np.zeros(args.batch_size*len(val_loader))
    criterion = nn.CrossEntropyLoss()

    prefetcher = DataPrefetcher(val_loader)
    batch = prefetcher.next()
    iter_id = 0
    while batch is not None:
        data = batch['data']

        target = batch['target']
        time = batch['time']
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # if args.cuda:
        #     data, target, time = data.cuda(), target.cuda(), time.cuda()
        # data, target, time = Variable(data), Variable(target), Variable(time)

        # output = model(data)
        output = model(data, time, 'test')
        test_loss += criterion(output, target).item()

        confusion_matrix.add(output.data, target.data)
        out_np[iter_id*args.batch_size:iter_id*args.batch_size+output.size(0), :] = output.data.cpu().numpy()
        tgt_np[iter_id*args.batch_size:iter_id*args.batch_size+output.size(0)] = target.data.cpu().numpy()
        iter_id += 1
        batch = prefetcher.next()

    pred_np = np.argmax(out_np, axis=1)
    print('Test_loss:', test_loss / (len(val_loader)*args.batch_size))
    result = classification_report(tgt_np, pred_np, target_names=['W', 'N1', 'N2', 'N3', 'R'], digits=4)
    print(result)
    print('kappa', kappa(confusion_matrix.value()))

    tgt_np = np.expand_dims(tgt_np, axis=1)
    length = test_files.shape[0]
    test_files = np.concatenate((test_files, tgt_np[:length, :], out_np[:length, :]), axis=1)
    np.savetxt('NAME_tgt_out_LMHT'+args.set+'.csv', test_files, fmt='%s', delimiter=',')
    return

if __name__ == "__main__":
    args = get_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print('cuda available:', args.cuda)

    if re.findall('train', args.valdir) != None:
        args.set = 'train'
    elif re.findall('test', args.valdir) != None:
        args.set = 'test'
    else:
        print('error!')

    for i in range(len(args.model_list)):
        args.model_name = args.model_list[i]
        val_net(args)


