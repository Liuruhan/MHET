import argparse
import torch
import os
import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report
from itertools import cycle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchnet import meter

# models
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

from MHFN.MFHN import MHFN

from sleep_dataset import BasicDataset
from dataloaderX import DataPrefetcher, DataLoaderX

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='B',
                        help='input ba tch size for training (default : 64)', dest='batch_size')
    parser.add_argument('-tb', '--test-batch-size', type=int, default=16, metavar='TB',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('-e', '--epochs', type=int, default=25, metavar='E',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('-n', '--nclasses', type=int, default=5, metavar='N',
                        help='number of sleep stages')
    parser.add_argument('-l', '--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('-m', '--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('-c', '--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-d', '--device', type=str, default='0,1,2', metavar='device',
                        help='CUDA device ID')
    parser.add_argument('-ml', '--model-list', type=list, default=['mhfn'], metavar='model_list',
                        help='training model list')
    parser.add_argument('-log', '--log-interval', type=int, default=400, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-p', '--train-prob', type=float, default=0.8, metavar='train_prob',
                        help='the probability of training used')
    parser.add_argument('-cp', '--save-cp', type=bool, default=True, metavar='save_cp',
                        help='is save the model?')
    parser.add_argument('-mm', '--multi-model', type=bool, default=False, metavar='mm',
                        help='is the multi-step model?')
    parser.add_argument('-ms', '--multi-step', type=int, default=128, metavar='ms',
                        help='the steps of multi-step model')
    parser.add_argument('-trdir', '--trdir', type=str, default='./sleepEDF/train/', metavar='trdir',
                        help='the path of saving models')
    parser.add_argument('-tedir', '--tedir', type=str, default='./sleepEDF/test/', metavar='tedir',
                        help='the path of saving models')
    parser.add_argument('-valdir', '--valdir', type=str, default='./sleepEDF/val/', metavar='valdir',
                        help='the path of saving models')
    parser.add_argument('-dir', '--cp-dir', type=str, default='./model_result/checkpoints/', metavar='cp_dir',
                        help='the path of saving models')
    parser.add_argument('-result', '--rd', type=str, default='./model_result/', metavar='rd',
                        help='the path of saving vis figures for models')
    return parser.parse_args()

def plot_AUC(pred, target, n_classes, epoch, path, din_color):
    n_target = np.eye(n_classes)[target.astype(int)]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(n_target[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(n_target.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot all ROC curves
    lw = 2

    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.4f})'
                   ''.format(roc_auc["micro"]),
             color='chocolate', linestyle=':', linewidth=3)
    plt.legend(loc="lower right")


    if din_color == plt.cm.Blues:
        colors = cycle(['lightskyblue', 'dodgerblue', 'royalblue', 'midnightblue', 'slategrey'])
    elif din_color  == plt.cm.Reds:
        colors = cycle(['mistyrose', 'lightcoral', 'firebrick', 'maroon', 'rosybrown'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.4f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    if din_color == plt.cm.Blues:
        plt.savefig(path + '/' + str(epoch)+'_test_AUC.jpg')
    elif din_color == plt.cm.Reds:
        plt.savefig(path + '/' + str(epoch) + '_extraVal_AUC.jpg')
    plt.close()
    return

def plot_Matrix(cm, classes, epoch, path, title=None, cmap=plt.cm.Blues):
    plt.rc('font', family='Times New Roman', size='8')
    print(cm)

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) == 0:
                cm[i, j]=0

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predicted')

    ax.set_xticks(np.arange(cm.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '0.4f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if int(cm[i, j] * 100 + 0.5) > 0:
                ax.text(j, i, format(float(cm[i, j] * 100 + 0.5), fmt) + '%',
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    if cmap == plt.cm.Blues:
        plt.savefig(path + '/' + str(epoch) + 'test_cm.jpg', dpi=300)
    elif cmap == plt.cm.Reds:
        plt.savefig(path + '/' + str(epoch) + 'extraVal_cm.jpg', dpi=300)
    plt.close()
    return

def testdata_cal(model, args, loader, png_flag, epoch, path_str, n_classes, color):
    model.eval()
    test_loss = 0
    confusion_matrix = meter.ConfusionMeter(n_classes)
    if args.multi_model == False:
        out_np = np.zeros((len(loader)*args.batch_size, n_classes))
        tgt_np = np.zeros(len(loader)*args.batch_size)
    else:
        out_np = np.zeros((len(loader) * args.batch_size * args.multi_step, n_classes))
        tgt_np = np.zeros(len(loader) * args.batch_size * args.multi_step)
    data_idx = 0

    criterion = nn.CrossEntropyLoss()

    prefetcher = DataPrefetcher(loader)
    batch = prefetcher.next()
    iter_id = 0
    while batch is not None:
        iter_id += 1
        data = batch['data']
        target = batch['target']
        time = batch['time']

        # if args.cuda:
        #     data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        if args.cuda:
            data, target, time = data.cuda(), target.cuda(), time.cuda()
        data, target, time = Variable(data), Variable(target), Variable(time)

        output = model(data, time, 'test')
        # output = model(data)
        test_loss += criterion(output, target).item()

        if args.multi_model:
            b, c, s = output.data.size()
            confusion_matrix.add(output.data.reshape(b * s, c), target.data.reshape(b * s))
            size = output.data.reshape(b * s, c).cpu().numpy().shape[0]
            out_np[data_idx:data_idx + size] = output.data.reshape(b * s, c).cpu().numpy()
            tgt_np[data_idx:data_idx + size] = target.data.reshape(b * s).cpu().numpy()
            data_idx += size
        else:
            confusion_matrix.add(output.data, target.data)
            size = output.data.cpu().numpy().shape[0]
            out_np[data_idx:data_idx+size] = output.data.cpu().numpy()
            tgt_np[data_idx:data_idx+size] = target.data.cpu().numpy()
            data_idx += size

        batch = prefetcher.next()
    
    pred_np = np.argmax(out_np, axis=1)
    print('Epoch:', epoch, 'Test_loss:', test_loss / len(loader))
    result = classification_report(tgt_np, pred_np, target_names=['W', 'N1', 'N2', 'N3', 'R'], digits=4)
    print(result)

    cm_value = confusion_matrix.value()
    if png_flag == True:
        if not os.path.exists(path_str):
            os.mkdir(path_str)
        plot_Matrix(cm_value, [1,2,3,4,5], epoch, path_str, title=None, cmap=color)
        plot_AUC(out_np, tgt_np, n_classes, epoch, path_str, color)
    return 

def train_net(args, test_loader):
# def train_net(model_name, n_classes, test_loader, extraVal_loader, cuda_device):
    # single models
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
    elif args.model_name == 'mhfn':
        model = MHFN(input_shape=(3000, 3), output_shape=(1, 1))

    if args.cuda:
        model = nn.DataParallel(model)
        model.cuda()
    if not os.path.exists(args.rd):
        os.mkdir(args.rd)
    if not os.path.exists(args.cp_dir):
        os.mkdir(args.cp_dir)

    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    # optimizer = optim.RMSprop(model.parameters(), lr=LR, alpha=0.9)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-12, momentum=0.95)
    # optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-12)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if n_classes > 1 else 'max', patience=2)
    # softmax = nn.Softmax()

    # weights = [0.424244352, 0.26854576, 0.045501815, 0.15071501, 0.110993063]
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    class_weights = torch.FloatTensor(weights).cuda()
    loss_fun = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(1, args.epochs + 1):
        confusion_matrix = meter.ConfusionMeter(args.nclasses)
        model.train()
        train_loss = 0
        prefetcher = DataPrefetcher(train_loader)
        batch = prefetcher.next()
        iter_id = 0
        while batch is not None:
            iter_id += 1
            data = batch['data']
            target = batch['target']
            time = batch['time']

            # if args.cuda:
            #     data, target = data.cuda(), target.cuda()
            # data, target = Variable(data), Variable(target)
            if args.cuda:
                data, target, time = data.cuda(), target.cuda(), time.cuda()
            data, target, time = Variable(data), Variable(target), Variable(time)

            optimizer.zero_grad()
            output = model(data, time, 'train')

            loss = loss_fun(output, target)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if args.multi_model:
                b, c, s = output.data.size()
                confusion_matrix.add(output.data.reshape(b*s, c), target.data.reshape(b*s))
            else:
                confusion_matrix.add(output.data, target.data)

            batch = prefetcher.next()

            # if (iter_id+1) % args.log_interval == 0:
            #     print("####################################################")
            #     print('Percentage:', (iter_id+1) / len(train_loader), 'Train_loss:', train_loss / iter_id)
            #     testdata_cal(model, args, test_loader, True, epoch*10000+iter_id, model_name, n_classes, plt.cm.Blues)
            #     testdata_cal(model, args, extraVal_loader, True, epoch*10000+iter_id, model_name, n_classes, plt.cm.Reds)

        print('Epoch:', epoch, 'Train_loss:', train_loss / len(train_loader))
        testdata_cal(model, args, test_loader, True, epoch, args.rd+args.model_name, args.nclasses, plt.cm.Blues)
        # testdata_cal(model, args, extraVal_loader, True, epoch, args.rd+model_name, n_classes, plt.cm.Reds)
        print("####################################################")

        # scheduler.step(final_acc)
        if epoch % 10 == 0: #epoch > 10 and
           for p in optimizer.param_groups:
               p['lr'] *= 0.95

        if args.save_cp:
            try:
                os.mkdir(args.cp_dir)
            except OSError:
                pass
            torch.save(model.state_dict(),
                       args.cp_dir + args.model_name + f'CP_epoch{epoch + 1}.pth')
            print(f'Checkpoint {epoch + 1} saved !')
    
    del model
    
    return

if __name__ == "__main__":
    args = get_args()
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cuda_device = torch.cuda.is_available()
    print('cuda available:', args.cuda)

    train = BasicDataset(args.trdir, args.nclasses)
    print('test')
    val = BasicDataset(args.tedir, args.nclasses)
    test = BasicDataset(args.valdir, args.nclasses)
    n_val = len(val)
    n_train = len(train)
    n_test = len(test)
    # print('samples:', n_train, n_val)
    print('samples:', n_train, n_val, n_test)

    train_loader = DataLoaderX(train, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoaderX(val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    extraVal_loader = DataLoaderX(test, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    for i in range(len(args.model_list)):
        print(args.model_list[i])
        args.model_name = args.model_list[i]
        train_net(args, test_loader)
        # train_net(model_list[i], n_classes, test_loader, extraVal_loader, cuda_device)


