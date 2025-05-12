import re
import h5py
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import logging

import torch
from torch.utils.data import Dataset

class BasicDataset(Dataset):
    def __init__(self, data_dir, n_classes):
        self.data_dir = data_dir
        self.n_classes = n_classes
        self.ids = [splitext(file)[0] for file in listdir(data_dir)
                    if not file.startswith('.')]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        #print(pil_img.size)
        img_nd = np.array(pil_img)
        #print(img_nd.shape)

        if len(img_nd.shape) == 2 :
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))

        #if img_trans.max() == 255:
        img_trans = img_trans / 255

        if img_trans.shape[0] == 4:
            img_return = img_trans[0:3, :, :]
        else:
            img_return = img_trans
        return img_return
    @classmethod
    def resampler(cls, ids):
        np.random.shuffle(ids)
        class_len = [0,0,0,0]
        new_ids = []
        new_idx = [[],[],[],[]]

        #print('id:', ids)
        for i in range(len(ids)):
            for j in range(4):
                if int(ids[i][:1]) == j:
                    class_len[j] += 1
        print(class_len)
        # rank = [len(new_idx[0]), len(new_idx[1]), len(new_idx[2]), len(new_idx[3])].sort()
        rank_idx = [index for index, value in sorted(list(enumerate(class_len)), key=lambda x:x[1])]

        for i in range(len(ids)):
            for j in range(4):
                if int(ids[i][:1]) == rank_idx[j]:
                    new_idx[j].append(ids[i])
        #print(len(new_idx[0]), len(new_idx[1]), len(new_idx[2]), len(new_idx[3]))

        for i in range(len(new_idx[3])):
            if i < len(new_idx[0]):
                for j in range(4):
                    new_ids.append(new_idx[j][i])
            elif i >= len(new_idx[0]) and i < len(new_idx[1]):
                for j in range(1):
                    new_ids.append(new_idx[j][np.random.randint(low=0, high=len(new_idx[j]) - 1)])
                for j in range(1, 4):
                    new_ids.append(new_idx[j][i])
            elif i >= len(new_idx[1]) and i < len(new_idx[2]):
                for j in range(2):
                    new_ids.append(new_idx[j][np.random.randint(low=0, high=len(new_idx[j]) - 1)])
                for j in range(2, 4):
                    new_ids.append(new_idx[j][i])
            elif i >= len(new_idx[2]) and i < len(new_idx[3]):
                for j in range(3):
                    new_ids.append(new_idx[j][np.random.randint(low=0, high=len(new_idx[j]) - 1)])
                for j in range(3, 4):
                    new_ids.append(new_idx[j][i])
        #print('self id:', new_ids)
        print(len(new_ids))
        return new_ids

    @classmethod
    def return_current_time(cls, current_t):
        if current_t[2] + 1 < 60:
            current_t[2] += 1
            return current_t
        else:
            if current_t[1] + 1 < 60:
                current_t[1] += 1
                current_t[2] = current_t[2] + 1 - 60
                return current_t
            else:
                if current_t[0] + 1 < 24:
                    current_t[0] += 1
                    current_t[1] = current_t[1] + 1 - 60
                    current_t[2] = current_t[2] + 1 - 60
                    return current_t
                else:
                    current_t[0] = current_t[0] + 1 - 24
                    current_t[1] = current_t[1] + 1 - 60
                    current_t[2] = current_t[2] + 1 - 60
                    return current_t
    
    def __getitem__(self, i):
        id = self.ids[i]
        data_file = self.data_dir + id + '.hdf5'
        with h5py.File(data_file, 'r') as h5:
            eeg = h5.get("EEG")[:]
            eog = h5.get("EOG")[:]
        single_mark = False
        if len(eog.shape) == 2:
            single_mark = True
            eog = np.expand_dims(eog, 2)
        if len(eeg.shape) == 2:
            single_mark = True
            eog = np.expand_dims(eeg, 2)
        epoch = np.concatenate((eeg, eog), axis=1)

        epoch = np.swapaxes(epoch, 0, 2)
        epoch = np.swapaxes(epoch, 0, 1)

        str_list = []
        last_idx = 0
        for j in re.finditer('_', id):
            if j:

                idx = j.span()[0]
                str_list.append(id[last_idx:idx])
                last_idx = j.span()[1]

        if single_mark:
            label = torch.tensor(int(str_list[3]))

            # label_one_hot = np.zeros(self.n_classes)
            # label_one_hot[label] = 1
        else:
            with h5py.File(data_file, 'r') as h5:
                label_np = h5.get("stages")[:]
            label = torch.tensor(label_np.astype(int))
        time_label = torch.Tensor([int(str_list[0]), int(str_list[1]), int(str_list[2])])

        # start_time = np.array([int(str_list[0]), int(str_list[1]), int(str_list[2])])
        # current_t = start_time
        # time_label_np = np.zeros((60, 3))
        # for i in range(30):
        #     for j in range(2):
        #         time_label_np[i * 2 + j, :] = current_t
        #     current_t = self.return_current_time(current_t)
        # time_label = torch.from_numpy(time_label_np)

        return {'data': torch.from_numpy(epoch).type(torch.FloatTensor), #, 'time': time_label.type(torch.LongTensor),
                'target': label.type(torch.LongTensor),
                'time': time_label.to(torch.int64)}  # } #}  torch.from_numpy(label_one_hot).type(torch.LongTensor)

    # def __getitem__(self, i):
    #     id = self.ids[i]
    #     #print('index:', idx)
    #     data_file = glob(self.data_dir + id + '.csv')
    #     #print('mask_file:', mask_file)
    #     #print('img_file:', img_file)
    #     assert len(data_file) == 1, \
    #         f'Either no image or multiple images found for the ID {id}: {data_file}'

    #     data = np.loadtxt(data_file[0], delimiter=',')
    #     # data = np.reshape(data, (60, 50, 2))
    #     # data = np.swapaxes(data, 0, 2)
    #     data = np.expand_dims(data, axis=1)
    #     data = np.swapaxes(data, 0, 2)
    #     # print('data:', data.shape)
    #     str_list = []
    #     last_idx = 0
    #     for j in re.finditer('_', id):
    #         if j:
    #             # print(j, j.group(0))
    #             idx = j.span()[0]
    #             str_list.append(id[last_idx:idx])
    #             last_idx = j.span()[1]
    #     # print(idx, str_list)

    #     # print(data.shape)
    #     # img = Image.open(data_file[0])
    #     #img = cv2.imread(img_file[0])

    #     # img = self.preprocess(img, self.scale)
    #     #label = int(idx[:re.search('_', idx).span()[0]])
    #     label = torch.tensor(int(str_list[3]))
    #     # print(label)
    #     label_one_hot  = np.zeros(self.n_classes)
    #     label_one_hot[label] = 1

    #     time_label = torch.Tensor([int(str_list[0]), int(str_list[1]), int(str_list[2])])
    #     # current_t = start_time
    #     # time_label_np = np.zeros((60, 3))
    #     # for i in range(30):
    #     #     for j in range(2):
    #     #         time_label_np[i*2+j, :] = current_t
    #     #     current_t = self.return_current_time(current_t)
    #     # # print(time_label_np)
    #     # # time_label_np = np.expand_dims(time_label_np, axis=0)
    #     # # print(time_label_np, time_label_np.shape)
    #     # time_label = torch.from_numpy(time_label_np)
    #     # .type(torch.FloatTensor)
    #     #print(img, label_one_hot)
    #     #print(np.max(img), np.min(img))
    #     #print(img.shape, label_one_hot.shape)
    #     #print(img_file[0], np.array(img).shape)
    #     #print('img:', img.shape)
    #     #print('mask:', mask[0].shape)
    #     #print('mask_M:', np.max(mask[0]))
    #     #print('img:', torch.from_numpy(img).type(torch.ByteTensor), 'mask:', torch.from_numpy(mask[0]).type(torch.ByteTensor))
    #     return {'data': torch.from_numpy(data).type(torch.FloatTensor), 'time': time_label.to(torch.int64), 'target': label.type(torch.LongTensor)}# } #}  torch.from_numpy(label_one_hot).type(torch.LongTensor)
    # 



