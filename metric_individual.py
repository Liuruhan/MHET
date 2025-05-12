import re
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score

from model_eval import kappa

def split_into_name_time(name_str):
    str_list = []
    last_idx = 0
    for j in re.finditer('_', name_str):
        if j:
            idx = j.span()[0]
            str_list.append(name_str[last_idx:idx])
            last_idx = j.span()[1]
    str_list.append(name_str[last_idx:-5])
    # print(str_list)
    if len(str_list) != 5:
        print('Error!')

    sleep_stg = str_list[3]
    EDF_name = str_list[4]
    # create_time = datetime.time(hour=int(str_list[0]), minute=int(str_list[1]), second=int(str_list[2]))
    # print(str_list[0], str_list[1], str_list[2])
    create_time = (int(str_list[0]) * 60 * 60 + int(str_list[1]) * 60 + int(str_list[2])) // 30
    return EDF_name, sleep_stg, create_time

def softmax(data_np):
    # instead: first shift the values of f so that the highest number is 0:
    for i in range(data_np.shape[0]):
        f = data_np[i, :]
        f -= np.max(f) # f becomes [-666, -333, 0]
        data_np[i, :] = np.exp(f) / np.sum(np.exp(f))
    return data_np # safe to do, gives the correct answer

def stage_sleep_cal(data):
    if len(data) == 0:
        return np.array([0, 0, 0, 0, 0])
    W, N1, N2, N3, R = 0, 0, 0, 0, 0
    for i in range(data.shape[0]):
        if data[i] == 0:
            W += 1
        elif data[i] == 1:
            N1 += 1
        elif data[i] == 2:
            N2 += 1
        elif data[i] == 3:
            N3 += 1
        elif data[i] == 4:
            R += 1
    N = W + N1 + N2 + N3 + R
    return np.array([W*1.0/N, N1*1.0/N, N2*1.0/N, N3*1.0/N, R*1.0/N])

def cal_all_metrics(single_EDF, EDF_list, time_list, sleep_stg_list, out_np, output_root):
    ave_r2, ave_kappa, ave_acc, ave_mf1 = 0, 0, 0, 0
    t_stgs = np.zeros((len(single_EDF), 5))
    p_stgs = np.zeros((len(single_EDF), 5))
    r2_p = np.zeros(len(single_EDF))
    ave_err_stgs_mean, ave_err_stg_std = 0, 0
    for i in range(len(single_EDF)):
        # print(single_EDF[i])
        p_sleep_stg_list, p_time_list = [], []
        p_idx_list = []

        # select 30s epochs from the same person
        # p_sleep_stg_list: sleep stages for one person
        # p_time_list: real time for each sleep epoch
        # p_out_np: output for one person
        for j in range(len(EDF_list)):
            if single_EDF[i] == EDF_list[j]:
                p_time_list.append(time_list[j])
                p_sleep_stg_list.append(sleep_stg_list[j])
                p_idx_list.append(j)

        p_out_np = out_np[p_idx_list, :]

        # sorted in time order
        sorted_idx = np.argsort(p_time_list)
        s_time_list = np.array(p_time_list)[sorted_idx]
        s_sleep_stg_list = np.array(p_sleep_stg_list)[sorted_idx]
        s_out_np = p_out_np[sorted_idx, :]

        # sorted in sleep order
        start, end = -1, -1
        sorted_out_idx = []
        if s_time_list[0] == 0 and s_time_list[len(s_time_list)-1] == 2879:
            # print(single_EDF[i], 'over 00:00:00')
            cur_t = s_time_list[0]
            for j in range(1, s_time_list.shape[0]):
                if s_time_list[j] - cur_t == 1:
                    cur_t = s_time_list[j]
                else:
                    end = j - 1
                    start = j
                    break
            # print(start, end)
            for j in range(start, len(sorted_idx)):
                sorted_out_idx.append(j)
            for j in range(0, end):
                sorted_out_idx.append(j)
        else:
            # print(single_EDF[i], 'not over 00:00:00')
            start = 0
            end = len(s_time_list)-1
            # print(start, end)
            for j in range(start, end):
                sorted_out_idx.append(j)

        # print(sorted_time)
        e_sleep_stg_list = s_sleep_stg_list[sorted_out_idx]
        e_out_np = s_out_np[sorted_out_idx]

        e_out_np = softmax(e_out_np)
        out = np.argmax(e_out_np, axis=1)
        tgt = np.zeros(e_sleep_stg_list.shape[0])
        for j in range(e_sleep_stg_list.shape[0]):
            tgt[j] = int(e_sleep_stg_list[j])

        # print(np.max(out), out)
        # print(np.max(tgt), tgt)
        tmp_t_stg = stage_sleep_cal(tgt)
        t_stgs[i, :] = tmp_t_stg
        tmp_p_stg = stage_sleep_cal(out)
        p_stgs[i, :] = tmp_p_stg
        err = np.abs(tmp_t_stg - tmp_p_stg)
        ave_err_stgs_mean += np.mean(err)
        ave_err_stg_std += np.std(err)

        # print('r2:', r2_score(tgt, out))
        if len(tgt) != 0:
            r2_p[i] = r2_score(tgt, out)
            ave_r2 += r2_score(tgt, out)
            result = classification_report(tgt, out, labels=[0, 1, 2, 3, 4], target_names=['W', 'N1', 'N2', 'N3', 'R'], digits=4, output_dict=True)
            ave_acc += accuracy_score(tgt, out)
            ave_mf1 += ((result['W']['f1-score']+result['N1']['f1-score']+result['N2']['f1-score']+result['N3']['f1-score']+result['R']['f1-score'])/5)
            tmp_cm = confusion_matrix(tgt, out)
            ave_kappa += kappa(tmp_cm)
            # print(result)

            plt.figure(figsize=(8, 3))
            plt.subplot(2, 1, 1)
            plt.plot(range(len(tgt)), tgt, color='#4A4A90')
            plt.subplot(2, 1, 2)
            plt.plot(range(len(out)), out, color='#C7C3DB')
            plt.savefig(output_root+single_EDF[i]+'.png')
            plt.close()

            out = np.expand_dims(out, 1)
            tgt = np.expand_dims(tgt, 1)
            save_file = np.concatenate((tgt, out, e_out_np), axis=1)
            # print(save_file.shape)
            np.savetxt(output_root+single_EDF[i]+'.csv', save_file, delimiter=',')

    ave_acc = ave_acc / len(single_EDF)
    ave_mf1 = ave_mf1 / len(single_EDF)
    ave_kappa = ave_kappa / len(single_EDF)
    ave_r2 = ave_r2 / len(single_EDF)
    ave_err_stgs_mean = ave_err_stgs_mean / len(single_EDF)
    ave_err_stg_std = ave_err_stg_std / len(single_EDF)
    print('Ave acc:', ave_acc)
    print('Ave mf1:', ave_mf1)
    print('Ave kappa:', ave_kappa)
    print('Ave r2:', ave_r2)
    print('Ave sleep propotion:', ave_err_stgs_mean, '+/-', ave_err_stg_std)
    # print(p_stgs)
    # print(t_stgs)

    stg_list = ['W', 'N1', 'N2', 'N3', 'R']
    for i in range(t_stgs.shape[1]):
        out_data = np.zeros((t_stgs.shape[0], 2))
        out_data[:, 0] = t_stgs[:, i]
        out_data[:, 1] = p_stgs[:, i]
        np.savetxt(output_root+stg_list[i]+'_prop.csv', out_data, delimiter=',')

    np.savetxt(model_name+'_r2.csv', r2_p, delimiter=',')
    return


if __name__ == "__main__":
    # model_list = ['BiLSTM', 'ConvLSTM', 'DeepSleepNet', 'EEGNet', 'InceptionTime', 'LMHT', 'ResConv', 'SalientSleepNet', 'SeqSleepNet', 'TinySleepNet', 'UTime', 'Xception']
    model_list = ['LMHT']
    for idx in range(len(model_list)):
        model_name = model_list[idx]
        root = './'
        file = np.loadtxt(root + 'NAME_tgt_out_' + model_name + '.csv', delimiter=',', dtype=str)
        output_root = './person_vis_' + model_name + '/'
        if not os.path.exists(output_root):
            os.mkdir(output_root)
        print(file.shape)

        single_EDF = []
        EDF_list, sleep_stg_list, time_list = [], [], []
        out_np = np.zeros((file.shape[0], 5))
        for i in range(file.shape[0]):
            name = file[i, 0]
            # print(name)
            EDF_name, sleep_stg, create_time = split_into_name_time(name)
            # print(EDF_name, sleep_stg, create_time)

            mark = False
            for j in range(len(single_EDF)):
                if single_EDF[j] == EDF_name:
                    mark = True
                    break
            if mark == False:
                single_EDF.append(EDF_name)

            for j in range(out_np.shape[1]):
                out_np[i, j] = float(file[i, j+2])
            EDF_list.append(EDF_name)
            sleep_stg_list.append(sleep_stg)
            time_list.append(create_time)

        cal_all_metrics(single_EDF, EDF_list, time_list, sleep_stg_list, out_np, output_root)