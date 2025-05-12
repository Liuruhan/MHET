import numpy as np
from metric_personal import split_into_name_time, softmax

def cal_all_metrics(out_name, single_EDF, EDF_list, time_list, sleep_stg_list, out_np):
    first_mark = True
    personal_len = []
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

        sql_len = 3
        personal_len.append(single_EDF[i])
        save_f = np.zeros((e_out_np.shape[0]-sql_len, e_out_np.shape[1]*sql_len))
        save_l = np.zeros(e_out_np.shape[0]-sql_len)
        for j in range(e_out_np.shape[0]-sql_len):
            tmp_f = e_out_np[j:j+sql_len].flatten()
            save_f[j, :] = tmp_f
            save_l[j] = tgt[j+sql_len//2]
        if first_mark == True:
            copy_f = save_f
            copy_l = save_l
            first_mark = False
        else:
            copy_f = np.concatenate((copy_f, save_f), axis=0)
            copy_l = np.concatenate((copy_l, save_l), axis=0)

    print('X', copy_f.shape)
    print('y', copy_l.shape)
    np.savetxt(out_name+'_X.csv', copy_f, delimiter=',')
    np.savetxt(out_name+'_y.csv', copy_l, delimiter=',')
    print(personal_len)
    return

def generate_LOH_dataset(file, out_name):
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
            out_np[i, j] = float(file[i, j + 2])
        EDF_list.append(EDF_name)
        sleep_stg_list.append(sleep_stg)
        time_list.append(create_time)

    cal_all_metrics(out_name, single_EDF, EDF_list, time_list, sleep_stg_list, out_np)
    return

if __name__ == "__main__":
    flie_list = ['NAME_tgt_out_LMHTtrain.csv', 'NAME_tgt_out_LMHTtest.csv']
    out_list = ['train', 'test']
    root = './'
    for i in range(len(flie_list)):
        file = np.loadtxt(root + flie_list[i], delimiter=',', dtype=str, skiprows=1)
        generate_LOH_dataset(file, out_list[i])