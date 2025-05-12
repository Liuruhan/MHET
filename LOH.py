import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score, auc, accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# from metric_personal import kappa, stage_sleep_cal
from model_eval import kappa
from metric_individual import stage_sleep_cal

if __name__ == "__main__":
    personal_len = np.loadtxt('personal_len.csv', delimiter=',')
    personal_list = np.loadtxt('personal_list.csv', dtype=str, delimiter=',')

    X_train, y_train = np.loadtxt('train_X.csv', delimiter=','), np.loadtxt('train_y.csv', delimiter=',')
    X_test, y_test = np.loadtxt('test_X.csv', delimiter=','), np.loadtxt('test_y.csv', delimiter=',')
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    # other_params = {'eta': 0.3, 'n_estimators': 100, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 1,
    #                 'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
    #                 'seed': 33}
    # # grid search
    # cv_params = {'n_estimators': np.linspace(100, 500, 5, dtype=int)}
    # model = xgb.XGBClassifier(**other_params)
    # gs = GridSearchCV(model, cv_params, verbose=2, refit=True, cv=5, n_jobs=-1)
    # gs.fit(X_train, y_train)
    #
    # print("Best hyperparameters:", gs.best_params_)
    # print("Best score:", gs.best_score_)

    other_params = {'eta': 0.3, 'n_estimators': 100, 'gamma': 0, 'max_depth': 6, 'min_child_weight': 1,
                    'colsample_bytree': 1, 'colsample_bylevel': 1, 'subsample': 1, 'reg_lambda': 1, 'reg_alpha': 0,
                    'seed': 33}
    model = xgb.XGBClassifier(**other_params)
    model.fit(X_train, y_train)
    print(model)

    def estimate_flops(model):
        flops = 0
        for tree in model.get_booster().get_dump():
            flops += tree.count('[')  # Count number of nodes in the tree
        return flops


    # Estimate FLOPs for the trained model
    flops = estimate_flops(model)

    print(f"Estimated FLOPs: {flops}")


    y_pred = model.predict(X_test)

    import time
    time_list = []
    for i in range(400):
        start_time = time.time()
        y_pred = model.predict(X_test[:16, :])
        end_time = time.time()

        # Calculate inference time
        inference_time = end_time - start_time
        time_list.append(inference_time)

    mean = np.mean(np.array(time_list))
    std = np.std(np.array(time_list))

    print(f"Inference mean time: {mean:.6f} seconds")
    print(f"Inference std time: {std:.6f} seconds")

    plt.bar(range(len(model.feature_importances_)), model.feature_importances_)
    plt.savefig('feature_importance.png')

    print(accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))
    tmp_cm = confusion_matrix(y_test, y_pred)
    print(kappa(tmp_cm))

    k = 0
    ave_r2, ave_kappa, ave_acc, ave_mf1 = [], [], [], []
    t_stgs = np.zeros((len(personal_len), 5))
    p_stgs = np.zeros((len(personal_len), 5))
    r2_p = np.zeros(len(personal_len))
    ave_err_stgs_mean, ave_err_stg_std = 0, 0
    for i in range(len(personal_len)):
        start = k
        end = k + personal_len[i]

        plt.figure(figsize=(8, 3))
        plt.subplot(2, 1, 1)
        plt.plot(range(len(y_test[start:end])), y_test[start:end], color='#4A4A90')
        plt.subplot(2, 1, 2)
        plt.plot(range(len(y_pred[start:end])), y_pred[start:end], color='#C7C3DB')
        plt.savefig('./person_vis/LMHT/' + personal_list[i] + '.png')
        plt.close()

        # metrics
        result = classification_report(y_test[start:end], y_pred[start:end], labels=[0, 1, 2, 3, 4],
                                       target_names=['W', 'N1', 'N2', 'N3', 'R'], digits=4, output_dict=True)
        ave_acc.append(accuracy_score(y_test[start:end], y_pred[start:end]))
        ave_mf1.append(((result['W']['f1-score'] + result['N1']['f1-score'] + result['N2']['f1-score'] + result['N3'][
            'f1-score'] + result['R']['f1-score']) / 5))
        ave_r2.append(r2_score(y_test[start:end], y_pred[start:end]))
        print('R2:', r2_score(y_test[start:end], y_pred[start:end]))
        tmp_cm = confusion_matrix(y_test[start:end], y_pred[start:end])
        ave_kappa.append(kappa(tmp_cm))

        tmp_t_stg = stage_sleep_cal(y_test[start:end])
        t_stgs[i, :] = tmp_t_stg
        tmp_p_stg = stage_sleep_cal(y_pred[start:end])
        p_stgs[i, :] = tmp_p_stg
        err = np.abs(tmp_t_stg - tmp_p_stg)
        ave_err_stgs_mean += np.mean(err)
        ave_err_stg_std += np.std(err)

        k = end

    ave_err_stgs_mean = ave_err_stgs_mean / len(personal_len)
    ave_err_stg_std = ave_err_stg_std / len(personal_len)
    print('Ave acc:', np.sum(np.array(ave_acc)) / len(personal_len), np.std(np.array(ave_acc)) / len(personal_len))
    print('Ave mf1:', np.sum(np.array(ave_mf1)) / len(personal_len), np.std(np.array(ave_mf1)) / len(personal_len))
    print('Ave kappa:', np.sum(np.array(ave_kappa)) / len(personal_len),
          np.std(np.array(ave_kappa)) / len(personal_len))
    print('Ave r2:', np.sum(np.array(ave_r2)) / len(personal_len), np.std(np.array(ave_r2)) / len(personal_len))
    print('Ave sleep propotion:', ave_err_stgs_mean, '+/-', ave_err_stg_std)

    stg_list = ['W', 'N1', 'N2', 'N3', 'R']
    for i in range(t_stgs.shape[1]):
        out_data = np.zeros((t_stgs.shape[0], 2))
        out_data[:, 0] = t_stgs[:, i]
        out_data[:, 1] = p_stgs[:, i]
        np.savetxt('./person_vis/LMHT/' + stg_list[i] + '_prop.csv', out_data, delimiter=',')
