import pickle
import numpy as np
import pandas as pd
import csv

def sampling_ratio():
    group_list = ['g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9', 'g10']
    pop_ratio = 0
    pref_ratio = 0
    poppref_ratio = 0
    
    user_weights = pd.read_csv("../data/userdata/user_weight.csv", header=None).iloc[0].to_list()

    for index, i in enumerate(group_list):
        # 保存したPickleファイルを読み込む
        filename = '../data/kde/kde_model_' + i + '.pkl'
        with open(filename, 'rb') as f:
            kde = pickle.load(f)

        # サンプリングを行う
        sampling_data = kde.resample(size=1)
        pop_ratio += np.clip(sampling_data[0], 0, 1)*user_weights[index]
        pref_ratio += np.clip(sampling_data[1], 0, 1)*user_weights[index]
        poppref_ratio += np.clip(sampling_data[2], 0, 1)*user_weights[index]
        #print(sampling_data)

    # print('pop_ratio: ', pop_ratio)
    # print('pref_ratio: ', pref_ratio)
    # print('poppref_ratio: ', poppref_ratio)
    
    # 足した時に1になるように変更
    scale = 1 / (pop_ratio + pref_ratio + poppref_ratio)
    pop_ratio_scale = pop_ratio * scale
    pref_ratio_scale = pref_ratio * scale
    poppref_ratio_scale = poppref_ratio * scale

    # print('pop_ratio_scale: ', pop_ratio_scale)
    # print('pref_ratio_scale: ', pref_ratio_scale)
    # print('poppref_ratio_scale: ', poppref_ratio_scale)
    
    return pop_ratio_scale, pref_ratio_scale, poppref_ratio_scale