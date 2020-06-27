import numpy as np
import pandas as pd
import os


def save_output(episode_reward, save_path):
    assert isinstance(episode_reward, list), 'episode_reward should be a list.'
    data = np.array(episode_reward)
    np.savetxt(save_path, data)


def concatenate_data(directory_path, num, algorithm):
    assert isinstance(algorithm, str), 'Algorithm should be a string!'
    df = pd.DataFrame({'Epoch': [i for i in range(1, 2001)]})
    for filename in os.listdir(directory_path):
        episode_reward = np.loadtxt(directory_path + filename)
        df[filename.split('.txt')[0]] = episode_reward

    df_copy = df.copy(deep=True)
    storage = []
    column_num = 1
    for column, row in df.iteritems():  # traverse columns to concatenate
        df_copy.columns = ['Epoch'] + [algorithm for _ in range(num)]
        if column != 'Epoch':
            a = df_copy.iloc[:, [0, column_num]]
            storage.append(a)
            column_num += 1

    df = pd.concat(storage)

    return df

