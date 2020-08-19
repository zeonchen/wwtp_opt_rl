import os
import time
import pandas as pd


def clear_text(save_path):
    with open(save_path, "r+") as f:
        f.seek(0)
        f.truncate()  # clear file

    size = os.path.getsize(save_path)

    return size


def txt_read(save_path):
    data_list = []
    size = os.path.getsize(save_path)

    while size == 0:
        # print('Wait for state writing...')
        size = os.path.getsize(save_path)

    time.sleep(0.5)
    with open(save_path, "r+") as f:
        data = f.readlines()
        for i in data:
            item = float(i.strip('\n'))
            data_list.append(item)

    # clear_text()

    return data_list
