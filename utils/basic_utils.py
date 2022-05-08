import csv
import json
import os
import torch
import yaml
import random
import pickle
import numpy as np
from datetime import datetime


def get_current_timestamp():
    return datetime.now().strftime("%Y%m%d%H%M")


def load_json(filename):
    with open(filename, "r") as f:
        return json.loads(f.readlines()[0].strip("\n"))


def save_json(data, filename):
    with open(filename, "w") as f:
        f.write("\n".join([json.dumps(data)]))


def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_pickle(data, data_path):
    with open(data_path, "wb") as f:
        pickle.dump(data, f)


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def load_pred_cap(filename):
    data_dict = load_json(filename)
    cap_dict = dict()
    for key, value in data_dict.items():
        cap_dict[key] = [v['caption'] for v in value]
    return cap_dict


def tsv_writer(values, tsv_file_name, sep='\t'):
    if not os.path.exists(os.path.dirname(tsv_file_name)):
        os.mkdir(os.path.dirname(tsv_file_name))
    tsv_file_name_tmp = tsv_file_name + '.tmp'
    with open(tsv_file_name_tmp, 'wb') as fp:
        assert values is not None
        for value in values:
            assert value is not None
            v = sep.join(map(lambda v: v.decode() if type(v) == bytes else str(v), value)) + '\n'
            v = v.encode()
            fp.write(v)
    os.rename(tsv_file_name_tmp, tsv_file_name)


def pred_writer(values, pred_file_name):
    if not os.path.exists(os.path.dirname(pred_file_name)):
        os.mkdir(os.path.dirname(pred_file_name))
    pred_file_name_tmp = pred_file_name + '.tmp'
    with open(pred_file_name_tmp, 'w') as f:
        assert values is not None
        values_dict = dict()
        for value in values:
            assert value is not None
            values_dict[value[0]] = value[1]
        f.write("\n".join([json.dumps(values_dict)]))
    os.rename(pred_file_name_tmp, pred_file_name)


def time_sampling(max_sample_num, time_list, timestamps):
    sampled_time = dict()
    # find nearest matching
    nearest_timestamps = [10000.00 for i in range(len(timestamps))]
    for time in time_list:
        time = float(time)
        for t_idx in range(len(timestamps)):
            ts = timestamps[t_idx]
            if abs(time - ts) < abs(nearest_timestamps[t_idx] - ts):
                nearest_timestamps[t_idx] = time

    # get candidates
    sampled_time['boundary'] = nearest_timestamps[1]
    sampled_time['before'] = []
    sampled_time['after'] = []
    for time in time_list:
        time = float(time)
        if nearest_timestamps[0] <= time < nearest_timestamps[1]:
            sampled_time['before'].append(time)
        elif nearest_timestamps[1] < time <= nearest_timestamps[2]:
            sampled_time['after'].append(time)

    # start sampling
    if len(sampled_time['before']) > max_sample_num:
        # if max_sample_num == 1:
        #     sampled_time['before'] = [sampled_time['before'][int(len(sampled_time['before']) / 2)]]
        # else:
        sampling_step = len(sampled_time['before']) / max_sample_num
        sampled_idx = [int((idx + 1) * sampling_step) - 1 for idx in range(max_sample_num)]
        sampled_time['before'] = [sampled_time['before'][idx] for idx in sampled_idx]
    if len(sampled_time['after']) > max_sample_num:
        # if max_sample_num == 1:
        #     sampled_time['after'] = [sampled_time['after'][int(len(sampled_time['after']) / 2)]]
        # else:
        sampling_step = len(sampled_time['after']) / max_sample_num
        sampled_idx = [int(idx * sampling_step) for idx in range(max_sample_num)]
        sampled_time['after'] = [sampled_time['after'][idx] for idx in sampled_idx]

    return sampled_time


def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def remove_rows_cols(t, row_start, row_end, col_start, col_end):

    t00 = t[:, :row_start, :col_start]
    t01 = t[:, :row_start, col_end:]
    t10 = t[:, row_end:, :col_start]
    t11 = t[:, row_end:, col_end:]
    res = torch.cat([torch.cat([t00, t01], dim=2), torch.cat([t10, t11],
                                                             dim=2)], dim=1)
    assert res.shape == (t.shape[0], t.shape[1] - row_end + row_start,
                         t.shape[2] - col_end + col_start)
    return res
