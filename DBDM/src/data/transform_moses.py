import json
import os

import numpy as np
from src.config_parser import get_args
args = get_args()

moses_atomic_num_list = [6, 7, 8, 9, 16, 17, 35, 0]  # 0 is for virtual node.
max_atoms = 26
n_bonds = 4


def one_hot_moses(data, out_size=26):
    num_max_id = len(moses_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = moses_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b


def transform_fn_moses(data):
    node, adj, label = data
    node = one_hot_moses(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label


def get_val_ids():
    file_path = os.path.join(args.data_dir, 'valid_idx_moses.json')
    print('loading train/valid split information from: {}'.format(file_path))
    with open(file_path) as json_data:
        data = json.load(json_data)
    val_ids = [idx-1 for idx in data]
    return val_ids

