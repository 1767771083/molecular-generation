import json
import os

import numpy as np
from src.config_parser import get_args
args = get_args()

bace_atomic_num_list = [6, 7, 8, 9, 16, 17, 35, 53, 0]  # 0 is for virtual node.
delaney_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
freesolv_atomic_num_list = [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
bbbp_atomic_num_list = [5, 6, 7, 8, 9, 11, 15, 16, 17, 20, 35, 53, 0]
lipo_atomic_num_list = [5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53, 0]
clintox_atomic_num_list = [5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 20, 22, 24, 25, 26, 29, 30, 33, 34, 35, 43, 53, 78, 79, 80, 81, 83, 0]

bace_max_atoms = 97
delaney_max_atoms = 55
freesolv_max_atoms = 24
lipo_max_atoms = 115
bbbp_max_atoms = 132
clintox_max_atoms = 136
n_bonds = 4

def one_hot_bace(data, out_size=bace_max_atoms):
    num_max_id = len(bace_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = bace_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b

def transform_fn_bace(data):
    node, adj, label = data
    # convert to one-hot vector
    node = one_hot_bace(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label

def one_hot_delaney(data, out_size=delaney_max_atoms):
    num_max_id = len(delaney_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = delaney_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b

def transform_fn_delaney(data):
    node, adj, label = data
    # convert to one-hot vector
    node = one_hot_delaney(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label

def one_hot_freesolv(data, out_size=freesolv_max_atoms):
    num_max_id = len(freesolv_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = freesolv_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b

def transform_fn_freesolv(data):
    node, adj, label = data
    # convert to one-hot vector
    # node = one_hot(node).astype(np.float32)
    node = one_hot_freesolv(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label

def one_hot_clintox(data, out_size=clintox_max_atoms):
    num_max_id = len(clintox_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = clintox_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b

def transform_fn_clintox(data):
    node, adj, label = data
    # convert to one-hot vector
    # node = one_hot(node).astype(np.float32)
    node = one_hot_clintox(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label


def one_hot_bbbp(data, out_size=bbbp_max_atoms):
    num_max_id = len(bbbp_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = bbbp_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b

def transform_fn_bbbp(data):
    node, adj, label = data
    # convert to one-hot vector
    # node = one_hot(node).astype(np.float32)
    node = one_hot_bbbp(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label

def one_hot_lipo(data, out_size=lipo_max_atoms):
    num_max_id = len(lipo_atomic_num_list)
    assert data.shape[0] == out_size
    b = np.zeros((out_size, num_max_id), dtype=np.float32)
    for i in range(out_size):
        ind = lipo_atomic_num_list.index(data[i])
        b[i, ind] = 1.
    return b

def transform_fn_lipo(data):
    node, adj, label = data
    # convert to one-hot vector
    # node = one_hot(node).astype(np.float32)
    node = one_hot_lipo(node).astype(np.float32)
    # single, double, triple and no-bond. Note that last channel axis is not connected instead of aromatic bond.
    adj = np.concatenate([adj[:3], 1 - np.sum(adj[:3], axis=0, keepdims=True)],
                         axis=0).astype(np.float32)
    return node, adj, label