import os
import sys
# for linux env. 改这个
sys.path.insert(0, '..')
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import time
from data.data_frame_parser import DataFrameParser
from data.data_loader import NumpyTupleDataset
from data.smile_to_graph import GGNNPreprocessor


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str, default='jak2',
                        choices=['qm9', 'zinc250k', 'moses', 'bs1', 'jak2', 'bace', 'delaney', 'freesolv', 'clintox'],
                        help='dataset to be downloaded')
    parser.add_argument('--data_type', type=str, default='relgcn',
                        choices=['gcn', 'relgcn'],)
    args = parser.parse_args()
    return args


start_time = time.time()
args = parse()
data_name = args.data_name
data_type = args.data_type
print('args', vars(args))

if data_name == 'qm9':
    max_atoms = 9
elif data_name == 'zinc250k':
    max_atoms = 38
elif data_name == 'moses':
    max_atoms = 27
elif data_name == 'bs1':
    max_atoms = 75
elif data_name == 'jak2':
    max_atoms = 53
elif data_name == 'bace':
    max_atoms = 97
elif data_name == 'delaney':
    max_atoms = 55
elif data_name == 'freesolv':
    max_atoms = 24
elif data_name == 'clintox':
    max_atoms = 136
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))


if data_type == 'relgcn':
    # preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True, return_is_real_node=False)
    preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)
else:
    raise ValueError("[ERROR] Unexpected value data_type={}".format(data_type))

data_dir = "."
os.makedirs(data_dir, exist_ok=True)

if data_name == 'qm9':
    print('Preprocessing qm9 data')
    df_qm9 = pd.read_csv('qm9.csv', index_col=0)
    labels = ['LogP', 'QED', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES')
    result = parser.parse(df_qm9, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']

elif data_name == 'zinc250k':
    print('Preprocessing zinc250k data')
    # dataset = datasets.get_zinc250k(preprocessor)
    df_zinc250k = pd.read_csv('zinc250k.csv', index_col=0)
    # 'smiles' column contains '\n', need to remove it.
    labels = ['logP', 'qed', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
    result = parser.parse(df_zinc250k, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']

elif data_name == 'moses':
    print('Preprocessing {} data'.format(data_name))
    df_moses = pd.read_csv('moses.csv', index_col=0)
    labels = ['LogP', 'QED', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES')
    result = parser.parse(df_moses, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
elif data_name == 'bs1':
    print('Preprocessing {} data'.format(data_name))
    df_bs1 = pd.read_csv('bs1.csv', index_col=0)
    labels = ['pIC50', 'QED', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES')
    result = parser.parse(df_bs1, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
elif data_name == 'jak2':
    print('Preprocessing {} data'.format(data_name))
    df_jak2 = pd.read_csv('jak2.csv', index_col=0)
    labels = ['pIC50', 'QED', 'SAS']
    parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES')
    result = parser.parse(df_jak2, return_smiles=True)
    dataset = result['dataset']
    smiles = result['smiles']
else:
    raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

NumpyTupleDataset.save(os.path.join(data_dir, '{}_{}_kekulized_ggnp.npz'.format(data_name, data_type)), dataset)
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
