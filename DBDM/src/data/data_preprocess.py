import os
import sys
sys.path.insert(0, '..')
import pandas as pd
import argparse
import time
from data.data_frame_parser import DataFrameParser
from data.data_loader import NumpyTupleDataset
from data.smile_to_graph import GGNNPreprocessor

def data_param(args):
    data_name = args.data_name
    data_type = args.data_type
    print('args', vars(args))

    if data_name == 'qm9':
        max_atoms = 9
    elif data_name == 'zinc250k':
        max_atoms = 38
    elif data_name == 'moses':
        max_atoms = 26
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
    elif data_name == 'bbbp':
        max_atoms = 132
    elif data_name == 'lipo':
        max_atoms = 115
    elif data_name == 'gen':
        max_atoms = 75
    else:
        raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))


    if data_type == 'relgcn':
        # preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True, return_is_real_node=False)
        preprocessor = GGNNPreprocessor(out_size=max_atoms, kekulize=True)
    else:
        raise ValueError("[ERROR] Unexpected value data_type={}".format(data_type))

    if data_name == 'qm9':
        print('Preprocessing qm9 data:')
        df_qm9 = pd.read_csv('qm9.csv', index_col=0)
        labels = ['A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2',
                  'zpve', 'U0', 'U', 'H', 'G', 'Cv']
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES1')
        result = parser.parse(df_qm9, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']
    elif data_name == 'zinc250k':
        print('Preprocessing zinc250k data')
        # dataset = datasets.get_zinc250k(preprocessor)
        df_zinc250k = pd.read_csv('zinc250k.csv', index_col=0)
        # Here we do not remove \n, because it represents atom N with single bond
        labels = ['logP', 'qed', 'SAS']
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
        result = parser.parse(df_zinc250k, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']
    elif data_name == 'moses':
        print('Preprocessing {} data'.format(data_name))
        df_moses = pd.read_csv('E:\Desktop\AED-GAN\data\moses.csv', index_col=0)
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
    elif data_name == 'bace':
        print('Preprocessing {} data'.format(data_name))
        df_moses = pd.read_csv('data/pred_data/bace.csv')
        labels = ['Class']
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='mol')
        result = parser.parse(df_moses, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']
    elif data_name == 'delaney':
        print('Preprocessing {} data'.format(data_name))
        df_moses = pd.read_csv('data/pred_data/delaney.csv')
        labels = ['logSolubility']
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
        result = parser.parse(df_moses, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']
    elif data_name == 'freesolv':
        print('Preprocessing {} data'.format(data_name))
        df_moses = pd.read_csv('data/pred_data/freesolv.csv')
        labels = ['freesolv']
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
        result = parser.parse(df_moses, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']
    elif data_name == 'clintox':
        print('Preprocessing {} data'.format(data_name))
        df_moses = pd.read_csv('data/pred_data/clintox.csv')
        labels = ['FDA_APPROVED', 'CT_TOX']
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
        result = parser.parse(df_moses, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']
    elif data_name == 'bbbp':
        print('Preprocessing {} data'.format(data_name))
        df_moses = pd.read_csv('data/pred_data/bbbp.csv')
        labels = ['p_np']
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
        result = parser.parse(df_moses, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']
    elif data_name == 'lipo':
        print('Preprocessing {} data'.format(data_name))
        df_moses = pd.read_csv('data/pred_data/lipo.csv')
        labels = ['lipo']
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='smiles')
        result = parser.parse(df_moses, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']
    elif data_name == 'gen':
        print('Preprocessing {} data'.format(data_name))
        df_bs1 = pd.read_csv('E:/Desktop/AED-GAN/data/bs1_pic_results.csv')
        labels = ['QED']
        parser = DataFrameParser(preprocessor, labels=labels, smiles_col='SMILES')
        result = parser.parse(df_bs1, return_smiles=True)
        dataset = result['dataset']
        smiles = result['smiles']
    else:
        raise ValueError("[ERROR] Unexpected value data_name={}".format(data_name))

    return result, dataset, smiles


def parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_name', type=str, default='gen',
                        choices=['qm9', 'zinc250k', 'moses', 'bs1', 'jak2', 'gen',
                                 'bace', 'delaney', 'freesolv', 'clintox', 'bbbp', 'lipo'],
                        help='dataset to be downloaded')
    parser.add_argument('--data_type', type=str, default='relgcn',
                        choices=['gcn', 'relgcn'],)
    parser.add_argument('--data_dir', type=str, default=".")

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    return args

args = parse()
start_time = time.time()

result, dataset, smiles = data_param(args)

NumpyTupleDataset.save(os.path.join(args.data_dir, '{}_{}_kekulized_ggnp.npz'.format(args.data_name, args.data_type)), dataset)
print('Total time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time)))
