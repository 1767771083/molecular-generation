import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit.Chem.Descriptors import qed
from rdkit.Chem.Crippen import MolLogP
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from src.utils import check_validity, calculate_sa_score


class NumpyTupleDataset(Dataset):
    """Dataset of a tuple of datasets.

        It combines multiple datasets into one dataset. Each example is represented
        by a tuple whose ``i``-th item corresponds to the i-th dataset.
        And each ``i``-th dataset is expected to be an instance of numpy.ndarray.

        Args:
            datasets: Underlying datasets. The ``i``-th one is used for the
                ``i``-th item of each example. All datasets must have the same
                length.

        """

    def __init__(self, datasets, transform=None):
        # Load dataset

        if not datasets:
            raise ValueError('no datasets are given')
        length = len(datasets[0])  # qm9/133885
        for i, dataset in enumerate(datasets):
            if len(dataset) != length:
                raise ValueError(
                    'dataset of the index {} has a wrong length'.format(i))
        # Initialization
        self._datasets = datasets
        self._length = length
        self.transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        batches = [dataset[index] for dataset in self._datasets]
        if isinstance(index, (slice, list, np.ndarray)):
            length = len(batches[0])
            batches = [tuple([batch[i] for batch in batches])
                    for i in range(length)]   # six.moves.range(length)]
        else:
            batches = tuple(batches)

        if self.transform:
            batches = self.transform(batches)
        return batches

    def get_datasets(self):
        return self._datasets


    @classmethod
    def save(cls, filepath, numpy_tuple_dataset):
        """save the dataset to filepath in npz format

        Args:
            filepath (str): filepath to save dataset. It is recommended to end
                with '.npz' extension.
            numpy_tuple_dataset (NumpyTupleDataset): dataset instance

        """
        if not isinstance(numpy_tuple_dataset, NumpyTupleDataset):
            raise TypeError('numpy_tuple_dataset is not instance of '
                            'NumpyTupleDataset, got {}'
                            .format(type(numpy_tuple_dataset)))
        np.savez(filepath, *numpy_tuple_dataset._datasets)
        print('Save {} done.'.format(filepath))

    @classmethod
    def load(cls, filepath, transform=None):
        print('Loading file {}'.format(filepath))
        if not os.path.exists(filepath):
            raise ValueError('Invalid filepath {} for dataset'.format(filepath))
            # return None
        load_data = np.load(filepath)   # load_data = np.load(filepath)
        result = []
        i = 0
        while True:
            key = 'arr_{}'.format(i)
            if key in load_data.keys():
                result.append(load_data[key])  # results.append(load_data[key])
                i += 1
            else:
                break
        return cls(result, transform)

def get_data_loader(args, data_file, transform_fn, atomic_num_list, valid_idx, split=True):

    # Datasets:
    dataset = NumpyTupleDataset.load(os.path.join(args.data_dir, data_file), transform=transform_fn)
    if split:
        if len(valid_idx) > 0:
            # 120803 = 133885-13082
            train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
            # n_train = len(train_idx)  # 120803
            train = Subset(dataset, train_idx)  # 120,803
            test = Subset(dataset, valid_idx)  # 13,082
        else:
            torch.manual_seed(42)
            train, test = random_split(
                dataset,
                [int(len(dataset) * 0.9),
                 len(dataset) - int(len(dataset) * 0.9)])
        test_dataloader = DataLoader(test,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     num_workers=0)
    else:
        train = dataset
        test_dataloader = None
    train_res = None
    if train is not None:
        train_x = np.stack([a[0].astype(np.float32) for a in train], axis=0)
        train_adj = np.stack([a[1].astype(np.float32) for a in train], axis=0)
        train_res = check_validity(train_adj,
                                   train_x,
                                   atomic_num_list,
                                   print_information=False)
    #
    #     train_smiles_qed = [qed(mol) for mol in train_res['valid_mols']]
    #     train_smiles_logp = [MolLogP(mol) for mol in train_res['valid_mols']]
    #     train_smiles_sa = [calculate_sa_score(mol) for mol in train_res['valid_mols']]
    #     print('train mean qed:{:.3f} max qed:{:.3f} min qed:{:.3f} train mean logp:{:.3f} train mean sa:{:.3f}'.format(
    #         np.mean(train_smiles_qed), np.max(train_smiles_qed), np.min(train_smiles_qed),
    #         np.mean(train_smiles_logp), np.mean(train_smiles_sa)))

    train_dataloader = DataLoader(train,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0)  # 12


    return train_dataloader, test_dataloader, train_res