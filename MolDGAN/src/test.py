import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from rdkit.Chem.Descriptors import qed
from src.data.data_loader import get_data_loader
from src.model import LatentDecoder, FakeEncoder
from src.utils import check_validity, check_novelty


def test(args, test_index):
    seed = test_index
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_file, transform_fn, atomic_num_list, \
    b_n_type, a_n_node, a_n_type, valid_idx = get_data_par(args.data_name)

    train_dataloader, test_dataloader, train_res = get_data_loader(args, data_file, transform_fn,
                                                                   atomic_num_list, valid_idx, split=True)

    latent_decoder = LatentDecoder(args.latent_size, a_n_node, b_n_type, a_n_type).to(device)

    generator = Generator(args.latent_size, a_n_node, args.latent_size).to(device)

    generator.load_state_dict(
        torch.load(os.path.join(args.model_dir, '{}_generator.pth'.format(args.data_name))))
    latent_decoder.load_state_dict(
        torch.load(os.path.join(args.model_dir, '{}_latent_decoder.pth'.format(args.data_name))))

    latent_size = 32
    hidden_size = 128
    x_list = []
    adj_list = []
    x_num = 0
    while x_num < 10000:
        with torch.no_grad():
            latent_decoder.eval()
            x, adj = latent_decoder(torch.randn(100,
                                                latent_size,
                                                device=device),
                                    gumbel_softmax=True)
            x_num += 100
            x_list.append(x)
            adj_list.append(adj)
    x_list = torch.cat(x_list, dim=0)[:10000]
    adj_list = torch.cat(adj_list, dim=0)[:10000]

    valid_res = check_validity(adj_list,
                               x_list,
                               atomic_num_list,
                               print_information=False)
    novel_r, abs_novel_r = check_novelty(valid_res['valid_smiles'],
                                         train_res['valid_smiles'],
                                         x_list.shape[0])

    print(
        "valid: {:.3f}%, unique: {:.3f}%, novelty: {:.3f}%"
            .format(valid_res['valid_ratio'], valid_res['unique_ratio'], novel_r))

    # cal property
    print('===================================================')
    test_smiles_qed = [qed(mol) for mol in valid_res['valid_mols']]
    print(
        'test max QED:{:.3f}'.format(np.mean(test_smiles_qed))
    )
    print('===================================================')

    valid_smiles_qed = [qed(mol) for mol in valid_res['valid_mols']]

    top_qed_index = np.argsort(valid_smiles_qed)[::-1]

    top_qed_mols = []
    top_qed_values = []
    for i in range(20):
        top_qed_mols.append(valid_res['valid_mols'][top_qed_index[i]])
        top_qed_values.append('{:.3f}'.format(
            valid_smiles_qed[top_qed_index[i]]))

    # Save results
    eval_dir = os.path.join(args.test_subdir, args.data_name)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    with open(os.path.join(eval_dir, f"{args.data_name}_test_results_{test_index}.txt"), "w") as file:
        file.write(f"valid: {valid_res['valid_ratio']:.3f}%\n")
        file.write(f"unique: {valid_res['unique_ratio']:.3f}%\n")
        file.write(f"novelty: {novel_r:.3f}%\n")
        file.write(f"G-Mean: {(np.mean(valid_res['valid_ratio'] + valid_res['unique_ratio'] + novel_r)) / 3:.3f}\n")
        file.write(f"test mean QED: {np.mean(test_smiles_qed):.3f}\n")


    print("Results saved to {}_test_results_{}.txt".format(args.data_name, test_index))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='moses', help='choose data name')
    parser.add_argument('--save_dir', type=str, help='choose save dir')
    args = parser.parse_args()
    args.save_dir = args.save_dir + '_' + args.data_name

    # test(args)
    for i in range(50):
        test(args, i)
