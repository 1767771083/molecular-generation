import json
import os
import sys
import argparse
import torch
import torch.nn as nn
from model import LatentDecoder, Discriminator, TrueEncoder, FakeEncoder
from data.data_loader import NumpyTupleDataset
import time
from utils import check_validity, save_mol_png, adj_to_smiles, check_novelty
# import cairosvg
from rdkit.Chem import Draw
import numpy as np
from rdkit.Chem.Descriptors import qed  # , Molqed
from rdkit.Chem.Descriptors import MolLogP

import functools
print = functools.partial(print, flush=True)


def train(args):
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.data_name == 'qm9':
        from data import transform_qm9
        data_file = 'qm9_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_qm9.transform_fn
        atomic_num_list = [6, 7, 8, 9, 0]
        b_n_type = 4
        b_n_squeeze = 3
        a_n_node = 9
        a_n_type = len(atomic_num_list)  # 5
        valid_idx = transform_qm9.get_val_ids(
        )  # len: 13,082, total data: 133,885
    elif args.data_name == 'zinc250k':
        from data import transform_zinc250k
        data_file = 'zinc250k_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_zinc250k.transform_fn_zinc250k
        # [6, 7, 8, 9, 15, 16, 17, 35, 53, 0]
        atomic_num_list = transform_zinc250k.zinc250_atomic_num_list
        # mlp_channels = [1024, 512]
        # gnn_channels = {'gcn': [16, 128], 'hidden': [256, 64]}
        b_n_type = 4
        b_n_squeeze = 19  # 2
        a_n_node = 38
        a_n_type = len(atomic_num_list)  # 10
        valid_idx = transform_zinc250k.get_val_ids()
    elif args.data_name == 'moses10k':
        from data import transform_moses10k
        data_file = 'moses10k_relgcn_kekulized_ggnp.npz'
        transform_fn = transform_moses10k.transform_fn_moses10k
        # [6, 7, 8, 9, 16, 17, 35, 0]
        atomic_num_list = transform_moses10k.moses10_atomic_num_list
        b_n_type = 4
        b_n_squeeze = 19  # 2
        a_n_node = 26
        a_n_type = len(atomic_num_list)  # 10
        valid_idx = []  # transform_moses10k.get_val_ids()
    else:
        raise ValueError(
            'Only support qm9, zinc250k and moses10k right now. '
            'Parameters need change a little bit for other dataset.')

    # Datasets:
    dataset = NumpyTupleDataset.load(os.path.join('../data', data_file),     # '../data'
                                     transform=transform_fn)  # 133885
    if len(valid_idx) > 0:
        # 120803 = 133885-13082
        train_idx = [t for t in range(len(dataset)) if t not in valid_idx]
        # n_train = len(train_idx)  # 120803
        train = torch.utils.data.Subset(dataset, train_idx)  # 120,803
        test = torch.utils.data.Subset(dataset, valid_idx)  # 13,082
    else:
        torch.manual_seed(42)
        train, test = torch.utils.data.random_split(
            dataset,
            [int(len(dataset) * 0.95),
             len(dataset) - int(len(dataset) * 0.95)])
    train_x = np.stack([a[0] for a in train], axis=0)
    train_adj = np.stack([a[1] for a in train], axis=0)
    train_res = check_validity(train_adj,
                               train_x,
                               atomic_num_list,
                               print_information=False)
    train_smiles_qed = [qed(mol) for mol in train_res['valid_mols']]
    train_smiles_logp = [MolLogP(mol) for mol in train_res['valid_mols']]
    print('train mean qed:{:.3f} train mean logp:{:.3f}'.format(
        np.mean(train_smiles_qed), np.mean(train_smiles_logp)))

    batch_size = 1024
    train_dataloader = torch.utils.data.DataLoader(train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=12)
    print('==========================================')
    print('Load data done! Time {:.2f} seconds'.format(time.time() - start))
    print('Num Train-size: {}'.format(len(train)))
    print('Num Iter/Epoch: {}'.format(len(train_dataloader)))
    print('==========================================')

    # Loss and optimizer
    latent_size = 32
    hidden_size = 128
    true_encoder = TrueEncoder(a_n_type, hidden_size, hidden_size, b_n_type,
                               latent_size).to(device)
    fake_encoder = FakeEncoder(latent_size).to(device)
    latent_decoder = LatentDecoder(latent_size, a_n_node, b_n_type,
                                   a_n_type).to(device)
    discriminator = Discriminator(latent_size).to(device)
    true_encoder_optimizer = torch.optim.Adam(true_encoder.parameters(),
                                              lr=1e-4)
    fake_encoder_optimizer = torch.optim.Adam(fake_encoder.parameters(),
                                              lr=1e-4)
    latent_decoder_optimizer = torch.optim.Adam(latent_decoder.parameters(),
                                                lr=1e-4)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                               lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    # Train the models
    iter_per_epoch = len(train_dataloader)
    max_epochs = 100
    for epoch in range(max_epochs):
        print("In epoch {}, Time: {}".format(epoch + 1, time.ctime()))
        total_reconstruction_loss = 0
        total_discriminator_loss = 0
        for i, batch in enumerate(train_dataloader):
            # turn off shuffle to see the order with original code
            x = batch[0].to(device)  # (256, 9, 5)
            adj = batch[1].to(device)  # (256, 4, 9, 9)
            true_encoder_output = true_encoder(x, adj)
            true_decoder_node, true_decoder_adj = latent_decoder(
                true_encoder_output, gumbel_softmax=False)
            fake_input = torch.randn(x.shape[0], latent_size, device=device)
            fake_encoder_output = fake_encoder(fake_input)
            true_discriminator_outputs = discriminator(
                true_encoder_output,
                torch.randint(low=0,
                              high=discriminator.total_time_step,
                              size=(x.shape[0], 1),
                              device=device))
            fake_discriminator_outputs = discriminator(
                fake_encoder_output,
                torch.randint(low=0,
                              high=discriminator.total_time_step,
                              size=(x.shape[0], 1),
                              device=device))

            reconstruction_loss = criterion(
                true_decoder_node.reshape(
                    true_decoder_node.shape[0] * true_decoder_node.shape[1],
                    true_decoder_node.shape[2]),
                x.reshape(x.shape[0] * x.shape[1], x.shape[2])) + criterion(
                    true_decoder_adj.transpose(3, 1).reshape(
                        true_decoder_adj.shape[0] * true_decoder_adj.shape[2] *
                        true_decoder_adj.shape[3], true_decoder_adj.shape[1]),
                    adj.transpose(3, 1).reshape(
                        adj.shape[0] * adj.shape[2] * adj.shape[3],
                        adj.shape[1]))

            discriminator_loss = torch.mean(
                fake_discriminator_outputs) - torch.mean(
                    true_discriminator_outputs)
            fake_encoder_optimizer.zero_grad()
            (-discriminator_loss).backward(retain_graph=True)
            fake_encoder_optimizer.step()
            true_encoder_optimizer.zero_grad()
            reconstruction_loss.backward(retain_graph=True)
            (-discriminator_loss).backward(retain_graph=True)
            true_encoder_optimizer.step()
            latent_decoder_optimizer.zero_grad()
            reconstruction_loss.backward(retain_graph=True)
            latent_decoder_optimizer.step()
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()
            total_reconstruction_loss += reconstruction_loss.item()
            total_discriminator_loss += discriminator_loss.item()
        print(
            'Epoch [{}/{}], Iter [{}/{}], Reconstruction Loss:{:}, Discriminator Loss:{:}'
            .format(epoch + 1, max_epochs, i + 1, iter_per_epoch,
                    total_reconstruction_loss / len(train_dataloader),
                    total_discriminator_loss / len(train_dataloader)))

        def print_validity(ith):
            with torch.no_grad():
                latent_decoder.eval()
                x, adj = latent_decoder(torch.randn(100,
                                                    latent_size,
                                                    device=device),
                                        gumbel_softmax=True)
                valid_res = check_validity(adj, x, atomic_num_list)
                valid_mols = valid_res['valid_mols']
                novel_r, abs_novel_r = check_novelty(valid_res['valid_smiles'],
                                                     train_res['valid_smiles'],
                                                     x.shape[0])
                print(
                    "valid: {:.3f}%, unique: {:.3f}%, abs unique: {:.3f}% novelty: {:.3f}% abs novelty: {:.3f}%"
                    .format(valid_res['valid_ratio'],
                            valid_res['unique_ratio'],
                            valid_res['abs_unique_ratio'], novel_r,
                            abs_novel_r))
                mol_dir = os.path.join(args.save_dir,
                                       'generated_{}'.format(ith))
                os.makedirs(mol_dir, exist_ok=True)
                for ind, mol in enumerate(valid_mols):
                    save_mol_png(mol,
                                 os.path.join(mol_dir, '{}.png'.format(ind)))
                latent_decoder.train()

        print_validity(epoch + 1)

    print("[Training Ends], Start at {}, End at {}".format(
        time.ctime(start), time.ctime()))
    print('Start Testing')

    def test():
        # test generation
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
            "valid: {:.3f}%, unique: {:.3f}%, abs unique: {:.3f}% novelty: {:.3f}% abs novelty: {:.3f}%"
            .format(valid_res['valid_ratio'], valid_res['unique_ratio'],
                    valid_res['abs_unique_ratio'], novel_r, abs_novel_r))

        valid_smiles_qed = [qed(mol) for mol in valid_res['valid_mols']]
        valid_smiles_logp = [MolLogP(mol) for mol in valid_res['valid_mols']]
        top_qed_index = np.argsort(valid_smiles_qed)[::-1]
        top_logp_index = np.argsort(valid_smiles_logp)[::-1]
        top_qed_mols = []
        top_qed_values = []
        top_logp_mols = []
        top_logp_values = []
        for i in range(100):
            top_qed_mols.append(valid_res['valid_mols'][top_qed_index[i]])
            top_qed_values.append('{:.3f}'.format(
                valid_smiles_qed[top_qed_index[i]]))
            top_logp_mols.append(valid_res['valid_mols'][top_logp_index[i]])
            top_logp_values.append('{:.3f}'.format(
                valid_smiles_logp[top_logp_index[i]]))

        # svg = Draw.MolsToGridImage(
        #     top_qed_mols,
        #     legends=top_qed_values,
        #     molsPerRow=10,  # 5,
        #     subImgSize=(240, 240),
        #     useSVG=True,
        #     maxFontSize=24)
        #
        # cairosvg.svg2pdf(bytestring=svg.encode('utf-8'),
        #                  write_to=os.path.join(args.save_dir, "top_qed.pdf"))
        # cairosvg.svg2pdf(bytestring=svg.encode('utf-8'),
        #                  write_to=os.path.join(args.save_dir, "top_qed.png"))
        # svg = Draw.MolsToGridImage(
        #     top_logp_mols,
        #     legends=top_logp_values,
        #     molsPerRow=10,  # 5,
        #     subImgSize=(240, 240),
        #     useSVG=True)  # , useSVG=True
        #
        # cairosvg.svg2pdf(bytestring=svg.encode('utf-8'),
        #                  write_to=os.path.join(args.save_dir, "top_logp.pdf"))
        # cairosvg.svg2pdf(bytestring=svg.encode('utf-8'),
        #                  write_to=os.path.join(args.save_dir, "top_logp.png"))

    test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='moses', help='choose data name')
    parser.add_argument('--save_dir', type=str, help='choose save dir')
    args = parser.parse_args()
    args.save_dir = args.save_dir + '_' + args.data_name
    train(args)
