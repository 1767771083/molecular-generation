import os
import torch
import torch.nn as nn
from src.model import LatentDecoder, Discriminator, GraphEncoder, Generator, Predictor, proPredictor
from data.data_loader import get_data_loader
import time
from src.utils import check_validity, save_mol_png, check_novelty, get_real_label, get_fake_label, \
    compute_gradient_penalty, get_data_par, \
    get_loss_curve, calculate_sa_score, simplified_function
from rdkit.Chem import Draw
import numpy as np
from rdkit.Chem.Descriptors import qed
from rdkit.Chem.Crippen import MolLogP
import matplotlib
import matplotlib.pyplot as plt
import functools

matplotlib.use('agg')

print = functools.partial(print, flush=True)

def train(args):
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_file, transform_fn, atomic_num_list, \
    b_n_type, a_n_node, a_n_type, valid_idx = get_data_par(args.data_name)

    train_dataloader, test_dataloader, train_res = get_data_loader(args, data_file, transform_fn,
                                                                   atomic_num_list, valid_idx, split=True)

    print('==========================================')
    print('Load {} data done! Time {:.2f} seconds'.format(args.data_name, time.time() - start))
    print('Max Epoch: {}'.format(args.max_epochs))
    print('Num Iter/Epoch: {}'.format(len(train_dataloader)))
    print('Batch size: {}'.format(args.batch_size))
    print('Weight of property: {}'.format(args.attr_weight_1))
    print('==========================================')

    # Loss and optimizer

    graph_encoder = GraphEncoder(a_n_type, args.hidden_size, args.hidden_size, b_n_type - 1,
                                 args.latent_size).to(device)
    latent_decoder = LatentDecoder(args.latent_size, a_n_node, b_n_type, a_n_type).to(device)

    generator = Generator(args.latent_size, a_n_node, args.latent_size).to(device)

    discriminator = Discriminator(a_n_node, args.latent_size, args.total_time_step).to(device)

    propredictor = proPredictor(args.latent_size, args.num_encoder, args.num_heads, args.num_ffn, args.num_neurons).to(device)

    generator_optimizer = torch.optim.Adam(generator.parameters(),
                                           lr=2e-4)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(),
                                               lr=1e-4)
    predictor_optimizer = torch.optim.Adam(propredictor.parameters(), lr=1e-6)

    criterion_m = nn.MSELoss()

    # Train the models
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    iter_per_epoch = len(train_dataloader)
    best_loss = float('inf')
    discriminator_losses = []
    generator_losses = []
    predictor_losses = []

    # Step 2: Train GAN and fine-tune the autoencoder and predictor
    graph_encoder.load_state_dict(
        torch.load('{}/{}_pretrain_graph_encoder.pth'.format(args.model_dir, args.data_name)))
    latent_decoder.load_state_dict(
        torch.load('{}/{}_pretrain_latent_decoder.pth'.format(args.model_dir, args.data_name)))
    propredictor.load_state_dict(torch.load('{}/{}_pretrain_propredictor.pth'.format(args.model_dir, args.data_name)))

    discriminator.train()
    generator.train()
    propredictor.train()
    print('Train start')
    for epoch in range(args.max_epochs):
        print("In epoch {}, Time: {}".format(epoch + 1, time.ctime()))

        total_discriminator_loss = 0
        total_generator_loss = 0
        total_predictor_loss = 0

        for param in graph_encoder.parameters():
            param.requires_grad = False
        for param in latent_decoder.parameters():
            param.requires_grad = False

        for i, batch in enumerate(train_dataloader):

            x = batch[0].to(device)
            adj = batch[1].to(device)
            label = batch[2].to(device).float()   # ['LogP/pIC50', 'QED', 'SAS']

            real_label = get_real_label(label, args.data_name, property='pic').to(device).float()
            # ['QED', 'SAS', '/pIC50']
            # (qed, sas, pic, both_train, both_tuning)

            pic_max = torch.max(real_label).item()
            pic_min = torch.min(real_label).item()

            graph_encoder_output = graph_encoder(x, adj)

            fake_label = np.random.uniform(low=pic_min, high=pic_max, size=(x.shape[0], 1))
            fake_label = torch.from_numpy(fake_label).float().to(device)

            for _ in range(args.n_critic):
                noise = torch.randn(x.shape[0], args.latent_size, device=device).float()
                generator_output = generator(noise)
                # Predictor forward and loss calculation
                pred_real, real_modified_features = propredictor(graph_encoder_output)
                pred_fake, fake_modified_features = propredictor(generator_output)
                true_discriminator_outputs = discriminator(
                    real_modified_features,
                    torch.randint(low=0,
                                  high=discriminator.total_time_step,
                                  size=(x.shape[0], 1),
                                  device=device))

                fake_discriminator_outputs = discriminator(
                    fake_modified_features.detach(),
                    torch.randint(low=0,
                                  high=discriminator.total_time_step,
                                  size=(x.shape[0], 1),
                                  device=device))

                # Discriminator Loss
                gradient_penalty = compute_gradient_penalty(discriminator,
                                                            real_modified_features, fake_modified_features)

                discriminator_loss = torch.mean(fake_discriminator_outputs) - torch.mean(true_discriminator_outputs) \
                                     + args.lambda_gp * gradient_penalty

                # Update Discriminator
                discriminator_optimizer.zero_grad()
                discriminator_loss.backward(retain_graph=True)
                discriminator_optimizer.step()

                total_discriminator_loss += discriminator_loss.item()

            # Update Generator
            noise = torch.randn(x.shape[0], args.latent_size, device=device).float()
            generator_output = generator(noise)
            pred_real, real_modified_features = propredictor(graph_encoder_output)
            pred_fake, fake_modified_features = propredictor(generator_output)

            fake_discriminator_outputs = discriminator(
                fake_modified_features,
                torch.randint(low=0,
                              high=discriminator.total_time_step,
                              size=(x.shape[0], 1),
                              device=device))

            pred_pic = torch.mean(pred_fake)

            generator_loss = -torch.mean(fake_discriminator_outputs) - args.attr_weight_1 * pred_pic
            # + args.attr_weight_2 * (pic_max - pred_pic)

            generator_optimizer.zero_grad()
            generator_loss.backward(retain_graph=True)
            generator_optimizer.step()

            # Predictor Loss
            predictor_loss_real = criterion_m(pred_real, real_label).float()

            predictor_loss_fake = criterion_m(pred_fake, fake_label).float()

            predictor_loss = 0.9 * predictor_loss_real + 0.1 * predictor_loss_fake

            # Update Predictor
            predictor_optimizer.zero_grad()
            predictor_loss.backward()
            predictor_optimizer.step()

            total_generator_loss += generator_loss.item()
            total_predictor_loss += predictor_loss.item()

        discriminator_losses.append(total_discriminator_loss / len(train_dataloader))
        generator_losses.append(total_generator_loss / len(train_dataloader))
        predictor_losses.append(total_predictor_loss / len(train_dataloader))

        print(
            'Epoch [{}/{}], Iter [{}/{}], '
            'Discriminator Loss:{:}, Generator Loss:{:}, Predictor Loss:{:}'
                .format(epoch + 1, args.max_epochs, i + 1, iter_per_epoch,
                        total_discriminator_loss / len(train_dataloader),
                        total_generator_loss / len(train_dataloader),
                        total_predictor_loss / len(train_dataloader)))

        # 保存模型参数
        if (total_generator_loss + total_discriminator_loss + total_predictor_loss) / len(train_dataloader) < best_loss:
            best_loss = (total_generator_loss + total_discriminator_loss +
                         total_predictor_loss) / len(train_dataloader)

            torch.save(generator.state_dict(), os.path.join(args.model_dir, '{}_generator_pic.pth'.format(args.data_name)))
            torch.save(discriminator.state_dict(),
                       os.path.join(args.model_dir, '{}_discriminator_pic.pth'.format(args.data_name)))
            torch.save(graph_encoder.state_dict(),
                       os.path.join(args.model_dir, '{}_graph_encoder_pic.pth'.format(args.data_name)))
            torch.save(latent_decoder.state_dict(),
                       os.path.join(args.model_dir, '{}_latent_decoder_pic.pth'.format(args.data_name)))
            torch.save(propredictor.state_dict(), os.path.join(args.model_dir, '{}_predictor_pic.pth'.format(args.data_name)))

        def print_validity(ith):
            with torch.no_grad():
                generator.eval()
                latent_decoder.eval()

                noise = torch.randn(100, args.latent_size, device=device).float()

                generated_features = generator(noise)
                x, adj = latent_decoder(generated_features, gumbel_softmax=True)
                valid_res = check_validity(adj, x, atomic_num_list)
                valid_mols = valid_res['valid_mols']
                novel_r, abs_novel_r = check_novelty(valid_res['valid_smiles'],
                                                     train_res['valid_smiles'],
                                                     x.shape[0])

                eval_smiles_qed = [qed(mol) for mol in valid_res['valid_mols']]
                eval_smiles_sa = [calculate_sa_score(mol) for mol in valid_res['valid_mols']]

                print(
                    "valid: {:.3f}%, unique: {:.3f}%, abs unique: {:.3f}% novelty: {:.3f}% abs novelty: {:.3f}%, \n"
                    "Mean QED:{:.3f}, Mean SA:{:.3f}"
                        .format(valid_res['valid_ratio'],
                                valid_res['unique_ratio'],
                                valid_res['abs_unique_ratio'], novel_r, abs_novel_r,
                                np.mean(eval_smiles_qed), np.mean(eval_smiles_sa))
                )
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
                generator.eval()
                latent_decoder.eval()
                z = torch.randn(100, args.latent_size, device=device).float()
                generated_features = generator(z)
                x, adj = latent_decoder(generated_features, gumbel_softmax=True)
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

        with open(f'{args.result_dir}/{args.attr_weight_1}_{args.data_name}_pic_generated_smiles.txt', 'w') as file:
            for smile in valid_res['valid_smiles']:
                file.write(smile + '\n')

        print(
            "valid: {:.3f}%, unique: {:.3f}%, abs unique: {:.3f}% novelty: {:.3f}% abs novelty: {:.3f}%"
                .format(valid_res['valid_ratio'], valid_res['unique_ratio'],
                        valid_res['abs_unique_ratio'], novel_r, abs_novel_r))

        # cal property
        print('===================================================')
        test_smiles_qed = [qed(mol) for mol in valid_res['valid_mols']]
        print(
            'test max QED:{:.3f}'.format(np.max(test_smiles_qed)), '\n',
            'test mean QED:{:.3f}'.format(np.mean(test_smiles_qed))
        )

        test_smiles_logp = [MolLogP(mol) for mol in valid_res['valid_mols']]
        print(
            'test min LogP:{:.3f}'.format(np.min(test_smiles_logp)), '\n',
            'test mean LogP:{:.3f}'.format(np.mean(test_smiles_logp))
        )

        test_smiles_sa = [calculate_sa_score(mol) for mol in valid_res['valid_mols']]
        print(
            'test min SA:{:.3f}'.format(np.min(test_smiles_sa)), '\n',
            'test mean SA:{:.3f}'.format(np.mean(test_smiles_sa))
        )
        print('===================================================')

        valid_smiles_qed = [qed(mol) for mol in valid_res['valid_mols']]
        valid_smiles_logp = [MolLogP(mol) for mol in valid_res['valid_mols']]
        valid_smiles_sa = [calculate_sa_score(mol) for mol in valid_res['valid_mols']]

        top_qed_index = np.argsort(valid_smiles_qed)[::-1]
        top_logp_index = np.argsort(valid_smiles_logp)[::-1]
        top_sa_index = np.argsort(valid_smiles_sa)

        top_qed_mols = []
        top_qed_values = []
        top_logp_mols = []
        top_logp_values = []
        top_sa_mols = []
        top_sa_values = []
        for i in range(20):
            top_qed_mols.append(valid_res['valid_mols'][top_qed_index[i]])
            top_qed_values.append('{:.3f}'.format(
                valid_smiles_qed[top_qed_index[i]]))
            top_logp_mols.append(valid_res['valid_mols'][top_logp_index[i]])
            top_logp_values.append('{:.3f}'.format(
                valid_smiles_logp[top_logp_index[i]]))
            top_sa_mols.append(valid_res['valid_mols'][top_sa_index[i]])
            top_sa_values.append('{:.3f}'.format(
                valid_smiles_sa[top_sa_index[i]]))

        # Save results to a TXT file
        eval_dir = os.path.join(args.result_dir, args.eval_subdir)
        if not os.path.exists(eval_dir):
            os.makedirs(eval_dir)

        with open(os.path.join(eval_dir, f"{args.data_name}_{args.attr_weight_1}_pic_test_results.txt"), "w") as file:
            file.write(f"valid: {valid_res['valid_ratio']:.3f}%\n")
            file.write(f"unique: {valid_res['unique_ratio']:.3f}%\n")
            file.write(f"abs unique: {valid_res['abs_unique_ratio']:.3f}%\n")
            file.write(f"novelty: {novel_r:.3f}%\n")
            file.write(f"abs novelty: {abs_novel_r:.3f}%\n")
            file.write(f"G-Mean: {(np.mean(valid_res['valid_ratio'] + valid_res['unique_ratio'] + novel_r)) / 3:.3f}\n")
            file.write(f"test mean QED: {np.mean(test_smiles_qed):.3f}\n")
            file.write(f"test max QED: {np.max(test_smiles_qed):.3f}\n")
            file.write(f"test mean SA: {np.mean(test_smiles_sa):.3f}\n")
            file.write(f"test min SA: {np.min(test_smiles_sa):.3f}\n")

        print("Results saved to {}_{}_pic_test_results.txt".format(args.data_name, args.attr_weight_1))

        # draw image

    test()

    get_loss_curve(discriminator_losses, generator_losses, predictor_losses,
                   args.data_name, args.result_dir)


from config_parser import get_args

if __name__ == '__main__':
    args = get_args()

    train(args)

