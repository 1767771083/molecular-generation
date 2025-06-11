import os
import time
import torch
from matplotlib import pyplot as plt
from torch import nn, cosine_similarity

from src.model import LatentDecoder, GraphEncoder, Predictor
from data.data_loader import get_data_loader
from src.utils import get_real_label, get_data_par

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'
import matplotlib
matplotlib.use('agg')
import functools
from config_parser import get_args
print = functools.partial(print, flush=True)

def train(args):
    start = time.time()
    print("Start at Time: {}".format(time.ctime()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    data_file, transform_fn, atomic_num_list, \
        b_n_type, a_n_node, a_n_type, valid_idx = get_data_par(args.fine_data_name)

    data_dataloader, test_dataloader, train_res = get_data_loader(args, data_file, transform_fn,
                                                                    atomic_num_list, valid_idx, split=True)

    print('==========================================')
    print('Fine-Tuning')
    print('Load {} data done! Time {:.2f} seconds'.format(args.fine_data_name, time.time() - start))
    print('Max Epoch: {}'.format(args.max_epochs))
    print('Num Iter/Epoch: {}'.format(len(data_dataloader)))
    print('Batch size: {}'.format(args.batch_size))
    print('==========================================')

    graph_encoder = GraphEncoder(a_n_type, args.hidden_size, args.hidden_size, b_n_type - 1,
                                 args.latent_size).to(device)
    latent_decoder = LatentDecoder(args.latent_size, a_n_node, b_n_type, a_n_type).to(device)

    predictor = Predictor(args.latent_size, args.num_encoder, args.num_heads, args.num_ffn, args.num_neurons).to(device)

    graph_encoder_optimizer = torch.optim.Adam(graph_encoder.parameters(),
                                               lr=1e-4)
    latent_decoder_optimizer = torch.optim.Adam(latent_decoder.parameters(),
                                                lr=1e-4)
    predictor_optimizer = torch.optim.Adam(predictor.parameters(), lr=1e-4)

    criterion_c = nn.CrossEntropyLoss()
    criterion_m = nn.MSELoss()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    iter_per_epoch = len(data_dataloader)
    best_loss = float('inf')
    reconstruction_losses = []
    predictor_losses = []

    graph_encoder.train()
    latent_decoder.train()
    predictor.train()

    for epoch in range(args.tuning_epochs):
        print("In epoch {}, Time: {}".format(epoch + 1, time.ctime()))

        total_reconstruction_loss = 0
        total_predictor_loss = 0

        for i, batch in enumerate(data_dataloader):

            x = batch[0].to(device)
            adj = batch[1].to(device)
            label = batch[2].to(device).float()
            # ['pIC50', 'QED', 'SAS']
            real_label = get_real_label(label, args.fine_data_name, property='both_tuning').to(device).float()

            graph_encoder_output = graph_encoder(x, adj)
            graph_decoder_node, graph_decoder_adj = latent_decoder(
                graph_encoder_output, gumbel_softmax=False)

            loss_node = criterion_c(graph_decoder_node.view(-1, graph_decoder_node.size(-1)),
                                    x.argmax(dim=-1).view(-1))
            loss_adj = criterion_m(graph_decoder_adj, adj)
            reconstruction_loss = loss_node + loss_adj

            pred, real_modified_features = predictor(graph_encoder_output)

            qed_predictor_loss = criterion_m(pred[:, 0], real_label[:, 0]).float()
            sa_predictor_loss = criterion_m(pred[:, 1], real_label[:, 1]).float()
            pic_predictor_loss = criterion_m(pred[:, 2], real_label[:, 2]).float()
            predictor_loss = qed_predictor_loss + sa_predictor_loss + pic_predictor_loss

            graph_encoder_optimizer.zero_grad()
            latent_decoder_optimizer.zero_grad()
            reconstruction_loss.backward(retain_graph=True)
            graph_encoder_optimizer.step()
            latent_decoder_optimizer.step()

            predictor_optimizer.zero_grad()
            predictor_loss.backward()
            predictor_optimizer.step()

            total_reconstruction_loss += reconstruction_loss.item()
            total_predictor_loss += predictor_loss.item()

        reconstruction_losses.append(total_reconstruction_loss / len(data_dataloader))
        predictor_losses.append(total_predictor_loss / len(data_dataloader))

        print(
            'Epoch [{}/{}], Iter [{}/{}], Reconstruction Loss:{:}, Predictor Loss:{:}'
            .format(epoch + 1, args.tuning_epochs, i + 1, iter_per_epoch,
                    total_reconstruction_loss / len(data_dataloader),
                    total_predictor_loss / len(data_dataloader)))

        if (total_reconstruction_loss + total_predictor_loss)/len(data_dataloader) < best_loss:
            best_loss = (total_reconstruction_loss + total_predictor_loss)/len(data_dataloader)
            torch.save(graph_encoder.state_dict(),
                       os.path.join(args.model_dir, '{}_pretrain_graph_encoder.pth'.format(args.fine_data_name)))
            torch.save(latent_decoder.state_dict(),
                       os.path.join(args.model_dir, '{}_pretrain_latent_decoder.pth'.format(args.fine_data_name)))
            torch.save(predictor.state_dict(),
                       os.path.join(args.model_dir, '{}_pretrain_predictor.pth'.format(args.fine_data_name)))

    def test():
        graph_encoder.eval()
        latent_decoder.eval()
        predictor.eval()

        total_reconstruction_loss = 0
        total_predictor_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                x = batch[0].to(device)
                adj = batch[1].to(device)
                label = batch[2].to(device).float()
                real_label = get_real_label(label, args.fine_data_name, property='both_tuning').to(device).float()

                graph_encoder_output = graph_encoder(x, adj)
                graph_decoder_node, graph_decoder_adj = latent_decoder(graph_encoder_output, gumbel_softmax=True)

                loss_node = criterion_c(graph_decoder_node.view(-1, graph_decoder_node.size(-1)),
                                        x.argmax(dim=-1).view(-1))
                loss_adj = criterion_m(graph_decoder_adj, adj)
                reconstruction_loss = loss_node + loss_adj

                pred, real_modified_features = predictor(graph_encoder_output)

                qed_predictor_loss = criterion_m(pred[:, 0], real_label[:, 0]).float()
                sa_predictor_loss = criterion_m(pred[:, 1], real_label[:, 1]).float()
                pic_predictor_loss = criterion_m(pred[:, 2], real_label[:, 2]).float()
                predictor_loss = qed_predictor_loss + sa_predictor_loss + pic_predictor_loss

                total_reconstruction_loss += reconstruction_loss.item()
                total_predictor_loss += predictor_loss.item()


        avg_reconstruction_loss = total_reconstruction_loss / len(test_dataloader)
        avg_predictor_loss = total_predictor_loss / len(test_dataloader)

        print('==========================================')
        print('Testing Results')
        print('Average Reconstruction Loss: {:.4f}'.format(avg_reconstruction_loss))
        print('Average Predictor Loss: {:.4f}'.format(avg_predictor_loss))
        print('==========================================')

    test()

    loss_subdir = 'loss'
    loss_dir = os.path.join(args.result_dir, loss_subdir)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(reconstruction_losses, label='Reconstruction Loss')
    plt.plot(predictor_losses, label='Predictor Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAE and Predictor Losses')
    # Save the plot
    plt.savefig(os.path.join(loss_dir, f"{args.fine_data_name}_Pretrain_GAE_Predictor_losses.png"))


if __name__ == '__main__':

    args = get_args()

    train(args)
