import os
import time
import torch
from matplotlib import pyplot as plt
from torch import nn

from src.model import GraphEncoder, Predictor, proPredictor
from data.data_loader import get_data_loader
from src.utils import get_real_label, get_data_par, get_pred_data_par
from torch.optim.lr_scheduler import StepLR

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

    # Generation
    data_file, transform_fn, atomic_num_list, \
        b_n_type, a_n_node, a_n_type, valid_idx = get_data_par(args.data_name)

    # Prediction
    # data_file, transform_fn, atomic_num_list, \
    # b_n_type, a_n_node, a_n_type, valid_idx = get_pred_data_par(args.data_name)

    data_dataloader, test_dataloader, train_res = get_data_loader(args, data_file, transform_fn,
                                                                  atomic_num_list, valid_idx, split=True)

    print('==========================================')
    print('Pretrain Predictor')
    print('Load {} data done! Time {:.2f} seconds'.format(args.data_name, time.time() - start))
    print('Max Epoch: {}'.format(args.max_epochs))
    print('Num Iter/Epoch: {}'.format(len(data_dataloader)))
    print('Batch size: {}'.format(args.batch_size))
    print('==========================================')

    graph_encoder = GraphEncoder(a_n_type, args.hidden_size, args.hidden_size, b_n_type - 1,
                                 args.latent_size).to(device)

    predictor = Predictor(args.latent_size, args.num_encoder, args.num_heads, args.num_ffn, args.num_neurons).to(device)

    propredictor = proPredictor(args.latent_size, args.num_encoder, args.num_heads, args.num_ffn, args.num_neurons).to(
        device)

    predictor_optimizer = torch.optim.Adam(propredictor.parameters(), lr=1e-3)

    criterion_m = nn.MSELoss()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    iter_per_epoch = len(data_dataloader)
    best_loss = float('inf')
    predictor_losses = []

    graph_encoder.load_state_dict(
        torch.load('{}/{}_pretrain_graph_encoder.pth'.format(args.model_dir, args.data_name)))
    propredictor.train()
    for epoch in range(args.pretrain_epochs):
        print("In epoch {}, Time: {}".format(epoch + 1, time.ctime()))
        total_predictor_loss = 0
        for i, batch in enumerate(data_dataloader):
            x = batch[0].to(device)
            adj = batch[1].to(device)
            label = batch[2].to(device).float()
            # ['pIC50', 'QED', 'SAS']
            real_label = get_real_label(label, args.data_name, property='pic').to(device).float()

            graph_encoder_output = graph_encoder(x, adj)

            pred, real_modified_features = propredictor(graph_encoder_output)

            predictor_loss = criterion_m(pred, real_label).float()

            predictor_optimizer.zero_grad()
            predictor_loss.backward()
            predictor_optimizer.step()

            total_predictor_loss += predictor_loss.item()

        predictor_losses.append(total_predictor_loss / len(data_dataloader))

        print(
            'Epoch [{}/{}], Iter [{}/{}], Predictor Loss:{:}'
                .format(epoch + 1, args.pretrain_epochs, i + 1, iter_per_epoch,
                        total_predictor_loss / len(data_dataloader)))

        if total_predictor_loss / len(data_dataloader) < best_loss:
            best_loss = total_predictor_loss / len(data_dataloader)
            torch.save(propredictor.state_dict(),
                       os.path.join(args.model_dir, '{}_pretrain_propredictor.pth'.format(args.data_name)))

    def test():
        graph_encoder.eval()
        propredictor.eval()

        total_predictor_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                x = batch[0].to(device)
                adj = batch[1].to(device)
                label = batch[2].to(device).float()
                real_label = get_real_label(label, args.data_name, property='pic').to(device).float()

                graph_encoder_output = graph_encoder(x, adj)

                pred, real_modified_features = propredictor(graph_encoder_output)

                predictor_loss = criterion_m(pred, real_label).float()

                total_predictor_loss += predictor_loss.item()

        avg_predictor_loss = total_predictor_loss / len(test_dataloader)

        print('==========================================')
        print('Testing Results')
        print('Average Predictor Loss: {:.4f}'.format(avg_predictor_loss))
        print('==========================================')

    test()

    loss_subdir = 'loss'
    loss_dir = os.path.join(args.result_dir, loss_subdir)
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    plt.figure(figsize=(10, 6))
    plt.plot(predictor_losses, label='Predictor Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Predictor Losses')
    # Save the plot
    plt.savefig(os.path.join(loss_dir, f"{args.data_name}_Pretrain_proPredictor_losses.png"))


if __name__ == '__main__':
    args = get_args()

    train(args)
