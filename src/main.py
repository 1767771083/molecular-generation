import torch
from torch import nn

from src.config_parser import get_args
from src.model import GraphDiffusionModel, p_losses, make_beta_schedule
import os
import logging
from data.data_loader import get_data_loader
from src.test import sample
from src.utils import get_data_par


def train(model, train_dataloader, optimizer, betas, device):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_dataloader):
        x_start = batch[0].to(device)
        adj_start = batch[1].to(device)
        t = torch.randint(0, model.time_steps, (x_start.shape[0],), device=device)
        loss = p_losses(model, x_start, adj_start, t, betas)
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_dataloader)

def main(args):
    os.makedirs(args.result_dir, exist_ok=True)
    log_file = os.path.join(args.result_dir, f"{args.data_name}_outputs.txt")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device: {}".format(device))

    data_file, transform_fn, atomic_num_list, \
    b_n_type, a_n_node, a_n_type, valid_idx = get_data_par(args.data_name)

    train_dataloader, test_dataloader, train_res = get_data_loader(args, data_file, transform_fn,
                                                                   atomic_num_list, valid_idx, split=True)
    logging.info('==========================================')
    logging.info('Dataset: {}'.format(args.data_name))
    logging.info('Max Epoch: {}'.format(args.max_epochs))
    logging.info('Batch size: {}'.format(args.batch_size))
    logging.info('==========================================')

    # 模型
    model = GraphDiffusionModel(node_num=a_n_node, node_feature_dim=a_n_type,
                                hidden_dim=args.hidden_dim, time_emb_dim=args.time_emb_dim, edge_type_num=4).to(device)
    model.time_steps = args.total_time_step
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # beta调度器
    betas = make_beta_schedule()
    betas = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in betas.items()}

    # 训练
    best_loss = float('inf')
    best_model_path = os.path.join(args.model_dir, f'{args.data_name}_model.pt')

    for epoch in range(args.max_epochs):
        epoch_loss = train(model, train_dataloader, optimizer, betas, device)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), best_model_path)

        logging.info(f"Epoch {epoch + 1}/{args.max_epochs}, Loss: {epoch_loss:.6f}")

    sample(args)


if __name__ == '__main__':
    args = get_args()
    main(args)
