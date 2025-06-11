import logging
import random

import numpy as np
import torch
from rdkit.Chem.QED import qed

from src.data.data_loader import get_data_loader
from src.model import GraphDiffusionModel, make_beta_schedule, p_sample_loop
from src.config_parser import get_args
import os

from src.utils import get_data_par, calculate_sa_score, check_novelty, check_validity


@torch.no_grad()
def sample(args, idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = idx
    random.seed(seed)
    np.random.seed(seed)

    # log_path = os.path.join(args.result_dir, f"{args.data_name}_result.log")
    # logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s %(message)s')
    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)
    # logging.getLogger().addHandler(console)

    data_file, transform_fn, atomic_num_list, \
    b_n_type, a_n_node, a_n_type, valid_idx = get_data_par(args.data_name)

    train_dataloader, test_dataloader, train_res = get_data_loader(args, data_file, transform_fn,
                                                                   atomic_num_list, valid_idx, split=True)

    model = GraphDiffusionModel(node_num=a_n_node, node_feature_dim=a_n_type,
                                hidden_dim=args.hidden_dim, time_emb_dim=args.time_emb_dim, edge_type_num=4).to(device)
    model.time_steps = args.total_time_step

    # 加载训练好的模型参数
    model_path = os.path.join(args.model_dir, f'{args.data_name}_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.device = device
    model.eval()

    betas = make_beta_schedule(args.total_time_step)
    betas = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in betas.items()}

    # 指定生成样本的形状
    B, N, D = 10000, a_n_node, a_n_type
    x_gen, adj_gen = p_sample_loop(model, (B, N, D), args.total_time_step, betas)

    # print("生成的节点特征：", x_gen.shape)
    # print("生成的邻接矩阵：", adj_gen.shape)

    valid_res = check_validity(adj_gen, x_gen, atomic_num_list)
    valid_mols = valid_res['valid_mols']
    novel_r, abs_novel_r = check_novelty(valid_res['valid_smiles'],
                                         train_res['valid_smiles'],
                                         x_gen.shape[0])


    print(
        "valid: {:.3f}%, unique: {:.3f}%, novelty: {:.3f}%".format(
            valid_res['valid_ratio'],
            valid_res['unique_ratio'],
            novel_r,
        )
    )
    result_path = os.path.join(args.result_dir, f"{args.data_name}/{args.data_name}_result_{idx}.txt")
    with open(result_path, 'w') as f:
        f.write(f"valid: {valid_res['valid_ratio']:.3f}%\n")
        f.write(f"unique: {valid_res['unique_ratio']:.3f}%\n")
        f.write(f"novelty: {novel_r:.3f}%\n")

if __name__ == '__main__':
    args = get_args()
    # sample(args)
    for idx in range(50):
        print(f"Running sample with {idx}...")
        sample(args, idx)

    print("All tests completed.")

