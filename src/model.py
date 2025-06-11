import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn


class GraphDiffusionModel(nn.Module):
    def __init__(self, node_num, node_feature_dim, hidden_dim, time_emb_dim, edge_type_num=4):
        super(GraphDiffusionModel, self).__init__()
        self.node_dim = node_feature_dim
        self.node_num = node_num
        self.hidden_dim = hidden_dim
        self.time_emb_dim = time_emb_dim
        self.edge_type_num = edge_type_num
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, time_emb_dim)
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(node_feature_dim + edge_type_num + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_feature_dim)
        )

        self.adj_mlp = nn.Sequential(
            nn.Linear(node_feature_dim + time_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, edge_type_num * node_num)
        )

    def forward(self, x, adj, t):
        # x: [B, N, node_dim]
        # adj: [B, edge_type_num, N, N]
        # t: [B]

        B, N, _ = x.shape

        t = t.view(B, 1).float() / 1000
        t_emb = self.time_mlp(t).unsqueeze(1).repeat(1, N, 1)

        adj_sum = adj.sum(dim=3).permute(0, 2, 1)   # [B, N, edge_type_num]
        h = torch.cat([x, adj_sum, t_emb], dim=-1)    # [B, N, node_dim + edge_type_num + time_emb_dim]

        x_gen = self.node_mlp(h)  # [B, N, node_dim]

        adj_input = torch.cat([x_gen, t_emb], dim=-1)  # [B, N, node_dim + time_emb_dim]
        adj_pred = self.adj_mlp(adj_input)  # [B, N, edge_type_num * node_dim]
        adj_gen = adj_pred.view(B, N, self.edge_type_num, N).permute(0, 2, 1, 3)  # [B, edge_type_num, N, N]

        return x_gen, adj_gen


def q_sample(x_start, adj_start, t, noise_x=None, noise_adj=None, betas=None):
    """
    对 x 和 adj 进行扩散。
    x_start: 初始节点特征 [B, N, D]
    adj_start: 初始邻接矩阵 [B, E, N, N]
    t: 时间步 [B]
    noise_x: 噪声（可选） [B, N, D]
    noise_adj: 噪声（可选） [B, E, N, N]
    """
    if noise_x is None:
        noise_x = torch.randn_like(x_start)
    if noise_adj is None:
        noise_adj = torch.randn_like(adj_start)

    sqrt_alphas_cumprod = torch.sqrt(betas['alphas_cumprod'][t])[:, None, None]
    sqrt_one_minus = torch.sqrt(1 - betas['alphas_cumprod'][t])[:, None, None]

    x_t = sqrt_alphas_cumprod * x_start + sqrt_one_minus * noise_x
    adj_t = sqrt_alphas_cumprod[:, :, None] * adj_start + sqrt_one_minus[:, :, None] * noise_adj
    return x_t, adj_t

@torch.no_grad()
def p_sample_loop(model, shape, timesteps, betas):
    B, N, D = shape
    E = 4  # edge_type_num
    torch.manual_seed(42)
    if next(model.parameters()).is_cuda:
        torch.cuda.manual_seed_all(42)
    x_t = torch.randn(B, N, D).to(model.device)
    adj_t = torch.randn(B, E, N, N).to(model.device)

    for t in reversed(range(timesteps)):
        t_batch = torch.full((B,), t, device=model.device, dtype=torch.long)
        x_pred, adj_pred = model(x_t, adj_t, t_batch)

        beta_t = betas['betas'][t]
        sqrt_one_minus = torch.sqrt(1 - betas['alphas_cumprod'][t])
        sqrt_recip = 1 / torch.sqrt(betas['alphas'][t])

        x_t = sqrt_recip * (x_t - beta_t / sqrt_one_minus * x_pred)
        adj_t = sqrt_recip * (adj_t - beta_t / sqrt_one_minus * adj_pred)

    return x_t, adj_t

def p_losses(model, x_start, adj_start, t, betas):
    noise_x = torch.randn_like(x_start)
    noise_adj = torch.randn_like(adj_start)
    x_noisy, adj_noisy = q_sample(x_start, adj_start, t, noise_x, noise_adj, betas)
    x_recon, adj_recon = model(x_noisy, adj_noisy, t)
    loss = F.mse_loss(x_recon, noise_x) + F.mse_loss(adj_recon, noise_adj)
    return loss


def make_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    return {
        'T': T,
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod
    }
