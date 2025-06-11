import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features,
                                                     out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        support = input @ self.weight
        output = adj @ support
        return output


class MultiGCN(nn.Module):
    def __init__(self, num_features, hidden_size, num_edge_types):
        super().__init__()
        self.convs = nn.ModuleList([
            GCNConv(num_features, hidden_size) for _ in range(num_edge_types)
        ])
        self.ln = nn.LayerNorm(hidden_size * num_edge_types)
        self.fc = nn.Linear(hidden_size * num_edge_types, hidden_size)

    def forward(self, x, edge_index):
        x = [conv(x, edge_index[:, i]) for i, conv in enumerate(self.convs)]
        x = torch.cat(x, dim=-1)  # Concatenate along the feature dimension
        x = self.ln(x)
        x = F.gelu(x)
        x = self.fc(x)
        return x


class TrueEncoder(nn.Module):
    def __init__(self, num_features, embedding_num, hidden_size,
                 num_edge_types, latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_num)
        self.multigcn_embedding = MultiGCN(embedding_num, hidden_size,
                                           num_edge_types)
        self.multigcn_backbone = MultiGCN(hidden_size, hidden_size,
                                          num_edge_types)
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x, edge_index):
        # x: Input features (num_nodes, num_features)
        # edge_index: List of edge_index tensors for each edge type
        x = x @ self.embedding.weight
        x = self.multigcn_embedding(x, edge_index)
        x = self.multigcn_backbone(x, edge_index)
        x = self.fc(x)
        x = torch.mean(x, dim=1)  # Aggregate over nodes
        return x


class FakeEncoder(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.GELU(),
            nn.Linear(latent_size, latent_size),
            nn.GELU(),
            nn.Linear(latent_size, latent_size),
        )

    def forward(self, z):
        return self.mlp(z)


class LatentDecoder(nn.Module):
    def __init__(self, latent_size, num_nodes, num_edge_types, node_feat_size):
        super().__init__()
        self.latent_size = latent_size
        self.num_nodes = num_nodes
        self.num_edge_types = num_edge_types
        self.node_feat_size = node_feat_size

        # Define multi-layer perceptron (MLP) for generating node features
        self.mlp_feat = nn.Sequential(
            nn.Linear(latent_size, latent_size), nn.GELU(),
            nn.Linear(latent_size, num_nodes * node_feat_size), nn.GELU(),
            nn.Linear(num_nodes * node_feat_size, num_nodes * node_feat_size))

        # Define multi-layer perceptron (MLP) for generating adjacency matrices
        self.mlp_adj = nn.Sequential(
            nn.Linear(latent_size, latent_size), nn.GELU(),
            nn.Linear(latent_size, num_edge_types * num_nodes * num_nodes),
            nn.GELU(),
            nn.Linear(num_edge_types * num_nodes * num_nodes,
                      num_edge_types * num_nodes * num_nodes))

    def forward(self, z, gumbel_softmax=False):
        # z: latent vector (latent_size,)

        # Generate node features
        node_features = self.mlp_feat(z)
        node_features = node_features.view(node_features.shape[0],
                                           self.num_nodes, self.node_feat_size)
        if gumbel_softmax:
            node_features = F.gumbel_softmax(node_features, hard=True, dim=-1)
        # Generate adjacency matrices
        adj_matrices = self.mlp_adj(z)
        adj_matrices = adj_matrices.view(adj_matrices.shape[0],
                                         self.num_edge_types, self.num_nodes,
                                         self.num_nodes)
        adj_matrices = (adj_matrices + adj_matrices.transpose(3, 2)) / 2
        if gumbel_softmax:
            adj_matrices = F.gumbel_softmax(adj_matrices, hard=True, dim=1)
        return node_features, adj_matrices


class Discriminator(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.total_time_step = 1000
        self.alpha_t_bar, self.beta_t_bar = self.get_alpha_t_beta_t(
            self.total_time_step)
        self.time_step_embedding = nn.Embedding(self.total_time_step,
                                                latent_size)
        self.mlp = nn.Sequential(
            nn.Linear(latent_size * 2, latent_size * 2),
            nn.GELU(),
            nn.Linear(latent_size * 2, latent_size * 2),
            nn.GELU(),
            nn.Linear(latent_size * 2, 1),
        )

    def get_alpha_t_beta_t(self, total_time_step):
        t = torch.arange(total_time_step)
        alpha_t = torch.sqrt(1 - 0.02 * t / total_time_step)
        alpha_t_bar = torch.cumprod(alpha_t, dim=-1)
        beta_t_bar = torch.sqrt(1 - alpha_t_bar**2)
        return alpha_t_bar, beta_t_bar

    def forward(self, x, time_step):
        x = self.alpha_t_bar.to(x.device)[time_step] * x + self.beta_t_bar.to(
            x.device)[time_step] * torch.randn_like(x)
        time_step_embedding = self.time_step_embedding(time_step.squeeze(dim=1))
        x = self.mlp(
            torch.cat([x, time_step_embedding], dim=-1))
        return x
