import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.Transformer import EncoderLayer

# ---------- Layer ------------
from src.utils import replace_nan_with_row_mean


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features, edge_type_num, dropout_rate=0.):
        super().__init__()
        self.edge_type_num = edge_type_num
        self.u = out_features
        self.adj_list = nn.ModuleList()
        for _ in range(self.edge_type_num):
            self.adj_list.append(nn.Linear(in_features, out_features))
        self.linear_2 = nn.Linear(in_features, out_features)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, n_tensor, adj_tensor):

        output = torch.stack([self.adj_list[i](n_tensor) for i in range(self.edge_type_num)], 1)
        output = torch.matmul(adj_tensor, output)
        out_sum = torch.sum(output, 1)
        out_linear_2 = self.linear_2(n_tensor)
        output = out_sum + out_linear_2
        output = self.activation(output)
        output = self.dropout(output)
        return output

class MultiGCN(nn.Module):
    def __init__(self, in_features, out_features, num_edge_types, num_layers, dropout_rate=0.3):
        super().__init__()
        self.layers = nn.ModuleList([
            GCNConv(
                in_features if i == 0 else out_features,
                out_features,
                num_edge_types,
                dropout_rate
            ) for i in range(num_layers)
        ])

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)

        return x

# Multi-head attention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model  # for example 512
        self.num_heads = num_heads  # for example for 8 heads
        self.head_dim = d_model // num_heads  # head_dim will be 64
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)  # 512 x 1536
        self.linear_layer = nn.Linear(d_model, d_model)  # 512 x 512

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add a batch dimension at the front
        batch_size, sequence_length, d_model = x.size()  # for example 30 x 200 x 512
        # sequence_length, d_model = x.size()     # for example 30 x 200 x 512
        qkv = self.qkv_layer(x)  # 30 x 200 x 1536
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim)  # 30 x 200 x 8 x 192
        # qkv = qkv.reshape(sequence_length, self.num_heads, 3 * self.head_dim) # 30 x 200 x 8 x 192
        qkv = qkv.permute(0, 2, 1, 3)  # 30 x 8 x 200 x 192
        q, k, v = qkv.chunk(3, dim=-1)  # breakup using the last dimension, each are 30 x 8 x 200 x 64

        values, attention = single_head_attention(q, k, v)
        values = values.reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        # values = values.reshape(sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)

        return out


# Layer normalization
class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-8):
        super().__init__()
        self.eps = eps  # to take care of zero division
        self.parameters_shape = parameters_shape
        self.gamma = nn.Parameter(torch.ones(parameters_shape))  # learnable parameter "std" (512,)
        self.beta = nn.Parameter(torch.zeros(parameters_shape))  # learnable parameter "mean" (512,)

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)  # eg. for (30, 200, 512) inputs, mean -> (30, 200, 1)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)  # (30, 200, 1)
        std = (var + self.eps).sqrt()  # (30, 200, 1)
        y = (inputs - mean) / std  # Normalized output (30, 200, 512)
        out = self.gamma * y + self.beta  # Apply learnable parameters

        return out

# Feedforward MLP
class MLP(nn.Module):
    def __init__(self, d_model, num_ffn, act_fn, dropout_r=0.1):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(d_model, num_ffn)
        self.linear2 = nn.Linear(num_ffn, d_model)
        self.act_fn = act_fn
        self.dropout = nn.Dropout(p=dropout_r)

    def forward(self, x):
        x = self.linear1(self.act_fn(x))
        x = self.dropout(x)
        x = self.linear2(self.act_fn(x))

        return x


def single_head_attention(q, k, v):
    d_k = q.size()[-1]  # 64
    val_before_softmax = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    attention = F.softmax(val_before_softmax, dim=-1)  # 200 x 200
    values = torch.matmul(attention, v)  # 200 x 64
    return values, attention

# ---------- Model ------------

class GraphEncoder(nn.Module):
    def __init__(self, num_features, embedding_num, hidden_size,
                 num_edge_types, latent_dim, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(num_features, embedding_num)
        self.multi_gcn_layer = MultiGCN(embedding_num, hidden_size,
                                        num_edge_types, num_layers)
        # self.agg_layer = MultiGCN(hidden_size, hidden_size,
        #                           num_edge_types, num_layers)
        self.fc = nn.Linear(hidden_size, latent_dim)
        self.act_fn = nn.Tanh()

    def forward(self, x, edge_index):
        # x: Input features (num_nodes, num_features)
        # edge_index: List of edge_index tensors for each edge type
        edge_index = edge_index[:, :-1, :, :]
        x = torch.matmul(x, self.embedding.weight)
        out = self.multi_gcn_layer(x, edge_index)
        # out = self.multigcn_backbone(out, edge_index)
        output = self.fc(out)
        output = self.act_fn(output)
        # x = torch.mean(x, dim=1)  # Aggregate over nodes
        return output


class Generator(nn.Module):
    def __init__(self, latent_size, num_nodes, feature_dim):
        super().__init__()
        self.latent_size = latent_size
        self.mlp = nn.Sequential(
            nn.Linear(latent_size, 2 * latent_size),
            nn.LayerNorm(2 * latent_size),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.02),
            nn.Linear(2 * latent_size, 4 * latent_size),
            nn.LayerNorm(4 * latent_size),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.02),
            nn.Linear(4 * latent_size, num_nodes * feature_dim),
            nn.LayerNorm(num_nodes * feature_dim),
            nn.Tanh()
        )
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim

    def forward(self, z):
        out = self.mlp(z)
        out = out.view(-1, self.num_nodes, self.feature_dim)
        return out


class LatentDecoder(nn.Module):
    def __init__(self, latent_size, num_nodes, num_edge_types, node_feat_size):
        super().__init__()
        self.latent_size = latent_size
        self.num_nodes = num_nodes
        self.num_edge_types = num_edge_types
        self.node_feat_size = node_feat_size

        # Define multi-layer perceptron (MLP) for generating node features
        self.mlp_feat = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.LeakyReLU(0.02),
            nn.Linear(latent_size, 2 * node_feat_size),
            nn.LeakyReLU(0.02),
            nn.Linear(2 * node_feat_size, node_feat_size))

        # Define multi-layer perceptron (MLP) for generating adjacency matrices
        self.mlp_adj = nn.Sequential(
            nn.Linear(num_nodes * latent_size, num_nodes * latent_size),
            nn.LeakyReLU(0.02),
            nn.Linear(num_nodes * latent_size, num_edge_types * num_nodes * num_nodes),
            nn.LeakyReLU(0.02),
            nn.Linear(num_edge_types * num_nodes * num_nodes,
                      num_edge_types * num_nodes * num_nodes))

    def forward(self, z, gumbel_softmax=True):    # gumbel_softmax=False
        # z: latent vector (batch_size, node_num, latent_size)
        # Generate node features
        z_n = z.view(-1, z.shape[-1])
        node_features = self.mlp_feat(z_n)
        node_features = node_features.view(z.shape[0],
                                           self.num_nodes, self.node_feat_size)
        if gumbel_softmax:
            node_features = F.gumbel_softmax(node_features, hard=True, dim=-1)
        # Generate adjacency matrices
        z_a = z.view(z.shape[0], -1)
        adj_matrices = self.mlp_adj(z_a)
        adj_matrices = adj_matrices.view(z.shape[0],
                                         self.num_edge_types, self.num_nodes, self.num_nodes)
        adj_matrices = (adj_matrices + adj_matrices.transpose(3, 2)) / 2
        if gumbel_softmax:
            adj_matrices = F.gumbel_softmax(adj_matrices, hard=True, dim=1)
        return node_features, adj_matrices


class Discriminator(nn.Module):
    def __init__(self, node_num, latent_size, total_time_step):
        super().__init__()
        self.node_num = node_num
        self.latent_size = latent_size
        self.total_time_step = total_time_step
        self.alpha_t_bar, self.beta_t_bar = self.get_alpha_t_beta_t(
            self.total_time_step)
        self.time_step_embedding = nn.Embedding(self.total_time_step,
                                                self.node_num * self.latent_size)
        self.mlp = nn.Sequential(
            nn.Linear(self.node_num * self.latent_size * 2, self.node_num * self.latent_size),
            nn.LayerNorm(self.node_num * self.latent_size),
            nn.LeakyReLU(0.02, inplace=False),
            nn.Linear(self.node_num * self.latent_size, self.latent_size * 2),
            nn.LayerNorm(self.latent_size * 2),
            nn.LeakyReLU(0.02, inplace=False),
            nn.Linear(self.latent_size * 2, 1),
        )

    def get_alpha_t_beta_t(self, total_time_step):
        t = torch.arange(total_time_step)
        alpha_t = torch.sqrt(1 - 0.02 * t / total_time_step + 1e-6)
        alpha_t_bar = torch.cumprod(alpha_t, dim=-1)
        beta_t_bar = torch.sqrt(torch.clamp(1 - alpha_t_bar**2, min=1e-12))
        return alpha_t_bar, beta_t_bar

    def forward(self, x, time_step, apply_noise=True):
        x = x.view(-1, self.node_num * self.latent_size)
        if apply_noise:
            x_step = self.alpha_t_bar.to(x.device)[time_step] * x + self.beta_t_bar.to(
                x.device)[time_step] * torch.randn_like(x)
        else:
            x_step = x

        if torch.any(torch.isnan(x_step)):
            x_modi = replace_nan_with_row_mean(x_step)
        else:
            x_modi = x_step
        time_step_embedding = self.time_step_embedding(time_step.squeeze(dim=1))

        out = self.mlp(
            torch.cat([x_modi, time_step_embedding], dim=-1))

        return out


class Predictor(nn.Module):
    def __init__(self,
                 # Transformer-Encoder parameters
                 latent_size, num_encoder, num_heads, num_ffn,
                 # Predictor Head parameter
                 num_neurons):

        super().__init__()

        self.act_lrelu = nn.LeakyReLU(0.02)
        self.act_tanh = nn.Tanh()
        self.act_relu = nn.ReLU()
        self.act_sigmoid = nn.Sigmoid()
        self.dropout_r = 0.3

        self.encoder_layers = nn.ModuleList([EncoderLayer(latent_size, num_heads, num_ffn,
                                                          self.dropout_r, self.act_lrelu, ) for _ in range(num_encoder)])

        self.pred_fc1 = nn.Sequential(nn.Linear(latent_size, num_neurons), nn.LayerNorm(num_neurons),
                                      self.act_lrelu, nn.Linear(num_neurons, 1), self.act_lrelu)
        self.pred_fc2 = nn.Sequential(nn.Linear(latent_size, num_neurons), nn.LayerNorm(num_neurons),
                                      self.act_relu, nn.Linear(num_neurons, 1), self.act_relu)
        self.pred_fc3 = nn.Sequential(nn.Linear(latent_size, num_neurons), nn.LayerNorm(num_neurons),
                                      self.act_tanh, nn.Linear(num_neurons, 1), self.act_relu)

    def forward(self, h):

        last_attention = None
        for layer in self.encoder_layers:
            h, attention = layer(h)
            last_attention = attention
        # h = h.squeeze(0)
        h_pool = h.sum(dim=1)

        out_qed = self.pred_fc1(h_pool)

        out_sa = self.pred_fc2(h_pool)

        out_pic50 = self.pred_fc3(h_pool)

        out = torch.cat((out_qed, out_sa, out_pic50), dim=-1)

        return out, h, last_attention


class proPredictor(nn.Module):
    def __init__(self,
                 # Transformer-Encoder parameters
                 latent_size, num_encoder, num_heads, num_ffn,
                 # Predictor Head parameter
                 num_neurons):

        super().__init__()

        self.act_lrelu = nn.LeakyReLU(0.02)
        self.act_tanh = nn.Tanh()
        self.act_relu = nn.ReLU()
        self.act_sigmoid = nn.Sigmoid()
        self.dropout_r = 0.3

        self.encoder_layers = nn.ModuleList([EncoderLayer(latent_size, num_heads, num_ffn,
                                                          self.dropout_r, self.act_lrelu, ) for _ in range(num_encoder)])

        self.pred_fc1 = nn.Sequential(nn.Linear(latent_size, num_neurons), nn.LayerNorm(num_neurons),
                                      self.act_lrelu, nn.Linear(num_neurons, 1))
        self.pred_fc2 = nn.Sequential(nn.Linear(latent_size, num_neurons), nn.LayerNorm(num_neurons),
                                      self.act_relu, nn.Linear(num_neurons, 1), self.act_relu)

    def forward(self, h):

        for layer in self.encoder_layers:
            h, attention = layer(h)
        # h = h.squeeze(0)
        h_pool = h.sum(dim=1)

        out_1 = self.pred_fc1(h_pool)

        # out_2 = self.pred_fc2(h_pool)
        # out = torch.cat((out_1, out_2), dim=-1)

        return out_1, h
