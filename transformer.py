import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    """Multi-layer single-head Transformer decoder."""

    def __init__(self, vocab_size, hidden_dim, context_len, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.num_layers = num_layers

        # 1. Token embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        # Positional embeddings
        self.pos_embedding = nn.Embedding(context_len, hidden_dim)

        # Create layers dynamically
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'W_Q': nn.Linear(hidden_dim, hidden_dim),
                'W_K': nn.Linear(hidden_dim, hidden_dim),
                'W_V': nn.Linear(hidden_dim, hidden_dim),
                'W_O': nn.Linear(hidden_dim, hidden_dim),
                'W_up': nn.Linear(hidden_dim, 4 * hidden_dim),
                'W_down': nn.Linear(4 * hidden_dim, hidden_dim),
            })
            self.layers.append(layer)

        # Layer norm parameters for each layer
        self.gamma_attn = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_layers)])
        self.beta_attn = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)])
        self.gamma_mlp = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_layers)])
        self.beta_mlp = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)])

    def layer_norm(self, x, gamma, beta):
        mu = x.mean(dim=-1, keepdim=True)
        sigma = x.std(dim=-1, keepdim=True, unbiased=False)
        x_hat = gamma * (x - mu) / (sigma + 1e-5) + beta
        return x_hat

    def forward(self, x):
        B, T = x.size()  # (Batch size, sequence length)

        # 1. Token embeddings + positional encodings
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        # Process through each layer
        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            residual = h

            # i. Project X into Q, K, and V matrices
            Q = layer['W_Q'](h)  # (B, T, hidden_dim)
            K = layer['W_K'](h)  # (B, T, hidden_dim)
            V = layer['W_V'](h)  # (B, T, hidden_dim)

            # ii. Compute attention scores
            d_k = K.size(-1)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (B, T, T)

            # iii. Causal masking
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

            # iv. Softmax and multiply by values
            attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, T)
            attn_output = torch.matmul(attn_weights, V)  # (B, T, hidden_dim)

            # v. Output projection
            attn_output = layer['W_O'](attn_output)  # (B, T, hidden_dim)

            # vi. Residual and LayerNorm
            h = self.layer_norm(residual + attn_output, self.gamma_attn[layer_idx], self.beta_attn[layer_idx])

            # MLP
            residual = h
            mlp_output = layer['W_up'](h)  # (B, T, 4 * hidden_dim)
            mlp_output = F.relu(mlp_output)
            mlp_output = layer['W_down'](mlp_output)  # (B, T, hidden_dim)

            # Residual & layer norm
            h = self.layer_norm(residual + mlp_output, self.gamma_mlp[layer_idx], self.beta_mlp[layer_idx])

        return h


class MultiHeadTransformer(Transformer):
    """Multi-layer multi-head Transformer decoder."""

    def __init__(self, vocab_size, hidden_dim, context_len, num_heads=4, num_layers=2):
        super().__init__(vocab_size, hidden_dim, context_len, num_layers)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num_heads"

    def forward(self, x):
        B, T = x.size()

        # Embeddings
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        # Process through each layer
        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            residual = h

            # i. Project X into Q, K, and V matrices
            Q = layer['W_Q'](h)  # (B, T, hidden_dim)
            K = layer['W_K'](h)  # (B, T, hidden_dim)
            V = layer['W_V'](h)  # (B, T, hidden_dim)

            # ii. Reshape for multi-head: (B, T, hidden_dim) -> (B, num_heads, T, head_dim)
            Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            K = K.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            # iii. Compute attention scores
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # iv. Causal masking
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

            # v. Softmax and multiply by values
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, T, head_dim)

            # vi. Concatenate heads
            attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)

            # vii. Output projection
            attn_output = layer['W_O'](attn_output)

            # viii. Residual and LayerNorm
            h = self.layer_norm(residual + attn_output, self.gamma_attn[layer_idx], self.beta_attn[layer_idx])

            # ix. MLP
            residual = h
            mlp_output = layer['W_up'](h)
            mlp_output = F.relu(mlp_output)
            mlp_output = layer['W_down'](mlp_output)

            # x. Residual and LayerNorm
            h = self.layer_norm(residual + mlp_output, self.gamma_mlp[layer_idx], self.beta_mlp[layer_idx])

        return h
