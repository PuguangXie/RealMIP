import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=64, projection_dim=None):
        super().__init__()
        self.projection_dim = projection_dim if projection_dim is not None else embedding_dim
        self.embedding = self._build_embedding(num_steps, embedding_dim // 2)
        self.projection1 = nn.Linear(embedding_dim, self.projection_dim)
        self.projection2 = nn.Linear(self.projection_dim, self.projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = F.silu(self.projection1(x))
        x = F.silu(self.projection2(x))
        return x

    def _build_embedding(self, num_steps, dim=128):
        steps = torch.arange(num_steps).unsqueeze(1).float()  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(
                side_dim=config["side_dim"],
                channels=self.channels,
                diffusion_embedding_dim=config["diffusion_embedding_dim"],
                nheads=config["nheads"],
            ) for _ in range(config["layers"])
        ])

    def forward(self, x, cond_info, diffusion_step, seq_length):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = F.relu(self.input_projection(x))
        x = x.view(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip_connections = []

        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb, seq_length)
            skip_connections.append(skip_connection)

        x = torch.sum(torch.stack(skip_connections), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = F.relu(self.output_projection1(x))
        x = self.output_projection2(x)
        x = x.view(B, K, L)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape, seq_length):
        B, channel, K, L = base_shape
        if L == 1:
            return y

        device = y.device
        length_mask = torch.arange(L, device=device).unsqueeze(0) >= seq_length.unsqueeze(1)
        length_mask = length_mask.unsqueeze(1).expand(-1, K, -1).reshape(B * K, L)

        y = y.view(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = y.permute(2, 0, 1)
        y = self.time_layer(y, src_key_padding_mask=length_mask)
        y = y.permute(1, 2, 0)
        return y.view(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.view(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        return y.view(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)

    def forward(self, x, cond_info, diffusion_emb, seq_length):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.view(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape, seq_length)
        y = self.forward_feature(y, base_shape)
        y = self.mid_projection(y)

        cond_info = cond_info.view(B, cond_info.size(1), K * L)
        y = y + self.cond_projection(cond_info)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        return (x.view(base_shape) + residual.view(base_shape)) / math.sqrt(2.0), skip.view(base_shape)