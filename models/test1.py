import torch
from torch import nn, einsum
from models.model_utils import grouping_operation, CBL
from knn_cuda import KNN
knn = KNN(k=16, transpose_mode=False)


class encTransformer(nn.Module):
    def __init__(self, in_channel, dim=64, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(encTransformer, self).__init__()
        self.conv_key = nn.Conv1d(dim, dim, 1)
        self.conv_query = nn.Conv1d(dim, dim, 1)
        self.conv_value = nn.Conv1d(dim, dim, 1)

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )

        self.linear_start = nn.Conv1d(in_channel, dim, 1)
        self.linear_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, x, pos):
        """feed forward of transformer
        Args:
            x: Tensor of features, (B, in_channel, n)
            pos: Tensor of positions, (B, 3, n)

        Returns:
            y: Tensor of features with attention, (B, in_channel, n)
        """
        identity = x
        x = self.linear_start(x)
        b, dim, n = x.shape

        # pos_flipped = pos.permute(0, 2, 1).contiguous()
        # idx_knn = query_knn(self.n_knn, pos_flipped, pos_flipped)
        with torch.no_grad():
            _, idx_knn = knn(pos, pos)  # bs k np
            idx_knn = idx_knn.int().transpose(2, 1).contiguous()
        key = self.conv_key(x)
        value = self.conv_value(x)
        query = self.conv_query(x)

        key = grouping_operation(key, idx_knn)  # b, dim, n, n_knn
        qk_rel = query.reshape((b, -1, n, 1)) - key  # b, dim, n, n_knn

        pos_rel = pos.reshape((b, -1, n, 1)) - grouping_operation(pos, idx_knn)  # b, 3, n, n_knn
        pos_embedding = self.pos_mlp(pos_rel)  # b, dim, n, n_knn

        attention = self.attn_mlp(qk_rel + pos_embedding)  # b, n, n_knn
        attention = torch.softmax(attention, -1)  # b, dim, n, n_knn

        value = value.reshape((b, -1, n, 1)) + pos_embedding

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # b, dim, n
        y = self.linear_end(agg)

        return y + identity


class decAttention(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)

        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, q, v):
        q = q.transpose(2, 1)
        v = v.transpose(2, 1)
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x.transpose(2, 1)
        return x


class decTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, global_channels):
        super(decTransformer, self).__init__()
        self.query_mlp = CBL(in_channels, out_channels)
        self.fuse_mlp = CBL(3+global_channels, out_channels)
        self.atten = decAttention(out_channels, out_channels, num_heads=8, qkv_bias=False, qk_scale=None)
        self.out_mlp = CBL(out_channels, out_channels)

    def forward(self, in_pos, encode_features, global_features):
        npoints = in_pos.shape[2]
        # generate v
        global_features = global_features.unsqueeze(-1).expand(-1, -1, npoints)
        # pos_features = self.pos_mlp(in_pos)
        v = torch.cat((in_pos, global_features), dim=1)

        v = self.fuse_mlp(v)
        # generate q
        q = self.query_mlp(encode_features)
        # attention
        trans_features = self.atten(q, v)
        res = self.out_mlp(q + trans_features)

        return res


class testAttention(nn.Module):
    def __init__(self,  dim, out_dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.out_dim = out_dim
        head_dim = out_dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.k_map = nn.Linear(dim, out_dim, bias=qkv_bias)
        self.v_map = nn.Linear(dim, out_dim, bias=qkv_bias)

        self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, q, v):
        q = q.transpose(2, 1)
        v = v.transpose(2, 1)
        B, N, _ = q.shape
        C = self.out_dim
        k = v
        NK = k.size(1)

        q = self.q_map(q).view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_map(k).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_map(v).view(B, NK, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = x.transpose(2, 1)
        return x


class TEFP(nn.Module):
    def __init__(self, in_channels, out_channels, global_channels):
        super().__init__()
        self.query_mlp = CBL(in_channels, out_channels)
        self.fuse_mlp = CBL(3+in_channels+global_channels, out_channels)
        self.atten = decAttention(out_channels, out_channels, num_heads=8, qkv_bias=False, qk_scale=None)
        self.out_mlp = CBL(out_channels, out_channels)

    def forward(self, in_pos, encode_features, global_features):
        npoints = in_pos.shape[2]
        # generate q
        global_features = global_features.unsqueeze(-1).expand(-1, -1, npoints)
        # pos_features = self.pos_mlp(in_pos)
        q = torch.cat((in_pos, encode_features, global_features), dim=1)

        q = self.fuse_mlp(q)
        # generate v
        v = self.query_mlp(encode_features)
        # attention
        trans_features = self.atten(q, v)
        res = self.out_mlp(q + trans_features)

        return res

if __name__ == "__main__":
    enc_features1 = torch.randn(8, 256, 64)
    global_fea1 = torch.randn(8, 1024)
    in_points1 = torch.randn(8, 3, 64)
    tran = TEFP(256, 512, 1024)
    ret1 = tran(in_points1, enc_features1, global_fea1)
    print(ret1.shape)

    enc_features2 = torch.randn(8, 512, 256)
    in_points2 = torch.randn(8, 3, 256)
    dec_points = ret1
    tran2 = TEFP(512, 1024, 1024)
    ret2 = tran2(in_points2, enc_features2, global_fea1)
    print(ret2.shape)
