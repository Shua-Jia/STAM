
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import rearrange

class FFN(nn.Module):
    def __init__(self, in_channels, inner_channels=None, dropout=0.0):
        super().__init__()
        inner_channels = in_channels if inner_channels is None else inner_channels
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, inner_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_channels, in_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)


class TemporalTransformer(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        num_frames=3,
        patch_size=4,
        num_heads=3,
        dropout=0.0,

    ):
        super().__init__()
        self.embedding = embedding_dim
        self.num_frames = num_frames
        self.patch_size = patch_size

        self.to_patches = nn.Sequential(Rearrange("b n c (hp p1) (wp p2) -> (n p1 p2) (b hp wp)  c",p1=patch_size,p2=patch_size,))

        # multi-head attention
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout)
        # ffn
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = FFN(embedding_dim, embedding_dim * 4, dropout)

        # temporal 1D pos encoding learnable for multi-frame fusion
        self.row_embed = nn.Parameter(torch.Tensor(self.patch_size, self.embedding // 3))
        self.col_embed = nn.Parameter(torch.Tensor(self.patch_size, self.embedding // 3))
        self.depth_embed = nn.Parameter(torch.Tensor(self.num_frames, self.embedding - self.embedding // 3 * 2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed)
        nn.init.uniform_(self.col_embed)
        nn.init.uniform_(self.depth_embed)

    def get_learnable_pos(self):
        pos = (torch.cat([
                    self.row_embed.unsqueeze(0).unsqueeze(2).repeat(self.num_frames, 1, self.patch_size, 1),
                    self.col_embed.unsqueeze(0).unsqueeze(0).repeat(self.num_frames, self.patch_size, 1, 1),
                    self.depth_embed.unsqueeze(1).unsqueeze(1).repeat(1, self.patch_size, self.patch_size, 1),],
                dim=-1,).flatten(0, 2).unsqueeze(1))  # (n*p*p, 1, c)

        return pos  # (n*p*p, 1, c)

    def forward(self, x, ref_idx=None):
        """
        Args:
            x: (b, n, c, h, w), 输入特征图
            ref_idx: (int 或 None), 参考帧索引
        Returns:
            Tensor: 输出特征图
        """
        assert x.dim() == 5, "输入特征图应有 5 个维度！"
        b, n, c, h, w = x.shape

        # 根据需要进行填充
        padding_h = (self.patch_size - h % self.patch_size) % self.patch_size
        padding_w = (self.patch_size - w % self.patch_size) % self.patch_size
        if padding_h > 0 or padding_w > 0:
            x = F.pad(x, (0, padding_w, 0, padding_h))
        h_padded, w_padded = x.shape[-2:]

        # 拆分为补丁
        x_patches = self.to_patches(x)  # (n*p*p, b*h/p*w/p, c)
        if ref_idx is not None:
            row, column = ref_idx * self.patch_size ** 2, (ref_idx + 1) * self.patch_size ** 2
            residual = x_patches[row:column]
        else:
            residual = x_patches

        # 添加位置编码
        x_patches = self.norm1(x_patches)
        x_patches += self.get_learnable_pos()

        # 多头注意力
        query = x_patches[row:column] if ref_idx is not None else x_patches
        x_attn, _ = self.attn(query=query, key=x_patches, value=x_patches)

        # 残差和 FFN
        x = x_attn + residual
        x = x + self.ffn(self.norm2(x))

        # 补丁到特征图
        if ref_idx is None:
            x = rearrange(
                x, "(n p1 p2) (b hp wp) c -> b n c (hp p1) (wp p2)",
                n=n, p1=self.patch_size, p2=self.patch_size,
                hp=h_padded // self.patch_size, wp=w_padded // self.patch_size,
            )
        else:
            x = rearrange(
                x, "(p1 p2) (b hp wp) c -> b c (hp p1) (wp p2)",
                p1=self.patch_size, p2=self.patch_size,
                hp=h_padded // self.patch_size, wp=w_padded // self.patch_size,
            )

        # 移除填充
        if padding_h > 0 or padding_w > 0:
            x = x[..., :h, :w]

        return x

# 测试用例
if __name__ == "__main__":
    batch_size = 2
    num_frames = 3
    channels = 2048
    height = 8
    width = 8

    ref_idx = num_frames // 2
    model = TemporalTransformer(embedding_dim=channels, num_frames=num_frames, patch_size=4, num_heads=8, dropout=0.0)
   #print(model)

    # 随机生成输入特征图
    x = torch.randn(batch_size, num_frames, channels, height, width)
    print(f"Input shape: {x.shape}")

    # 前向传播
    output = model(x, ref_idx=ref_idx)
    print(f"Output shape: {output.shape}")