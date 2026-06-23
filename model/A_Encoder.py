import torch
from torch import nn
from einops import rearrange
# from model.Base_attention import OptimizedLinearAttention, LayerNorm
from model.Base_attention import OptimizedLinearAttention, LayerNorm

def extract_middle_from_batches(input_tensor):
    # 确保输入的batch大小是3的倍数
    assert input_tensor.shape[0] % 3 == 0, "Batch size must be a multiple of 3"
    # 重塑张量
    reshaped = input_tensor.view(-1, 3, *input_tensor.shape[1:])
    # 提取中间的值
    result = reshaped[:, 1]  # 选择中间的值
    # 重塑回原始形状
    return result

def get_learnable_t(self):
    t = (torch.cat([
                self.row_embed.unsqueeze(0).unsqueeze(2).repeat(self.num_frames, 1, self.patch_size, 1),
                self.col_embed.unsqueeze(0).unsqueeze(0).repeat(self.num_frames, self.patch_size, 1, 1),
                self.depth_embed.unsqueeze(1).unsqueeze(1).repeat(1, self.patch_size, self.patch_size, 1),],
            dim=-1,).flatten(0, 2).unsqueeze(1))  # (n*p*p, 1, c)

    return t


def get_learnable_t(self):
    # Expand dimensions for broadcasting without using repeat
    row_embed = self.row_embed.view(1, -1, 1, 1)  # Shape: (1, row_dim, 1, 1)
    col_embed = self.col_embed.view(1, 1, -1, 1)  # Shape: (1, 1, col_dim, 1)
    depth_embed = self.depth_embed.view(-1, 1, 1, 1)  # Shape: (depth_dim, 1, 1, 1)
    time_embed = self.time_embed.view(1, 1, 1, -1)  # Shape: (1, 1, 1, time_dim)

    # Use broadcasting to combine embeddings
    combined = torch.cat(
        [
            row_embed.expand(self.num_frames, -1, self.patch_size, 1),
            col_embed.expand(self.num_frames, self.patch_size, -1, 1),
            depth_embed.expand(self.num_frames, self.patch_size, self.patch_size, 1),
            time_embed.expand(self.num_frames, self.patch_size, self.patch_size, -1),
        ],
        dim=-1,  # Concatenate along the last dimension
    )

    # Flatten the spatial dimensions and add an extra dimension
    t = combined.flatten(0, 2).unsqueeze(1)  # Shape: (num_frames * p * p, 1, combined_dim)

    return t



class OptimizedConvNextBlock(nn.Module):
    """ConvNext改进模块"""
    def __init__(self, dim, dim_out, *, time_emb_dim=None, exp_ratio=4, norm=True):
        super().__init__()
        self.use_time_emb = time_emb_dim is not None

        if self.use_time_emb:
            self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, dim * 2)
            )

        self.ds_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * exp_ratio, 1),
            nn.GELU(),
            nn.Conv2d(dim * exp_ratio, dim_out, kernel_size=3, padding=1),
            LayerNorm(dim_out) if norm else nn.Identity(),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if self.use_time_emb:
            assert time_emb is not None, "time_emb必须提供"
            time_cond = self.time_mlp(time_emb)
            time_cond = rearrange(time_cond, "b c -> b c 1 1")
            weight, bias = torch.split(time_cond, h.shape[1], dim=1)
            h = h * (1 + weight) + bias

        h = self.net(h)
        return h + self.alpha * self.res_conv(x)

class OptimizedConvNextEncoder(nn.Module):
    """基于优化的ConvNext块的编码器"""
    def __init__(self, initial_channels, time_emb_dim, dimensions):
        super().__init__()
        #self.initial_conv = nn.Conv2d(in_channels, initial_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        h = 3
        for i, (in_dim, out_dim, stride) in enumerate(dimensions):   # i: 0-4
            block = OptimizedConvNextBlock(dim=in_dim, dim_out=out_dim, time_emb_dim=time_emb_dim)
            self.blocks.append(block)
            downsample_conv = nn.Conv2d(out_dim, out_dim, kernel_size=stride * 2, stride=stride, padding=(stride * 2) // 2 - 1)
            self.blocks.append(downsample_conv)

            # 添加注意力层
            self.attention_layers.append(OptimizedLinearAttention(dim=out_dim, heads=4, dim_head=32,compress_times=h))
            if h != 0:
                h -= i


    def forward(self, x, time_emb):
        h = []
        h.append(extract_middle_from_batches(x))
        attention_index = 0
       # x = self.initial_conv(x)
        for i, layer in enumerate(self.blocks):
            if isinstance(layer, OptimizedConvNextBlock):
                x = layer(x, time_emb)
                time_emb+=1
            else:  # 假设 layer 是 nn.Conv2d 的实例
                x = layer(x)
               # print("Shape after downsample_conv:", x.shape)  # 打印下采样后的形状

            # 应用对应的注意力层
            if i % 2 == 1:  # 每两个层后应用一个注意力层
                x = self.attention_layers[attention_index](x)
                attention_index += 1
                h.append(extract_middle_from_batches(x))
        return x,h