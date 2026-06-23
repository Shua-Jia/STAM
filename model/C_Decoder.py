import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class LayerNorm(nn.Module):
    """通道维度上的 LayerNorm，适用于卷积层"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class OptimizedConvNextBlock(nn.Module):
    """修改后的ConvNext改进模块，支持上采样"""
    def __init__(self, dim, dim_out, *, scale_factor=2, time_emb_dim=None, exp_ratio=2, norm=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.use_time_emb = time_emb_dim is not None

        if self.use_time_emb:
            self.time_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, dim * 2)
            )

        self.up_sample = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
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
            assert time_emb is not None, "time_emb must be provided"
            time_cond = self.time_mlp(time_emb)
            time_cond = rearrange(time_cond, "b c -> b c 1 1")
            weight, bias = torch.split(time_cond, h.shape[1], dim=1)
            h = h * (1 + weight) + bias

        h = self.up_sample(h)
        h = self.net(h)

        x_upsampled = self.up_sample(x)
        return h + self.alpha * self.res_conv(x_upsampled)

class OptimizedLinearAttention(nn.Module):
    """优化的线性注意力模块"""
    def __init__(self, dim, heads=4, dim_head=32, dynamic_heads=False, norm=True):
        super().__init__()
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dynamic_heads = dynamic_heads
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)  # 将输入映射到QKV
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)  # 将输出映射回原始维度
        self.norm = LayerNorm(dim) if norm else nn.Identity()  # 可选的层归一化

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入张量的形状
        x = self.norm(x)  # 应用层归一化
        heads = self.heads if not self.dynamic_heads else c // self.dim_head  # 计算头的数量

        # QKV计算
        qkv = self.to_qkv(x).chunk(3, dim=1)  # 将QKV分开
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=heads), qkv)  # 调整形状
        q = q * self.scale  # 缩放查询向量
        # 注意力计算
        k = k - k.amax(dim=-1, keepdim=True).detach()  # 减去最大值以稳定softmax
        k = k.softmax(dim=-1)  # 应用softmax
        context = torch.matmul(k, v.transpose(-2, -1))  # 计算上下文向量
        out = torch.matmul(context.transpose(-2, -1), q)  # 计算输出向量

        # 恢复形状并输出
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=heads, x=h, y=w)  # 调整形状
        return self.to_out(out) + x  # 加上残差并返回

class UpSamplingNetwork(nn.Module):
    def __init__(self, channel_list):
        super().__init__()

        self.up_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        for i in range(len(channel_list) - 1):
            if i==0:
                self.up_blocks.append(OptimizedConvNextBlock(dim=channel_list[i], dim_out=channel_list[i + 1], scale_factor=1))
            else:
                self.up_blocks.append(OptimizedConvNextBlock(dim=channel_list[i], dim_out=channel_list[i + 1], scale_factor=2))
            self.attention_blocks.append(OptimizedLinearAttention(dim=channel_list[i + 1]))

    def forward(self, x,h):
        for i, (up_block, attention_block) in enumerate(zip(self.up_blocks, self.attention_blocks)):
            x = up_block(x)
            #print(f"Block {i + 1} output shape after upsampling: {x.shape}")
            x = attention_block(x)  # 添加注意力机制
            last_element = h.pop()

            if x.shape[2:] != last_element.shape[2:]:
                # 计算需要填充的大小
                diff_height = last_element.shape[2] - x.shape[2]  # 高度差异
                diff_width = last_element.shape[3] - x.shape[3]  # 宽度差异

                # 如果差异为正数，说明 x 的尺寸较小，需要填充
                pad_top = diff_height // 2
                pad_bottom = diff_height - pad_top
                pad_left = diff_width // 2
                pad_right = diff_width - pad_left

                # 使用 F.pad 填充 x，使其与 last_element 的尺寸匹配
                x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
            x+=last_element   # 64 360
            # print(f"Block {i + 1} output shape after attention: {x.shape}")

        return x

# 测试代码
if __name__ == "__main__":
    x = torch.randn(2, 512, 16, 16)  # 输入
    de_channels = x.shape[1]
    channel_list = [de_channels, de_channels//2, de_channels//4, de_channels//8, 3]  # 可以根据需要传入不同的通道数
    model = UpSamplingNetwork(channel_list)
    output = model(x)
    print("Output shape:", output.shape)  # 应输出: torch.Size([2, 3, 256, 256])