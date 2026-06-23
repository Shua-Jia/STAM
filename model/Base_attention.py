import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """通道维度上的 LayerNorm，适用于卷积层"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps  # 初始化一个小常数以防止除零错误
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))  # 学习的缩放参数
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))  # 学习的偏移参数

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)  # 计算每个通道的方差
        mean = torch.mean(x, dim=1, keepdim=True)  # 计算每个通道的均值
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b  # 应用层归一化公式
#
# class OptimizedLinearAttention(nn.Module):
#     """优化的线性注意力模块"""
#     def __init__(self, dim, heads=4, dim_head=32, dynamic_heads=False, norm=True):
#         super().__init__()
#         self.dim_head = dim_head
#         self.scale = dim_head ** -0.5
#         self.dynamic_heads = dynamic_heads
#         self.heads = heads
#         hidden_dim = dim_head * heads
#         self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)  # 将输入映射到QKV
#         self.to_out = nn.Conv2d(hidden_dim, dim, 1)  # 将输出映射回原始维度
#         self.norm = LayerNorm(dim) if norm else nn.Identity()  # 可选的层归一化
#
#     def forward(self, x):
#         b, c, h, w = x.shape  # 获取输入张量的形状
#         x = self.norm(x)  # 应用层归一化
#         heads = self.heads if not self.dynamic_heads else c // self.dim_head  # 计算头的数量
#
#         # QKV计算
#         qkv = self.to_qkv(x).chunk(3, dim=1)  # 将QKV分开
#
#         q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=heads), qkv)  # 调整形状
#         q = q * self.scale  # 缩放查询向量
#         # 注意力计算
#         k = k - k.amax(dim=-1, keepdim=True).detach()  # 减去最大值以稳定softmax
#         k = k.softmax(dim=-1)  # 应用softmax
#
#         context = torch.matmul(k, v.transpose(-2, -1))  # 计算上下文向量
#         out = torch.matmul(context.transpose(-2, -1), q)  # 计算输出向量  6 4 32 1024
#
#         # 恢复形状并输出
#         out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=heads, x=h, y=w)  # 调整形状
#         return self.to_out(out) + x  # 加上残差并返回

# def pad_tensor_to_match(out, x):
#     # 获取两个张量的形状
#     out_shape = out.shape  # (batch_size, channels, height, width)
#     x_shape = x.shape  # (batch_size, channels, height, width)
#
#     # 检查第三个维度（height）
#     if x_shape[2] != out_shape[2]:
#         # 计算需要的 padding
#         padding_needed = out_shape[2] - x_shape[2]
#
#         # 在height维度上进行padding，padding=(left, right)
#         # 这里的left为padding_needed // 2，right为padding_needed - left
#         left_padding = padding_needed // 2
#         right_padding = padding_needed - left_padding
#
#         # 使用F.pad进行padding
#         x = F.pad(x, (0, 0, left_padding, right_padding), mode='constant', value=0)
#
#     return x
def pad_tensor_to_match(out, x):
    # 获取两个张量的形状
    out_shape = out.shape  # (batch_size, channels, height, width)
    x_shape = x.shape      # (batch_size, channels, height, width)

    # 初始化 padding 参数
    # (padding for width (left, right), padding for height (top, bottom))
    padding = [0, 0, 0, 0]  # (left, right, top, bottom)

    # 第三维度（height）不同时，计算需要的 padding
    if x_shape[2] != out_shape[2]:
        height_padding_needed = out_shape[2] - x_shape[2]
        top_padding = height_padding_needed // 2
        bottom_padding = height_padding_needed - top_padding
        padding[2] = top_padding  # top padding
        padding[3] = bottom_padding  # bottom padding

    # 第四维度（width）不同时，计算需要的 padding
    if x_shape[3] != out_shape[3]:
        width_padding_needed = out_shape[3] - x_shape[3]
        left_padding = width_padding_needed // 2
        right_padding = width_padding_needed - left_padding
        padding[0] = left_padding  # left padding
        padding[1] = right_padding  # right padding

    # 使用 F.pad 增加 padding
    # F.pad expects padding in the order (left, right, top, bottom)
    x = F.pad(x, padding, mode='constant', value=0)

    return x

class OptimizedLinearAttention(nn.Module):
    """优化的线性注意力模块，支持空间压缩与恢复"""
    def __init__(self, dim, heads=4, dim_head=32, dynamic_heads=False, norm=True,compress_times=2):
        super().__init__()
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dynamic_heads = dynamic_heads
        self.heads = heads
        hidden_dim = dim_head * heads

        # 卷积压缩部分
        layers = []
        for _ in range(compress_times):
            layers.extend([
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),  # 空间压缩一半
                nn.ReLU()
            ])
        self.compress = nn.Sequential(*layers)

        # QKV映射
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        # 输出映射
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

        # 反卷积恢复部分
        delayers = []
        for _ in range(compress_times):
            delayers.extend([
                nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=False),  # 再次恢复一倍
                nn.ReLU()
            ])
        self.decompress = nn.Sequential(*delayers)
        # 可选的层归一化
        self.norm = LayerNorm(dim) if norm else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape  # 获取输入张量的形状
        x = self.norm(x)  # 应用层归一化

        # 压缩空间维度
        x_compressed = self.compress(x)  # 形状变为 (b, c, h//4, w//4)
        _, _, h_compressed, w_compressed = x_compressed.shape

        # 动态计算头的数量
        heads = self.heads if not self.dynamic_heads else c // self.dim_head
        # QKV计算
        qkv = self.to_qkv(x_compressed).chunk(3, dim=1)  # 将QKV分开
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=heads), qkv)  # 调整形状

        # 缩放查询向量
        q = q * self.scale

        # 注意力计算
        k = k - k.amax(dim=-1, keepdim=True).detach()  # 减去最大值以稳定softmax
        k = k.softmax(dim=-1)  # 应用softmax
        context = torch.matmul(k, v.transpose(-2, -1))  # 计算上下文向量
        out = torch.matmul(context.transpose(-2, -1), q)  # 计算输出向量

        # 恢复形状
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=heads, x=h_compressed, y=w_compressed)  # 调整形状

        # 应用输出映射
        out = self.to_out(out)

        # 反卷积恢复到原始空间维度
        out = self.decompress(out)  # 形状恢复为 (b, c, h, w)

        # 调用函数
        x = pad_tensor_to_match(out, x)
        return out + x  # 残差连接


# 测试 OptimizedLinearAttention 模块
if __name__ == "__main__":
    # 定义参数
    batch_size = 6
    channels = 48
    height = 64
    width = 64

    # 创建输入数据
    x = torch.randn(batch_size, channels, height, width)  # Batch size = 5, 输入通道数 = 64, 分辨率 = 256x256

    # 实例化 OptimizedLinearAttention
    attention_module = OptimizedLinearAttention(dim=channels, heads=4, dim_head=32,compress_times=0)

    # 前向传播
    output = attention_module(x)

    # 打印输入和输出形状
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")