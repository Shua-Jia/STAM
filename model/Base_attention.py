import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F


class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

def pad_tensor_to_match(out, x):

    out_shape = out.shape  # (batch_size, channels, height, width)
    x_shape = x.shape      # (batch_size, channels, height, width)


    padding = [0, 0, 0, 0]

    if x_shape[2] != out_shape[2]:
        height_padding_needed = out_shape[2] - x_shape[2]
        top_padding = height_padding_needed // 2
        bottom_padding = height_padding_needed - top_padding
        padding[2] = top_padding
        padding[3] = bottom_padding

    if x_shape[3] != out_shape[3]:
        width_padding_needed = out_shape[3] - x_shape[3]
        left_padding = width_padding_needed // 2
        right_padding = width_padding_needed - left_padding
        padding[0] = left_padding
        padding[1] = right_padding


    x = F.pad(x, padding, mode='constant', value=0)

    return x

class OptimizedLinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=32, dynamic_heads=False, norm=True,compress_times=2):
        super().__init__()
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.dynamic_heads = dynamic_heads
        self.heads = heads
        hidden_dim = dim_head * heads


        layers = []
        for _ in range(compress_times):
            layers.extend([
                nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
                nn.ReLU()
            ])
        self.compress = nn.Sequential(*layers)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

        delayers = []
        for _ in range(compress_times):
            delayers.extend([
                nn.ConvTranspose2d(dim, dim, kernel_size=4, stride=2, padding=1, bias=False),
                nn.ReLU()
            ])
        self.decompress = nn.Sequential(*delayers)

        self.norm = LayerNorm(dim) if norm else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)


        x_compressed = self.compress(x)
        _, _, h_compressed, w_compressed = x_compressed.shape

        heads = self.heads if not self.dynamic_heads else c // self.dim_head
        # QKV计算
        qkv = self.to_qkv(x_compressed).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=heads), qkv)

        q = q * self.scale

        k = k - k.amax(dim=-1, keepdim=True).detach()
        k = k.softmax(dim=-1)  # 应用softmax
        context = torch.matmul(k, v.transpose(-2, -1))
        out = torch.matmul(context.transpose(-2, -1), q)

        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=heads, x=h_compressed, y=w_compressed)
        out = self.to_out(out)
        out = self.decompress(out)  # 形状恢复为 (b, c, h, w)

        x = pad_tensor_to_match(out, x)
        return out + x

