
import torch.nn as nn
import torch
from einops import rearrange
from copy import deepcopy
import torch
import os
import torchvision.utils as vutils
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(
            self,
            dim=32,
            dim_mults=(1, 2, 4, 4,6,6,8),
            channels=3,
            with_time_emb=True,
    ):
        super().__init__()
        self.channels = dim

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.model_depth = len(dim_mults)

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)

        self.initial = nn.Conv2d(channels, dim, 1)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ConvNextBlock_dis(dim_in, dim_out, norm=ind != 0),
                nn.AvgPool2d(2),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))) if ind >= (num_resolutions - 3) and not is_last else nn.Identity(),
                ConvNextBlock_dis(dim_out, dim_out),
            ]))
        dim_out = dim_mults[-1] * dim

        self.out = nn.Conv2d(dim_out, 1, 1, bias=False)


    def forward(self, x):
        x = self.initial(x)
        for convnext, downsample, convnext2 in self.downs:
            x = convnext(x)
            x = downsample(x)
            # x = attn(x)
            x = convnext2(x)
        return self.out(x).view(x.shape[0], -1)


class ConvNextBlock_dis(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim*2)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            nn.BatchNorm2d(dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim_out * mult, dim_out, 3, 1, 1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            condition = rearrange(condition, 'b c -> b c 1 1')
            weight, bias = torch.split(condition, x.shape[1],dim=1)
            h = h * (1 + weight) + bias

        h = self.net(h)
        return h + self.res_conv(x)


def exists(x):
    return x is not None



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            m.weight.data.normal_(0.0, 0.02)
        except:
            pass
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def DiffAugment(x, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x = f(x)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
        x = x.contiguous()
    return x


def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x


def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x


def rand_contrast(x):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, ratio=0.5):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}




def save_models( model_en,model_de,epoch,index,ckpts_path):
    torch.save(model_en.state_dict(), os.path.join(ckpts_path, f'FVSRs_en_epo_{epoch}_ind{index}.pth'))
    torch.save(model_de.state_dict(), os.path.join(ckpts_path, f'FVSRs_de_epo_{epoch}_ind{index}.pth'))
   # print(f'Epoch {epoch} completed. Models saved.')

def save_de_out_image(de_out, video_name, index, epoch,sample_path):
    # 保存每个图像

    for i in range(de_out.size(0)):  # 遍历 batch 中的每个图像
        image_name = f'epo{epoch}_bat{index}_img{i}_{video_name[0][:8]}.png'
        save_path = os.path.join(sample_path, image_name)
        vutils.save_image(de_out[i], save_path, normalize=True)
    log_file_path = os.path.join(sample_path, 'log.txt')

    if not os.path.exists(log_file_path):
        with open(log_file_path, 'w') as log_file:  # 创建新文件
            log_file.write('Epoch,  Index,  Video Name\n')  # 可选：写入表头

    with open(log_file_path, 'a') as log_file:  # 以追加模式打开文件
        log_file.write(f'Epoch: {epoch}, Index: {index}, Video Name: {video_name}\n')



def save_input_image(input, index, epoch,sample_path):
    # 保存每个图像
    for i in range(input.size(0)):  # 遍历 batch 中的每个图像
        image_name = f'epo{epoch}_bat{index}_img{i}.png'
        save_path = os.path.join(sample_path, image_name)
        vutils.save_image(input[i], save_path, normalize=True)
       # print(f'Saved image: {sample_path}')
def save_gt_image(input, index, epoch,sample_path):
    # 保存每个图像
    for i in range(input.size(0)):  # 遍历 batch 中的每个图像
        image_name = f'epo{epoch}_bat{index}_img{i}_gt.png'
        save_path = os.path.join(sample_path, image_name)
        vutils.save_image(input[i], save_path, normalize=True)
        #print(f'Saved image: {sample_path}')


def calculate_psnr(img1, img2):
    """
    计算PSNR（峰值信噪比）
    数据范围 [0, 1]
    """
    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])

    # 防止除以零
    mse = torch.where(mse == 0, torch.tensor(1e-10, device=mse.device), mse)

    PIXEL_MAX = 1.0
    return 20 * torch.mean(torch.log10(PIXEL_MAX / torch.sqrt(mse)))


class CharbonnierLoss(nn.Module):
    def __init__(self,eps=1e-8):
        super(CharbonnierLoss,self).__init__()
        self.eps=eps

    def forward(self,pred,target):

        return torch.mean(torch.sqrt((pred - target)**2 + self.eps)) / target.shape[0]


















