


import argparse

parser = argparse.ArgumentParser(description="Train FVSRs model with DWNet")
# 数据相关参数
# parser.add_argument("--data_path", default=r"/home/user/Downloads/class_E")
parser.add_argument("--data_path", default=r"E:\data\video coding\videoSRC02_1920x1080_30")
parser.add_argument("--ckpts_path", default=r"E:\AI_Lab\2025-11-26 Summary\lab17_com\TMM\ckpts\samples")
parser.add_argument("--sample_path", default=r"E:\AI_Lab\2025-11-26 Summary\lab17_com\TMM\ckpts\samples")
parser.add_argument("--is_train", default=True)
parser.add_argument("--batch_size", type=int, default=2, help="number of batch size")  # 32 64
parser.add_argument("--num_frames", type=int, default=3)
parser.add_argument("--interval", type=int, default=1)
parser.add_argument("--epoch", type=int, default=2001)
parser.add_argument('--gt_folder', type=str, default='sharp', help='the ground truth folder')
parser.add_argument('--latent', type=str, default='mini_2_264', help='the input folder')
parser.add_argument("--patch", type=int, default=4)
parser.add_argument("--ckpt", type=str, default=r'E:\AI_Lab\2025-11-26 Summary\lab17_com\TMM\ckpts\samples99_0.pth')
parser.add_argument("--ganckpt", type=str, default=r'E:\AI_Lab\2025-11-26 Summary\lab17_com\TMM\ckpts\Gan45_0.pth')
# parser.add_argument("--ckpt", type=str, default='None')
# 模型相关参数
# parser.add_argument("--in_channels", type=int, default=3, help="输入通道数")
parser.add_argument("--initial_channels", type=int, default=12, help="初始卷积后的通道数")
parser.add_argument("--time_emb_dim", type=int, default=128, help="时间嵌入的维度")
parser.add_argument("--patch_size", type=int, default=4, help="补丁大小")
parser.add_argument("--num_heads", type=int, default=4, help="头数")
parser.add_argument("--dropout", type=float, default=0.0, help="Dropout 概率")

# 解析参数
args = parser.parse_args()