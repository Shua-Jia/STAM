import torch.nn as nn

from model.A_DW_Encoder import OptimizedConvNextEncoder
from model.B_Latent_fusion import TemporalTransformer
from model.C_DW_Decoder import UpSamplingNetwork
from DT_Utils.base_args import args
from einops.layers.torch import Rearrange
from model.Base_attention import OptimizedLinearAttention


class UNetModel(nn.Module):



    def __init__(self, encoder_params, num_frames, patch_size, num_heads, dropout, decoder_params,laten_channels=384):

        super(UNetModel, self).__init__()

        self.initial_downsample = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (p1 p2 c) h w", p1=2, p2=2)  # times is 2
        )
        encoder_params["dimensions"] = [
            (args.initial_channels, args.initial_channels * 2, 2),
            (args.initial_channels * 2, args.initial_channels * 4, 2),
            (args.initial_channels * 4, args.initial_channels * 8, 2),
            (args.initial_channels * 8, args.initial_channels * 16, 2),
            (args.initial_channels * 16, args.initial_channels * 32, 2),
        ]
        self.channel_list = [
            args.initial_channels * 32,
            args.initial_channels * 32,
            args.initial_channels * 16,
            args.initial_channels * 8,
            args.initial_channels * 4,
            args.initial_channels * 2,
            12,
        ]

        self.encoder = OptimizedConvNextEncoder(
            initial_channels=encoder_params["initial_channels"],
            time_emb_dim=encoder_params["time_emb_dim"],
            dimensions=encoder_params["dimensions"],
        )

        self.temporal_transformer = TemporalTransformer(
            embedding_dim=encoder_params["dimensions"][-1][-2],
            num_frames=num_frames,
            patch_size=patch_size,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.re_initial_downsample = nn.Sequential(
            Rearrange("b (p1 p2 c) h w->b c (h p1) (w p2)", p1=2, p2=2)  # times is 2
        )

        self.decoder = UpSamplingNetwork(self.channel_list)

        self.in_attention_modules = nn.ModuleList([
            OptimizedLinearAttention(dim=laten_channels, heads=4, dim_head=32, compress_times=0)
            for _ in range(5)
        ])

        self.out_attention_modules = nn.ModuleList([
            OptimizedLinearAttention(dim=laten_channels, heads=4, dim_head=32, compress_times=0)
            for _ in range(15)
        ])



    def forward(self, x, time_emb, ref_idx=None):

        x = self.initial_downsample(x)

        encoded_output,input_h = self.encoder(x, time_emb)
        b, c, h, w = encoded_output.size()
        num_frames = b // self.temporal_transformer.num_frames
        for attention_module in self.in_attention_modules:
            encoded_output = attention_module(encoded_output)
        encoded_output = encoded_output.reshape(num_frames, self.temporal_transformer.num_frames, c, h, w)

        transformed_output = self.temporal_transformer(encoded_output, ref_idx)

        # from sklearn.decomposition import PCA
        # import numpy as np
        # import matplotlib.pyplot as plt
        # # Simulate the tensor of size (1, 384, 17, 30)
        # # Reshape to (384, 17 * 30) for PCA
        #
        # transformed_output1=transformed_output.clone().cpu()
        #
        # tensor_flat = transformed_output1.reshape(384, -1).T  # Shape: (17*30, 384)
        # # Apply PCA to reduce to 3 components
        # pca = PCA(n_components=3)
        # tensor_pca = pca.fit_transform(tensor_flat)  # Shape: (17*30, 3)
        # # Reshape back to (3, 17, 30)
        # tensor_3 = tensor_pca.T.reshape(3, 17, 30)
        #
        # # Plot the 3 PCA channels
        # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # for i in range(3):
        #     axes[i].imshow(tensor_3[i], cmap='viridis')  # Visualize each channel
        #     axes[i].set_title(f'PCA Channel {i + 1}')
        #     axes[i].axis('off')
        #
        # plt.tight_layout()
        # plt.show()


        for ou_attention_module in self.out_attention_modules:
            transformed_output = ou_attention_module(transformed_output)
        decoded_output = self.decoder(transformed_output,input_h)
        decoded_output = self.re_initial_downsample(decoded_output)
        return decoded_output













