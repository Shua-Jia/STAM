from DT_Utils.base_args_eval import args
import torch
from einops import rearrange
import data_preprocess
from model.DW_Net import UNetModel
import os
import torchvision.utils as vutils




def save_de_out_image(output, img_name, folder_name, output_dir):

    save_folder = os.path.join(output_dir, folder_name[0])
    os.makedirs(save_folder, exist_ok=True)


    if not isinstance(output, torch.Tensor):
        raise TypeError("output must be tensor 。")


    for i in range(output.shape[0]):

        img_name_current = os.path.splitext(img_name[i][0])[0]

        save_path = os.path.join(save_folder, f"{img_name_current}.png")

        vutils.save_image(output[i], save_path, normalize=True)
        print(f"save：{save_path}")

def calculate_psnr(img1, img2):

    img1 = img1.clamp(0, 1)
    img2 = img2.clamp(0, 1)

    mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])

    # 防止除以零
    mse = torch.where(mse == 0, torch.tensor(1e-10, device=mse.device), mse)

    PIXEL_MAX = 1.0
    return 20 * torch.mean(torch.log10(PIXEL_MAX / torch.sqrt(mse)))

class evaler:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.dataset = data_preprocess.data(args)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size)   #,shuffle=True)


        encoder_params = {
            #"in_channels": args.in_channels,
            "initial_channels": args.initial_channels,
            "time_emb_dim": args.time_emb_dim
        }

        self.model = UNetModel(
            encoder_params=encoder_params,
            num_frames=args.num_frames,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            dropout=args.dropout,
            decoder_params=None
        ).to(self.device)

        checkpoint = args.ckpt
        if checkpoint != 'None':
            ckpt = torch.load(checkpoint)
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['codec'].items()})
            current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
            del ckpt
    def count_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'Total trainable parameters: {total_params}')

    def eval(self):
        self.count_parameters()  # 打印参数量
        total_psnr = 0
        total_batches = 0
        for index, batch_data in enumerate(self.dataloader):

            input_fr = batch_data["input_frames"].to(self.device)
            input = rearrange(input_fr, 'b t c n d -> (b t) c n d')
            gt = batch_data["gt_frames"].to(self.device)
            gt = rearrange(gt, 'b t c n d -> (b t) c n d')
            folder_name = batch_data['folder_name']
            img_name = batch_data['gt_names']
            batch_size = input.size(0)
            time_emb = torch.randn(batch_size, self.args.time_emb_dim).to(self.device)

            ref_idx = self.args.num_frames // 2

            with torch.no_grad():
                output = self.model(input, time_emb, ref_idx)
            psnr=calculate_psnr(output,gt)
            print(folder_name,'--------------', psnr)
            total_psnr += psnr
            total_batches += 1

            save_de_out_image(output, img_name, folder_name,args.output_dir)

        average_psnr = total_psnr / total_batches if total_batches > 0 else 0
        print(f'Average PSNR: {average_psnr}')

if __name__ == "__main__":
    evalers = evaler(args)
    evalers.eval()



























































































































































































































































































