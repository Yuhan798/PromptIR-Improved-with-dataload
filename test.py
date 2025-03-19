import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DenoiseTestDataset, DeDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import pytorch_lightning as pl
from utils.schedulers import LinearWarmupCosineAnnealingLR
import torch.nn.functional as F

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=100)

        return [optimizer],[scheduler]



# def test_Denoise(net, dataset, sigma=15):
#     output_path = testopt.output_path + 'denoise/' + str(sigma) + '/'
#     subprocess.check_output(['mkdir', '-p', output_path])
    

#     dataset.set_sigma(sigma)
#     testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

#     psnr = AverageMeter()
#     ssim = AverageMeter()

#     with torch.no_grad():
#         for ([clean_name], degrad_patch, clean_patch) in tqdm(testloader):
#             degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

#             restored = net(degrad_patch)
#             temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

#             psnr.update(temp_psnr, N)
#             ssim.update(temp_ssim, N)
#             save_image_tensor(restored, output_path + clean_name[0] + '.png')

#         print("Denoise sigma=%d: psnr: %.2f, ssim: %.4f" % (sigma, psnr.avg, ssim.avg))



def test_DeTask(net, dataset, task):
    output_path = os.path.join(testopt.output_path, task)
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()
            
            restored = net(degrad_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor(restored, os.path.join(output_path, f'{degraded_name[0]}.png'))
        print(f"{task} PSNR: {psnr.avg:.2f}, SSIM: {ssim.avg:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--output_path', type=str, default="output/", 
                        help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="Combined_Endovis17.ckpt", 
                        help='checkpoint save path')
    
    parser.add_argument('--deoverexp_Test_input_dir', type=str, default='data_18/Test/deoverexp/overexp', 
                        help='Path to deoverexp input images')
    parser.add_argument('--deoverexp_Test_gt_dir', type=str, default='data_18/Test/deoverexp/gt', 
                        help='Path to deoverexp ground truth images')
    parser.add_argument('--delow_Test_input_dir', default='data_18/Test/delow/low', 
                        help='Path to delow input images')
    parser.add_argument('--delow_Test_gt_dir', default='data_18/Test/delow/gt',
                        help='Path to delow ground truth images')
    parser.add_argument('--dehazy_Test_input_dir', type=str, default='data_18/Test/dehazy/hazy', 
                        help='Path to dehazy input images')
    parser.add_argument('--dehazy_Test_gt_dir', type=str, default='data_18/Test/dehazy/gt', 
                        help='Path to dehazy ground truth images')
    parser.add_argument('--decombined_Test_input_dir', type=str, default='data_combined_17/Test/combined',
                    help='Path to decombined input images')
    parser.add_argument('--decombined_Test_gt_dir', type=str, default='data_combined_17/Test/gt',
                    help='Path to decombined ground truth images')
    testopt = parser.parse_args()
    
    

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)
    
    ckpt_path = os.path.join("ckpt", testopt.ckpt_name)
    
    print("CKPT name : {}".format(ckpt_path))
    print("start testing...")
    
    net = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()


    
    # 根据模式执行测试
    if testopt.mode == 0:
        delow_set = DeDataset(testopt, task="delow")
        test_DeTask(net, delow_set, "delow")
    elif testopt.mode == 1:
        deoverexp_set = DeDataset(testopt, task="deoverexp")
        test_DeTask(net, deoverexp_set, "deoverexp")
    elif testopt.mode == 2:
        dehazy_set = DeDataset(testopt, task="dehazy")
        test_DeTask(net, dehazy_set, "dehazy")
    elif testopt.mode == 3:
        decombined_set = DeDataset(testopt, task="decombined")
        test_DeTask(net, decombined_set, "decombined")
    elif testopt.mode == 4:
        # 全任务测试
        delow_set = DeDataset(testopt, task="delow")
        test_DeTask(net, delow_set, "delow")
        deoverexp_set = DeDataset(testopt, task="deoverexp")
        test_DeTask(net, deoverexp_set, "deoverexp")
        dehazy_set = DeDataset(testopt, task="dehazy")
        test_DeTask(net, dehazy_set, "dehazy")