import argparse
import os
import copy
import itertools
import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.autograd import Variable
from models import Generator
from models import Discriminator
from data_utils import TrainDatasetFromFolder, ValDatasetFromFolder, TrainDatasetFromFolder_2,ValDatasetFromFolder_2,p2ploss,cal_psnr
from math import log10
import pytorch_ssim
from utils import weights_init_normal
from utils import ReplayBuffer
from pathlib import Path
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--upscale_factor', type=int, default=8)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--patch_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--num-epochs', type=int, default=500)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--cuda', action='store_false', help='use GPU computation')
parser.add_argument('--save_path', type=str, default='SDDT-SR/')
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if __name__ == '__main__':
    opt = parser.parse_args()
    Path(opt.save_path).mkdir(parents=True, exist_ok=True)
    CROP_SIZE = opt.img_size
    IMG_SIZE = opt.img_size
    UPSCALE_FACTOR = opt.upscale_factor
    PATCH_SIZE = opt.patch_size
    NUM_EPOCHS = opt.num_epochs
    BATCHSIZE = opt.batch_size

    netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
    netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)
    netD_A = Discriminator(opt.input_nc).to(device)
    netD_B = Discriminator(opt.output_nc).to(device)

    netG_sr = Generator(opt.input_nc, opt.output_nc).to(device)
    netD_sr = Discriminator(opt.input_nc).to(device)
    netD_consist = Discriminator(opt.input_nc).to(device)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG_A2B = torch.nn.DataParallel(netG_A2B, device_ids=range(torch.cuda.device_count()))
        netG_B2A = torch.nn.DataParallel(netG_B2A, device_ids=range(torch.cuda.device_count()))
        netD_A = torch.nn.DataParallel(netD_A, device_ids=range(torch.cuda.device_count()))
        netD_B = torch.nn.DataParallel(netD_B, device_ids=range(torch.cuda.device_count()))
        netG_sr = torch.nn.DataParallel(netG_sr, device_ids=range(torch.cuda.device_count()))
        netD_sr = torch.nn.DataParallel(netD_sr, device_ids=range(torch.cuda.device_count()))
        netD_consist = torch.nn.DataParallel(netD_consist, device_ids=range(torch.cuda.device_count()))

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
    netG_sr.apply(weights_init_normal)
    netD_sr.apply(weights_init_normal)
    netD_consist.apply(weights_init_normal)

    # Lossess
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()
    criterion_sr = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr)
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=[0.5,0.999])
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=[0.5,0.999])
    optimizer_G_sr = torch.optim.Adam(netG_sr.parameters(), lr=opt.lr, betas=[0.5,0.999])
    optimizer_D_sr = torch.optim.Adam(netD_sr.parameters(), lr=opt.lr, betas=[0.5,0.999])
    optimizer_D_consist = torch.optim.Adam(netD_sr.parameters(), lr=opt.lr, betas=[0.5,0.999])



    train_set = TrainDatasetFromFolder_2('/public/lsm/Data/SGSRD/sentinel',
                                           '/public/lsm/Data/SGSRD/google',
                                           upscale_factor=UPSCALE_FACTOR, patch_size=PATCH_SIZE)
    val_set = ValDatasetFromFolder_2('/public/lsm/Data/SGSRD/sentinel_val',
                                       '/public/lsm/Data/SGSRD/google_val', patch_size=PATCH_SIZE,
                                       upscale_factor=UPSCALE_FACTOR)

    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=BATCHSIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False, drop_last=True)
    Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
    target_real = Variable(Tensor(BATCHSIZE).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(BATCHSIZE).fill_(0.0), requires_grad=False)
    rs = transforms.Resize((512, 512))
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    results = {'loss': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'loss': 0}
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()
        netG_sr.train()
        netD_sr.train()
        netD_consist.train()
        psnr = 0
        ssim = 0
        #        for data, target, filenames in train_bar:
        for input, bicubic, target in train_bar:
            g_update_first = True
            batch_size = input.size(0)
            running_results['batch_sizes'] += batch_size
            input = Variable(input).cuda()
            bicubic = Variable(bicubic).cuda()
            target = Variable(target).cuda()

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()
            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(bicubic)
            loss_identity_B = criterion_identity(same_B, bicubic) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(input)
            loss_identity_A = criterion_identity(same_A, input) * 5.0

            # GAN loss
            fake_B = netG_A2B(input)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            # P2P loss
            loss_p2p = p2ploss(fake_B, bicubic) * 10

            fake_A = netG_B2A(bicubic)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, input) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, bicubic) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_p2p
            loss_G.backward()

            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(input)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(bicubic)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()

            ########################################

            ###### SR ######
            optimizer_G_sr.zero_grad()

            lr_rs_resize = rs(fake_B)
            lr_uav_resize = rs(bicubic)
            sr_rs = netG_sr(lr_rs_resize)
            sr_uav = netG_sr(lr_uav_resize)
            loss_sr_uav = criterion_sr(sr_uav, target)
            # loss_sr_rs = p2ploss(sr_rs, target)
            loss_sr = loss_sr_uav * 10
            pred_sr_uav = netD_sr(sr_uav)
            pred_sr_rs = netD_sr(sr_rs)
            sr_rs_consist = netD_consist(sr_rs)
            sr_uav_consist = netD_consist(sr_uav)
            loss_sr_uav_g = criterion_GAN(pred_sr_uav, target_real)
            loss_sr_rs_g = criterion_GAN(pred_sr_rs, target_real)
            loss_sr_rs_consist = criterion_GAN(sr_rs_consist, target_real)
            loss_sr_uav_consist = criterion_GAN(sr_uav_consist, target_real)
            loss_g = loss_sr + loss_sr_uav_g + loss_sr_rs_g + loss_sr_rs_consist + loss_sr_uav_consist

            loss_g.backward(retain_graph=True)
            optimizer_G_sr.step()

            optimizer_D_sr.zero_grad()
            sr_rs = netG_sr(lr_rs_resize)
            sr_uav = netG_sr(lr_uav_resize)
            pred_sr_uav = netD_sr(sr_uav)
            pred_sr_rs = netD_sr(sr_rs)
            pred_sr_uav_real = netD_sr(hr)
            loss_sr_uav_d_real = criterion_GAN(pred_sr_uav_real, target_real)
            loss_sr_uav_d = criterion_GAN(pred_sr_uav, target_fake)
            loss_sr_rs_d = criterion_GAN(pred_sr_rs, target_fake)
            loss_d = (loss_sr_uav_d_real+loss_sr_rs_d + loss_sr_uav_d) / 3.0

            loss_d.backward(retain_graph=True)
            optimizer_D_sr.step()

            optimizer_D_consist.zero_grad()
            sr_rs_consist = netD_consist(sr_rs)
            sr_uav_consist = netD_consist(sr_uav)
            loss_sr_uav_consist = criterion_GAN(sr_uav_consist, target_real)
            loss_sr_rs_consist = criterion_GAN(sr_rs_consist, target_fake)

            loss_consist = loss_sr_uav_consist + loss_sr_rs_consist
            loss_consist.backward(retain_graph=True)
            optimizer_D_consist.step()

            # loss for current batch before optimization
            running_results['loss'] += loss_sr.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss: %.4f' % (
                epoch, NUM_EPOCHS, running_results['loss'] / running_results['batch_sizes']))

        netG_A2B.eval()
        netG_B2A.eval()
        netD_A.eval()
        netD_B.eval()
        netG_sr.eval()
        netD_sr.eval()
        netD_consist.eval()


        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'psnrs': 0, 'ssim': 0, 'clipscore': 0, 'batch_sizes': 0}
            val_images = []
            for val_input, val_target in val_bar:
                #            for val_lr, val_hr, filenames in val_bar:
                batch_size = val_target.size(0)
                valing_results['batch_sizes'] += batch_size
                if torch.cuda.is_available():
                    lr = val_input.cuda()
                    hr = val_target.cuda()
                #                lr = F.interpolate(lr, size=hr.size()[2:], mode='bicubic',align_corners=True)
                sr = netG_A2B(lr)
                sr = rs(sr)
                sr = netG_sr(sr)
                sr = sr[:, :, UPSCALE_FACTOR:-UPSCALE_FACTOR, UPSCALE_FACTOR:-UPSCALE_FACTOR]
                hr = hr[:, :, UPSCALE_FACTOR:-UPSCALE_FACTOR, UPSCALE_FACTOR:-UPSCALE_FACTOR]
                sr = sr.mul(255).clamp(0, 255).round().div(255)
                psnr = cal_psnr(hr, sr)

                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnrs'] += psnr * batch_size
                valing_results['psnr'] = valing_results['psnrs'] / valing_results['batch_sizes']
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']

                val_bar.set_description(desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))


        # save model parameters
        torch.save(netG_A2B.state_dict(), opt.save_path + 'netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), opt.save_path + 'netG_B2A.pth')
        torch.save(netD_A.state_dict(), opt.save_path + 'netD_A.pth' )
        torch.save(netD_B.state_dict(), opt.save_path + 'netD_B.pth' )
        torch.save(netG_sr.state_dict(), opt.save_path + 'netG_sr.pth')
        torch.save(netD_sr.state_dict(), opt.save_path + 'netD_sr.pth')
        torch.save(netD_consist.state_dict(), opt.save_path + 'netD_consist.pth')
        # save loss\scores\psnr\ssim
        results['loss'].append(running_results['loss'] / running_results['batch_sizes'])
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])

        out_path = opt.save_path
        data_frame = pd.DataFrame(
            data={'Loss': results['loss'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
            index=range(1, epoch + 1))
