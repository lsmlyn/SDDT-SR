import argparse
import os
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from models import Generator
from data_utils import TestDatasetFromFolder_2,cal_psnr,display_transform
import pytorch_ssim
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn.functional as F
import clip

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=8)
parser.add_argument('--upscale_factor', type=int, default=8)
parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--patch_size', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_path', type=str, default='SDDT-SR/images/')
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    opt = parser.parse_args()
    Path(opt.save_path).mkdir(parents=True, exist_ok=True)
    CROP_SIZE = opt.img_size
    IMG_SIZE = opt.img_size
    UPSCALE_FACTOR = opt.upscale_factor
    PATCH_SIZE = opt.patch_size
    BATCHSIZE = opt.batch_size
    

    test_set = TestDatasetFromFolder_2('/Data/SGSRD/sentinel_test',
                                       '/Data/SGSRD/google_test',
                                       upscale_factor=UPSCALE_FACTOR)

    test_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

    netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
    netG_sr = Generator(opt.input_nc, opt.output_nc).to(device)
    rs = transforms.Resize((512, 512))

    clip_model, _ = clip.load("RN50")
    clip_model = clip_model.encode_image

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        netG_A2B = torch.nn.DataParallel(netG_A2B, device_ids=range(torch.cuda.device_count()))
        netG_sr = torch.nn.DataParallel(netG_sr, device_ids=range(torch.cuda.device_count()))

    netG_A2B.load_state_dict(torch.load("SDDT-SR/netG_A2B.pth"))
    netG_sr.load_state_dict(torch.load("SDDT-SR/netG_sr.pth"))

    criterion = nn.MSELoss()

    netG_A2B.eval()
    netG_sr.eval()

    out_path = opt.save_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    result_path = opt.save_path + "result_images/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        testing_results = {'mse': 0, 'fids': 0,'fid': 0, 'ssims': 0, 'psnr': 0, 'psnrs': 0,'ssim': 0, 'clipscore': 0, 'clipscores': 0,'batch_sizes': 0}
        test_images = []
        images_name = []
        for test_lr, test_hr, im_name in test_bar:
            batch_size = test_hr.size(0)
            testing_results['batch_sizes'] += batch_size
            lr = test_lr
            hr = test_hr
            if torch.cuda.is_available():
                lr = lr.cuda()
                hr = hr.cuda()
            sr = netG_A2B(lr)
            sr = rs(sr)
            sr = netG_sr(sr)

            OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
            OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

            x1 = F.interpolate(sr, (224, 224))
            x2 = F.interpolate(hr, (224, 224))

            nor = transforms.Normalize(mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)

            clipscore_sr = clip_model(nor(x1))
            clipscore_hr = clip_model(nor(x2))

            clipscore = F.cosine_similarity(clipscore_sr, clipscore_hr).mean()

            psnr = cal_psnr(hr, sr)

            batch_mse = ((sr - hr) ** 2).data.mean()
            testing_results['mse'] += batch_mse * batch_size
            testing_results['clipscores'] += clipscore * batch_size
            # testing_results['fids'] += fid * batch_size
            batch_ssim = pytorch_ssim.ssim(sr, hr).item()
            testing_results['ssims'] += batch_ssim * batch_size
            testing_results['psnrs'] += psnr * batch_size

            testing_results['psnr'] = testing_results['psnrs'] / testing_results['batch_sizes']
            testing_results['ssim'] = testing_results['ssims'] / testing_results['batch_sizes']
            testing_results['clipscore'] = testing_results['clipscores'] / testing_results['batch_sizes']

            test_bar.set_description(desc='FID: %.4f Clipscore: %.4f PSNR: %.4f dB SSIM: %.4f'
                                         % (testing_results['fid'], testing_results['clipscore'], testing_results['psnr'], testing_results['ssim']))
            result_images = display_transform()(sr.data.cpu().squeeze(0))
            test_images.extend([display_transform()(rs(lr).data.cpu().squeeze(0)),
                                display_transform()(sr.data.cpu().squeeze(0)),
                                display_transform()(hr.data.cpu().squeeze(0))])
            images_name.append(im_name[0])
            utils.save_image(result_images, result_path + '%s.png' % (im_name[0]), padding=5)

        test_images = test_images[:(len(test_images)//3) * 3]
        test_images = torch.stack(test_images)
        test_images = torch.chunk(test_images, test_images.size(0) // 3)
        test_save_bar = tqdm(test_images, desc='[saving training results]')

        index = 0
        for image in test_save_bar:
            image = utils.make_grid(image, nrow=3, padding=5)
            utils.save_image(image, out_path + '%s.png' % (images_name[index]), padding=5)
            index += 1