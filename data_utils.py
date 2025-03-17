from os import listdir
from os.path import join
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
from torchvision.transforms import InterpolationMode
import random
import numpy as np
import torch
import cv2
from math import log10
import torchvision.models as models
import numpy
from scipy.linalg import sqrtm
from torchvision import transforms

def calculate_fid(images1, images2):

    transform = transforms.Compose([
        transforms.Resize(299),
        # transforms.CenterCrop(299),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    images1 = transform(images1)
    images2 = transform(images2)

    # calculate activations
    model = models.inception_v3(pretrained=True).cuda()
    act1 = model(images1)
    act2 = model(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(np.dot(sigma1, sigma2))
    # check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def p2ploss(fake, real, start_thred= 0.2, n_min=50000):
    N, C, H, W = fake.size()

    distance_squared = torch.sum(torch.abs(fake-real),dim=1).view(-1)
    similarity = distance_squared
    # max_distance_squared = torch.max(distance_squared)
    # similarity = distance_squared / max_distance_squared
    top_indices,_ = torch.sort(similarity)
    # top_indices = top_indices[top_indices>start_thred]

    thresh = torch.mean(top_indices)
    # self.thresh =torch.mean(loss)
    # cal = top_indices < thresh
    # num = cal.sum()
    # print(num)
    if top_indices[n_min] < thresh:  # 当loss大于阈值(由输入概率转换成loss阈值)的像素数量比n_min多时，取所有大于阈值的loss值
        loss = top_indices[top_indices < thresh]
    else:
        # print("hard sample")
        loss = top_indices[:n_min]
    # loss = top_indices
    # p = torch.exp(-loss)
    # loss = torch.exp(p) * loss
    return torch.mean(loss)

def cal_psnr(hr_tensor,sr_tensor):
    batchsize = hr_tensor.size()[0]
    n_chan = hr_tensor.size()[1]
    psnr = 0  
    for i in range(batchsize):
        for j in range(n_chan):
            mse = ((hr_tensor[i,j,:,:]-sr_tensor[i,j,:,:])**2).data.mean()
            single_psnr = 10 * log10(1/mse)
            psnr += single_psnr
    psnr = psnr/(batchsize*n_chan)
    return psnr       


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG', '.tif'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def hr_transform():
    return Compose([
        ToTensor()
    ])


def lr_transform(img_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(img_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir_lr, dataset_dir_hr, upscale_factor=8, patch_size=512, noise='.', ups=False):
        super(TrainDatasetFromFolder, self).__init__()
        dataset_dir = listdir(dataset_dir_lr)
        
        self.image_filenames_hr = [join(dataset_dir_hr, x) for x in dataset_dir if is_image_file(x)]
        self.image_filenames_lr = [join(dataset_dir_lr, x) for x in dataset_dir if is_image_file(x)]        
        # self.hr_transform = hr_transform()
        # self.lr_transform = lr_transform(img_size, upscale_factor)
        self.patch_size = patch_size
        self.scale = upscale_factor
        self.noise = noise
        self.ups = ups
        self.get_patch = get_patch
        self.augment = augment
        self.add_noise = add_noise
        self.set_channel = set_channel
        self.np2Tensor = np2Tensor

    def __getitem__(self, index):
        hr = cv2.imread(self.image_filenames_hr[index], cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        lr = cv2.imread(self.image_filenames_lr[index], cv2.IMREAD_COLOR)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        bicubic_down = cv2.resize(hr, (hr.shape[0]// self.scale, hr.shape[1]// self.scale), interpolation=cv2.INTER_CUBIC)
        lr, bicubic_down, hr = self.augment([lr, bicubic_down, hr])
        lr = self.add_noise(lr, self.noise)
        if self.ups:
            lr = cv2.resize(lr, (hr.shape[0], hr.shape[1]), interpolation=cv2.INTER_CUBIC)
        lr, bicubic_down, hr = self.set_channel([lr, bicubic_down, hr], 3)
        lr_tensor, bicubic_down_tensor, hr_tensor = self.np2Tensor([lr, bicubic_down, hr], 1)
        return lr_tensor, bicubic_down_tensor, hr_tensor

    def __len__(self):
        return len(self.image_filenames_hr)

class TrainDatasetFromFolder_2(Dataset):
    def __init__(self, dataset_dir_lr, dataset_dir_hr, upscale_factor=8, patch_size=512, noise='.', ups=True):
        super(TrainDatasetFromFolder_2, self).__init__()
        dataset_list_lr = listdir(dataset_dir_lr)
        dataset_list_hr = listdir(dataset_dir_hr)
        dataset_list = list(set(dataset_list_lr)&set(dataset_list_hr))
        self.image_filenames_hr = [join(dataset_dir_hr, x) for x in dataset_list if is_image_file(x)]
        self.image_filenames_lr = [join(dataset_dir_lr, x) for x in dataset_list if is_image_file(x)]
        # self.hr_transform = hr_transform()
        # self.lr_transform = lr_transform(img_size, upscale_factor)
        self.patch_size = patch_size
        self.scale = upscale_factor
        self.noise = noise
        self.ups = ups
        self.get_patch = get_patch
        self.augment = augment
        self.add_noise = add_noise
        self.set_channel = set_channel
        self.np2Tensor = np2Tensor

    def __getitem__(self, index):
        hr = cv2.imread(self.image_filenames_hr[index], cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        lr = cv2.imread(self.image_filenames_lr[index], cv2.IMREAD_COLOR)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        lr = cv2.resize(lr, (hr.shape[0]//self.scale* 2, hr.shape[1]//self.scale* 2), interpolation=cv2.INTER_CUBIC)
        bicubic_down = cv2.resize(hr, (hr.shape[0] // self.scale * 2, hr.shape[1] // self.scale * 2), interpolation=cv2.INTER_CUBIC)  # , interpolation=cv2.INTER_CUBIC
        # hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        # lr_image = self.lr_transform(hr_image)
        # lr, hr = self.get_patch(lr, hr, self.patch_size, self.scale)
        lr, bicubic_down, hr = self.augment([lr, bicubic_down, hr])
        lr = self.add_noise(lr, self.noise)
        lr = cv2.resize(lr, (hr.shape[0] // self.scale * 2, hr.shape[1] // self.scale * 2), interpolation=cv2.INTER_CUBIC)
        lr, bicubic_down, hr = self.set_channel([lr, bicubic_down, hr], 3)
        lr_tensor, bicubic_down_tensor, hr_tensor = self.np2Tensor([lr, bicubic_down, hr], 1)
        return lr_tensor, bicubic_down_tensor, hr_tensor

    def __len__(self):
        return len(self.image_filenames_hr)

class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir_lr, dataset_dir_hr, upscale_factor=8, patch_size=512, ups=False):
        super(ValDatasetFromFolder, self).__init__()
        dataset_dir = listdir(dataset_dir_hr)
        self.upscale_factor = upscale_factor
        self.image_filenames_hr = [join(dataset_dir_hr, x) for x in dataset_dir if is_image_file(x)]
        self.image_filenames_lr = [join(dataset_dir_lr, x) for x in dataset_dir if is_image_file(x)]
        self.patch_size = patch_size
        self.scale = upscale_factor
        self.ups = ups
        self.get_patch = get_patch
        self.np2Tensor = np2Tensor
        self.set_channel = set_channel
#        self.hr_transform = hr_transform()
#        self.lr_transform = lr_transform(img_size, upscale_factor)

    def __getitem__(self, index):
        hr = cv2.imread(self.image_filenames_hr[index], cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        lr = cv2.imread(self.image_filenames_lr[index], cv2.IMREAD_COLOR)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)

        lr, hr = self.get_patch(lr, hr, self.patch_size, self.scale)
        if self.ups:
            lr = cv2.resize(lr, (hr.shape[0], hr.shape[1]), interpolation=cv2.INTER_CUBIC)
        lr, hr = self.set_channel([lr, hr], 3)
        lr_tensor, hr_tensor = self.np2Tensor([lr, hr], 1)
        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.image_filenames_hr)

class ValDatasetFromFolder_2(Dataset):
    def __init__(self, dataset_dir_lr, dataset_dir_hr, upscale_factor=8, patch_size=512, ups=False):
        super(ValDatasetFromFolder_2, self).__init__()
        dataset_list_lr = listdir(dataset_dir_lr)
        dataset_list_hr = listdir(dataset_dir_hr)
        dataset_list = list(set(dataset_list_lr)&set(dataset_list_hr))
        self.image_filenames_hr = [join(dataset_dir_hr, x) for x in dataset_list if is_image_file(x)]
        self.image_filenames_lr = [join(dataset_dir_lr, x) for x in dataset_list if is_image_file(x)]
        self.patch_size = patch_size      
        self.scale = upscale_factor
        self.ups = ups
        self.get_patch = get_patch
        self.np2Tensor = np2Tensor
        self.set_channel = set_channel
#        self.hr_transform = hr_transform()
#        self.lr_transform = lr_transform(img_size, upscale_factor)

    def __getitem__(self, index):               
        hr = cv2.imread(self.image_filenames_hr[index], cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        lr = cv2.imread(self.image_filenames_lr[index], cv2.IMREAD_COLOR) 
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)               
        lr = cv2.resize(lr, (hr.shape[0]//self.scale* 2, hr.shape[1]//self.scale* 2), interpolation=cv2.INTER_CUBIC)

        # hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        # lr_image = self.lr_transform(hr_image)
        # lr, hr = self.get_patch(lr, hr, self.patch_size, self.scale)
#        lr, hr = self.augment([lr, hr])
#        lr = self.add_noise(lr, self.noise)
        lr, hr = self.set_channel([lr, hr], 3)
        lr_tensor, hr_tensor = self.np2Tensor([lr, hr], 1)
        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.image_filenames_hr)

class TestDatasetFromFolder_2(Dataset):
    def __init__(self, dataset_dir_lr, dataset_dir_hr, upscale_factor=8, ups=False):
        super(TestDatasetFromFolder_2, self).__init__()
        dataset_dir = listdir(dataset_dir_hr)
        self.image_filenames = dataset_dir
        self.upscale_factor = upscale_factor
        self.image_filenames_hr = [join(dataset_dir_hr, x) for x in dataset_dir if is_image_file(x)]
        self.image_filenames_lr = [join(dataset_dir_lr, x) for x in dataset_dir if is_image_file(x)]     
        self.scale = upscale_factor
        self.ups = ups
        self.get_patch = get_patch
        self.np2Tensor = np2Tensor
        self.set_channel = set_channel

    def __getitem__(self, index):               
        hr = cv2.imread(self.image_filenames_hr[index], cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        lr = cv2.imread(self.image_filenames_lr[index], cv2.IMREAD_COLOR) 
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        lr = cv2.resize(lr, (hr.shape[0] // self.scale * 2, hr.shape[1] // self.scale * 2),
                        interpolation=cv2.INTER_CUBIC)
        lr, hr = self.set_channel([lr, hr], 3)
        lr_tensor, hr_tensor = self.np2Tensor([lr, hr], 1)
        return lr_tensor, hr_tensor, self.image_filenames[index].split('.')[0]

    def __len__(self):
        return len(self.image_filenames_hr)

class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir_lr, dataset_dir_hr, upscale_factor=8, ups=False):
        super(TestDatasetFromFolder, self).__init__()
        dataset_dir = listdir(dataset_dir_hr)
        self.image_filenames = dataset_dir
        self.upscale_factor = upscale_factor
        self.image_filenames_hr = [join(dataset_dir_hr, x) for x in dataset_dir if is_image_file(x)]
        self.image_filenames_lr = [join(dataset_dir_lr, x) for x in dataset_dir if is_image_file(x)]
        self.scale = upscale_factor
        self.ups = ups
        self.get_patch = get_patch
        self.np2Tensor = np2Tensor
        self.set_channel = set_channel

    def __getitem__(self, index):
        hr = cv2.imread(self.image_filenames_hr[index], cv2.IMREAD_COLOR)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        lr = cv2.imread(self.image_filenames_lr[index], cv2.IMREAD_COLOR)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        if self.ups:
            lr = cv2.resize(lr, (hr.shape[0], hr.shape[1]),interpolation=cv2.INTER_CUBIC)
        lr, hr = self.set_channel([lr, hr], 3)
        lr_tensor, hr_tensor = self.np2Tensor([lr, hr], 1)
        return lr_tensor, hr_tensor, self.image_filenames[index].split('.')[0]

    def __len__(self):
        return len(self.image_filenames_hr)


def get_patch(img_in, img_tar, patch_size, scale, multi_scale=False):
    ih, iw = img_in.shape[:2]
    """randomly get patches from lr and hr images """
    p = scale if multi_scale else 1
    tp = p * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    img_in = img_in[iy:iy + ip, ix:ix + ip, :]
    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]
    return img_in, img_tar


def set_channel(l, n_channel):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channel == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channel == 3 and c == 1:
            img = np.concatenate([img] * n_channel, 2)

        return img

    return [_set_channel(_l) for _l in l]


def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(_l) for _l in l]


def add_noise(x, noise='.'):
    """ add lr with noise """
    if noise is not '.':
        noise_type = noise[0]
        noise_value = int(noise[1:])
        if noise_type == 'G':
            noises = np.random.normal(scale=noise_value, size=x.shape)
            noises = noises.round()
        elif noise_type == 'S':
            noises = np.random.poisson(x * noise_value) / noise_value
            noises = noises - noises.mean(axis=0).mean(axis=0)

        x_noise = x.astype(np.int16) + noises.astype(np.int16)
        x_noise = x_noise.clip(0, 255).astype(np.uint8)
        return x_noise
    else:
        return x


def augment(l, hflip=True, rot=True):
    """augment train data by horizontal flip, vertical flip and rotation"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]

