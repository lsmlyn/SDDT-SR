import os
from PIL import Image
import numpy as np
from lpips import LPIPS
import torchvision.transforms as transforms
import torch

loss_fn = LPIPS(net='vgg')


def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC) 
    img_np = np.array(img).astype(np.float32) / 127.5 - 1 
    img_np = img_np[np.newaxis, :]
    img_tensor = torch.from_numpy(img_np).to(torch.float32).permute(0,3,1,2)
    return img_tensor


def calculate_lpips(folder1, folder2):
    lpips = []
    for filename in os.listdir(folder1):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg','tif')):
            name = filename.split('.')[0]
            img_path1 = os.path.join(folder1, name+ ".tif")
            img_path2 = os.path.join(folder2, name+ ".png")

            if os.path.exists(img_path2):
                img1 = load_image(img_path1)
                img2 = load_image(img_path2)

                dist = loss_fn.forward(img1, img2).item()
                lpips.append(dist)
            else:
                print(f"Warning: {filename} not found in {folder2}")

    print("lpips: %.4f"%(np.mean(lpips)))


folder1 = 'google_test'
folder2 = 'result_images'

calculate_lpips(folder1, folder2)