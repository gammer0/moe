
from PIL import Image
import numpy as np
from os import listdir, mkdir
from os.path import isfile, join, isdir
from tqdm import tqdm
import glob
from tqdm import tqdm
import torch
from torch.autograd import Variable
import time


import sys
sys.path.insert(1, '../1_Train_deep_model')
from utils import PSNR, _rgb2ycbcr, _load_img_array

from moeNet import MOENet



# USER PARAMS
SAMPLING_INTERVAL = "001000"        # N bit uniform sampling
START_ITER = "001000" 
TEST_DIR = './test/'      # Test images

model_G =MOENet().cuda()

lm = torch.load('model_G_i{}.pth'.format(START_ITER), weights_only=False)
model_G.load_state_dict(lm.state_dict(), strict=True)




# Test LR images
files_lr = glob.glob(TEST_DIR + '/fog_1080/*.png')
files_lr.sort()

# Test GT images
files_gt = glob.glob(TEST_DIR + '/GT_1080/*.png')
files_gt.sort()


psnrs = []

if not isdir('./output'):
    mkdir('./output')

def moe(model_G, files_lr, files_gt, psnrs):
    for ti, fn in enumerate(tqdm(files_gt)):
    # Load LR image
        img_lr = _load_img_array(files_lr[ti]).astype(np.float32)
        img_lr = img_lr.transpose((2,0,1))
        img_lr =img_lr.reshape(-1,*img_lr.shape)

    # Load GT image
        img_gt = _load_img_array(files_gt[ti])
        img_gt =np.asarray(img_gt).astype(np.float32)

    
    # Rotational ensemble
        img_lr =Variable(torch.from_numpy(img_lr)).cuda()
        out_r0 = model_G(img_lr)


        out_r0 =torch.clamp(out_r0, -1, 1)*255
        img_out = out_r0 / 255.0
        img_out = img_out[0,:,:,:].data.cpu().numpy().transpose((1,2,0))
        img_out = np.round(np.clip(img_out, 0, 1) * 255).astype(np.uint8)

    # Matching image sizes 
        if img_gt.shape[0] < img_out.shape[0]:
            img_out = img_out[:img_gt.shape[0]]
        if img_gt.shape[1] < img_out.shape[1]:
            img_out = img_out[:, :img_gt.shape[1]]

        if img_gt.shape[0] > img_out.shape[0]:
            img_out = np.pad(img_out, ((0,img_gt.shape[0]-img_out.shape[0]),(0,0),(0,0)))
        if img_gt.shape[1] > img_out.shape[1]:
            img_out = np.pad(img_out, ((0,0),(0,img_gt.shape[1]-img_out.shape[1]),(0,0)))

    # Save to file
        
        Image.fromarray(img_out).save(f'./output/{time.time()}.png')

        CROP_S = 4
        psnr = PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(img_out)[:,:,0], CROP_S)
        psnrs.append(psnr)

with torch.no_grad():
    moe(model_G, files_lr, files_gt, psnrs)

print('AVG PSNR: {}'.format(np.mean(np.asarray(psnrs))))

