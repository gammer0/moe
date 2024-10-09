

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from PIL import Image
import numpy as np
import time
from os import mkdir
from os.path import join, isdir
from tqdm import tqdm
import glob

from moeNet import MOENet
from utils import PSNR, GeneratorEnqueuer, DirectoryIterator_DIV2K, _load_img_array, _rgb2ycbcr
from tensorboardX import SummaryWriter


### USER PARAMS ###
EXP_NAME = "moe"
VERSION = ""    

NB_BATCH = 32       # mini-batch
CROP_SIZE = 128     # input LR training patch size

START_ITER = 0      # Set 0 for from scratch, else will load saved params and trains further
NB_ITER =200000    # Total number of training iterations

I_DISPLAY = 100     # display info every N iteration
I_VALIDATION = 1000  # validate every N iteration
I_SAVE = 1000       # save models every N iteration

TRAIN_DIR = './train/'  # Training images: png files should just locate in the directory (eg ./train/img0001.png ... ./train/img0800.png)
VAL_DIR = './val_1080/'      # Validation images

LR_G = 1e-4         # Learning rate for the generator

### Tensorboard for monitoring ###
writer = SummaryWriter(log_dir='./log/{}'.format(str(VERSION)))



model_G = MOENet().cuda()



## Optimizers
params_G = list(filter(lambda p: p.requires_grad, model_G.parameters()))
opt_G = optim.Adam(params_G, lr=LR_G)



## Load saved params
if START_ITER > 0:
    lm = torch.load('checkpoint/{}/model_G_i{:06d}.pth'.format(str(VERSION), START_ITER))
    model_G.load_state_dict(lm.state_dict(), strict=True)

    lm = torch.load('checkpoint/{}/opt_G_i{:06d}.pth'.format(str(VERSION), START_ITER))
    opt_G.load_state_dict(lm.state_dict())


# Training dataset
Iter_H = GeneratorEnqueuer(DirectoryIterator_DIV2K( 
                                datadir = TRAIN_DIR,
                                crop_size = CROP_SIZE, 
                                crop_per_image = NB_BATCH//4,
                                out_batch_size = NB_BATCH,
                                shuffle=True))
Iter_H.start(max_q_size=16, workers=4)


## Prepare directories
if not isdir('checkpoint'):
    mkdir('checkpoint')
if not isdir('result'):
    mkdir('result')
if not isdir('checkpoint/{}'.format(str(VERSION))):
    mkdir('checkpoint/{}'.format(str(VERSION)))
if not isdir('result/{}'.format(str(VERSION))):
    mkdir('result/{}'.format(str(VERSION)))




## Some preparations 
print('===> Training start')
l_accum = [0.,0.,0.]
dT = 0.
rT = 0.
accum_samples = 0


def SaveCheckpoint(i, best=False):
    str_best = ''
    if best:
        str_best = '_best'

    torch.save(model_G, 'checkpoint/{}/model_G_i{:06d}{}.pth'.format(str(VERSION), i, str_best ))
    torch.save(opt_G, 'checkpoint/{}/opt_G_i{:06d}{}.pth'.format(str(VERSION), i, str_best))
    print("Checkpoint saved")



### TRAINING
for i in tqdm(range(START_ITER+1, NB_ITER+1)):

    model_G.train()

    # Data preparing
    st = time.time()
    batch_Fog, batch_GT = Iter_H.dequeue()
    batch_GT = Variable(torch.from_numpy(batch_GT)).cuda()      # BxCxHxW, range [0,1]
    batch_Fog = Variable(torch.from_numpy(batch_Fog)).cuda()      # BxCxHxW, range [0,1]
    dT += time.time() - st


    ## TRAIN G
    st = time.time()
    opt_G.zero_grad()

    # Rotational ensemble training
    batch_S = model_G(batch_Fog)

    batch_S =torch.clamp(batch_S,-1,1)*255
    batch_S /= 255.0

    loss_Pixel = torch.mean( ((batch_S - batch_GT)**2)  )
    loss_G = loss_Pixel

    # Update
    loss_G.backward()
    opt_G.step()
    rT += time.time() - st

    # For monitoring
    accum_samples += NB_BATCH
    l_accum[0] += loss_Pixel.item()


    ## Show information
    if i % I_DISPLAY == 0:
        writer.add_scalar('loss_Pixel', l_accum[0]/I_DISPLAY, i)

        print("{} {}| Iter:{:6d}, Sample:{:6d}, GPixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
            EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
        l_accum = [0.,0.,0.]
        dT = 0.
        rT = 0.


    ## Save models
    if i % I_SAVE == 0:
        SaveCheckpoint(i)


    # Validation
    if i % I_VALIDATION == 0:
        with torch.no_grad():
            model_G.eval()

            # Test for validation images
            files_fog = glob.glob(VAL_DIR + '/fog/*.png')
            files_fog.sort()
            files_gt = glob.glob(VAL_DIR + '/GT/*.png')
            files_gt.sort()

            psnrs = []
            lpips = []

            for ti, fn in enumerate(files_gt):
                # Load HR image
                tmp = _load_img_array(files_gt[ti])
                val_g = np.asarray(tmp).astype(np.float32)  # HxWxC

                # Load LR image
                tmp = _load_img_array(files_fog[ti])
                val_f = np.asarray(tmp).astype(np.float32)  # HxWxC
                val_f = np.transpose(val_f, [2, 0, 1])      # CxHxW
                val_f = val_f.reshape(-1,*val_f.shape)            # BxCxHxW

                val_f = Variable(torch.from_numpy(val_f.copy()), volatile=True).cuda()
                        
                # Run model
                batch_S = model_G(val_f)

                
                batch_S =torch.clamp(batch_S,-1,1)*255
                batch_S /= 255.0


                # Output 
                image_out = (batch_S).cpu().data.numpy()
                image_out = np.clip(image_out[0], 0. , 1.)      # CxHxW
                image_out = np.transpose(image_out, [1, 2, 0])  # HxWxC

                # Save to file
                image_out = ((image_out)*255).astype(np.uint8)
                Image.fromarray(image_out).save('./result/{}'.format(fn.split('\\')[-1]))
 
                # PSNR on Y channel
                img_gt = (val_g*255).astype(np.uint8)
                CROP_S = 0
                psnrs.append(PSNR(_rgb2ycbcr(img_gt)[:,:,0], _rgb2ycbcr(image_out)[:,:,0], CROP_S))

        print('AVG PSNR: Validation: {}'.format(np.mean(np.asarray(psnrs))))

        writer.add_scalar('PSNR_valid', np.mean(np.asarray(psnrs)), i)
        writer.flush()
