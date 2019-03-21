import numpy as np
import pickle as pkl
import os
import parser

import torch
from torchvision import datasets

import utils
import cs_dip
import baselines as baselines 
import time
from pyr_utils import PyrDown

NEW_RECONS = False

args = parser.parse_args('configs.json')
print(args)

NUM_MEASUREMENTS_LIST, ALG_LIST = utils.convert_to_list(args)

# WE ASSUME dataset folder is all frames of a video
dataloader = utils.get_data(args) # get dataset of images

for num_meas in NUM_MEASUREMENTS_LIST:
    args.NUM_MEASUREMENTS = num_meas 
    
    # init measurement matrix
    A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, args.NUM_MEASUREMENTS)
    
    net = None
    z = None
    gan_logger = []
    z_logger = []
    # evolutions
    noisy = []
    z_s = []
    x_hats = []
    for c, (batch, _, im_path) in enumerate(dataloader):
        x = batch.view(1,-1).cpu().numpy() # define image
        y = x @ A
        for alg in ALG_LIST:
            args.ALG = alg

            if utils.recons_exists(args, im_path): # to avoid redundant reconstructions
                continue
            NEW_RECONS = True

            if alg == 'csdip' and not args.GAUSSIAN_PYR:
                estimator = cs_dip.dip_estimator(args)
            elif alg == 'csdip' and args.GAUSSIAN_PYR:
                module = PyrDown(channels = args.NUM_CHANNELS, n_levels = args.N_LEVELS)
                z_estimator, gan_estimator  = cs_dip.vid_dip_estimator(args,module)
                y = module(torch.from_numpy(np.reshape(x,(1,args.NUM_CHANNELS,args.IMG_SIZE,args.IMG_SIZE))).cuda()).view(-1,1).cpu()
            elif alg == 'dct':
                estimator = baselines.lasso_dct_estimator(args)
            elif alg == 'wavelet':
                estimator = baselines.lasso_wavelet_estimator(args)
            elif alg == 'bm3d' or alg == 'tval3':
                raise NotImplementedError('BM3D-AMP and TVAL3 are implemented in Matlab. \
                                            Please see GitHub repository for details.')
            else:
                raise NotImplementedError
            
             
            np.save('vid_noisy_' + str(c),y)
           
            # EM training, don't train z on first frame, need trained gan first... 
            if net is not None:
                z = z_estimator(net, z, A, y, args, z_logger)
            net, z, x_hat = gan_estimator(A, y, args, gan_logger, net=net, z=z)
            # z shouldn't change if net is not None, since that is not trainable in gan_estimator
            z_s.append(z.data.cpu().numpy())
            x_hats.append(x_hat)          
                     
            # utils.save_reconstruction(x_hat, args, im_path)
    np.save('gan_losses',gan_logger)
    np.save('z_losses',z_logger)
    np.save('noisy',noisy)
    np.save('z_s', z_s)
    np.save('x_hats', x_hats)


if NEW_RECONS == False:
    print('Duplicate reconstruction configurations. No new data generated.')
else:
    print('Reconstructions generated!')
