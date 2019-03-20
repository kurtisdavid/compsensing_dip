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

dataloader = utils.get_data(args) # get dataset of images

for num_meas in NUM_MEASUREMENTS_LIST:
    args.NUM_MEASUREMENTS = num_meas 
    
    # init measurement matrix
    A = baselines.get_A(args.IMG_SIZE*args.IMG_SIZE*args.NUM_CHANNELS, args.NUM_MEASUREMENTS)

    for c, (batch, _, im_path) in enumerate(dataloader):
        print(batch.shape)
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
                estimator = cs_dip.dip_estimator(args,module)
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
            
            x_hat = estimator(A, y, args)
            np.save('noisy_' + str(c),y)
            utils.save_reconstruction(x_hat, args, im_path)

if NEW_RECONS == False:
    print('Duplicate reconstruction configurations. No new data generated.')
else:
    print('Reconstructions generated!')
