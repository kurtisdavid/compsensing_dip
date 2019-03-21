import numpy as np
import parser
import torch
from torch.autograd import Variable
import baselines

import utils
import time

args = parser.parse_args('configs.json') 

CUDA = torch.cuda.is_available()
dtype = utils.set_dtype(CUDA)
se = torch.nn.MSELoss(reduction='none').type(dtype)

BATCH_SIZE = 1
EXIT_WINDOW = 51
loss_re, recons_re = utils.init_output_arrays(args)

# performs EM-like optimization, trading latent z training and GAN training.
def vid_dip_estimator(args, custom=None):
    def z_estimator(net, z_, A_val, y_batch_val, args, z_logger):
        # make it trainable
        z = Variable(z_.copy(), requires_grad=True)
        z_param = torch.nn.Parameter(z)
        optim = torch.optim.SGD([z_param], lr=0.01, momentum=0.9)
        if CUDA:
            net.cuda()
        z_iter = []
        loss_iter = [] 
        for i in range(args.z_NUM_ITER):
            optim.zero_grad()
            G = net(z)
            if custom is None:
                AG = torch.matmul(G.view(BATCH_SIZE,-1),A) # A*G(z)
            else:
                AG = custom(G).view(-1,1)
            y_loss = torch.mean(torch.sum(se(AG,y),dim=1))
            # calculate total variation loss 
            tv_loss = (torch.sum(torch.abs(G[:,:,:,:-1] - G[:,:,:,1:]))\
                    + torch.sum(torch.abs(G[:,:,:-1,:] - G[:,:,1:,:]))) 

            total_loss = y_loss + tvc*tv_loss               
            # stopping condition to account for optimizer convergence
            if i >= args.NUM_ITER - EXIT_WINDOW: 
                z_iter.append(z.data.cpu().numpy())
                loss_iter.append(total_loss.data.cpu().numpy())

            total_loss.backward() # backprop
            optim.step() # only update the latent vector
        z_logger.append(loss_iter)
        idx_re = np.argmin(loss_iter,axis=0)
        z_hat = recons_iter[idx_re]

        return z_hat
    def gan_estimator(A_val, y_batch_val, args, gan_logger, net=None, z=None):
        y = torch.FloatTensor(y_batch_val).type(dtype) # init measurements y
        A = torch.FloatTensor(A_val).type(dtype)       # init measurement matrix A

        mu, sig_inv, tvc, lrc = utils.get_constants(args, dtype)
        # should only be true for both at first iteration (train weights first...) 
        if net is None:
            net = utils.init_dcgan(args)
        if z is None: 
            z = torch.zeros(BATCH_SIZE*args.Z_DIM).type(dtype).view(BATCH_SIZE,args.Z_DIM,1,1)
            z.data.normal_().type(dtype) #init random input seed
        else:
            z = z.copy()
            z.requires_grad = False
        
        if CUDA:
            net.cuda() # cast network to GPU if available
    
        optim = torch.optim.RMSprop(net.parameters(),lr=0.001, momentum=0.9, weight_decay=0)
        loss_iter = []
        recons_iter = []
        
        for i in range(args.NUM_ITER):

            optim.zero_grad()

            # calculate measurement loss || y - A*G(z) ||
            G = net(z)
            if custom is None:
                AG = torch.matmul(G.view(BATCH_SIZE,-1),A) # A*G(z)
            else:
                AG = custom(G).view(-1,1)
            y_loss = torch.mean(torch.sum(se(AG,y),dim=1))
            # calculate total variation loss 
            tv_loss = (torch.sum(torch.abs(G[:,:,:,:-1] - G[:,:,:,1:]))\
                    + torch.sum(torch.abs(G[:,:,:-1,:] - G[:,:,1:,:]))) 

                # calculate learned regularization loss
#                layers = net.parameters()
#                layer_means = torch.cat([layer.mean().view(1) for layer in layers])
#                lr_loss = torch.matmul(layer_means-mu,torch.matmul(sig_inv,layer_means-mu))
                
#                total_loss = y_loss + lrc*lr_loss + tvc*tv_loss # total loss for iteration i
            total_loss = y_loss + tvc*tv_loss               
            # stopping condition to account for optimizer convergence
            if i >= args.NUM_ITER - EXIT_WINDOW: 
                recons_iter.append(G.data.cpu().numpy())
                loss_iter.append(total_loss.data.cpu().numpy())
            
            total_loss.backward() # backprop
            optim.step()
        # to view later...
        gan_logger.append(loss_iter)
        
        idx_re = np.argmin(loss_iter,axis=0)
        x_hat = recons_iter[idx_re]
        return net, z, x_hat
    
    return z_estimator, gan_estimator  

def dip_estimator(args, custom=None):
    def estimator(A_val, y_batch_val, args):

        y = torch.FloatTensor(y_batch_val).type(dtype) # init measurements y
        A = torch.FloatTensor(A_val).type(dtype)       # init measurement matrix A

        mu, sig_inv, tvc, lrc = utils.get_constants(args, dtype)

        for j in range(args.NUM_RESTARTS):
            
            net = utils.init_dcgan(args)

            z = torch.zeros(BATCH_SIZE*args.Z_DIM).type(dtype).view(BATCH_SIZE,args.Z_DIM,1,1)
            z.data.normal_().type(dtype) #init random input seed
            if CUDA:
                net.cuda() # cast network to GPU if available
            
            optim = torch.optim.RMSprop(net.parameters(),lr=0.001, momentum=0.9, weight_decay=0)
            loss_iter = []
            recons_iter = [] 

            for i in range(args.NUM_ITER):

                optim.zero_grad()

                # calculate measurement loss || y - A*G(z) ||
                G = net(z)
                if custom is None:
                    AG = torch.matmul(G.view(BATCH_SIZE,-1),A) # A*G(z)
                else:
                    AG = custom(G).view(-1,1)
                y_loss = torch.mean(torch.sum(se(AG,y),dim=1))
                # calculate total variation loss 
                tv_loss = (torch.sum(torch.abs(G[:,:,:,:-1] - G[:,:,:,1:]))\
                            + torch.sum(torch.abs(G[:,:,:-1,:] - G[:,:,1:,:]))) 

                # calculate learned regularization loss
#                layers = net.parameters()
#                layer_means = torch.cat([layer.mean().view(1) for layer in layers])
#                lr_loss = torch.matmul(layer_means-mu,torch.matmul(sig_inv,layer_means-mu))
                
#                total_loss = y_loss + lrc*lr_loss + tvc*tv_loss # total loss for iteration i
                total_loss = y_loss + tvc*tv_loss               
                # stopping condition to account for optimizer convergence
                if i >= args.NUM_ITER - EXIT_WINDOW: 
                    recons_iter.append(G.data.cpu().numpy())
                    loss_iter.append(total_loss.data.cpu().numpy())
                    if i == args.NUM_ITER - 1:
                        idx_iter = np.argmin(loss_iter)

                total_loss.backward() # backprop
                optim.step()
        
            recons_re[j] = recons_iter[idx_iter]       
            loss_re[j] = y_loss.data.cpu().numpy()

        idx_re = np.argmin(loss_re,axis=0)
        x_hat = recons_re[idx_re]
        print(loss_re)         
        return x_hat

    return estimator
