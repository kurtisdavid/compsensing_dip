import torch
import torch.nn as nn
import math
import numbers
from torch.nn import functional as F
import numpy

# adjusted from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
class PyrDown(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels = 3, kernel_size = 5, dim = 2, n_levels = 3, device="cuda:0"):
        super(PyrDown, self).__init__()
        
        # manually set weights...
        kernel = math.sqrt(2) * numpy.array( [ [1,4,6,4,1], [4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4], [1,4,6,4,1] ])
        kernel = torch.from_numpy(kernel) 
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
    
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        kernel = kernel.type(torch.FloatTensor).to(device)
        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        self.n_levels = n_levels
        self.device = device

    def _gather_odds(self, input, dim, other_dim):
        if dim < other_dim:
            odds = torch.from_numpy(numpy.array([i for i in range(input.shape[other_dim]) if i % 2 == 1])).repeat(input.shape[0],input.shape[1],input.shape[dim],1)
        else:
            odds = torch.from_numpy(numpy.array([[i for _ in range(input.shape[dim])] for i in range(input.shape[other_dim]) if i % 2 == 1])).repeat(input.shape[0],input.shape[1],1,1)
        return torch.gather(input, other_dim, odds.to(self.device))

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        for i in range(self.n_levels):
            input = self.conv(input, weight=self.weight, groups=self.groups, padding=2)
            # downsample by removing evens in each dimension
            input = self._gather_odds(input, 2, 3)
            input = self._gather_odds(input, 3, 2)
        return input


