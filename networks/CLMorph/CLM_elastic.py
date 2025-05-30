import torch
import torch.nn as nn
from   torch.nn import LeakyReLU
import torch.nn.functional as nnf

def conv(dim=2):
    if dim == 2:
        return nn.Conv2d
    return nn.Conv3d

def trans_conv(dim=2):
    if dim == 2:
        return nn.ConvTranspose2d
    return nn.ConvTranspose3d

def convolve(in_channels, out_channels, kernel_size, stride, dim=2):
    return conv(dim=dim)(in_channels, out_channels, kernel_size, stride=stride, padding=1)

def convolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=2, leakyr_slope=0.1):
    return nn.Sequential(LeakyReLU(leakyr_slope), convolve(in_channels, out_channels, kernel_size, stride, dim=dim))

def upconvolve(in_channels, out_channels, kernel_size, stride, dim=2):
    return trans_conv(dim=dim)(in_channels, out_channels, kernel_size, stride, padding=1)

def upconvolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=2):
    return nn.Sequential(LeakyReLU(0.1), upconvolve(in_channels, out_channels, kernel_size, stride, dim=dim))

def pad_or_truncate(tensor, target_size, dim):
    """
    Pad or truncate a tensor along the specified dimension to the target size.
    """
    current_size = tensor.size(dim)
    #print(f"Current size: {current_size}, Target size: {target_size}, Dimension: {dim}")
    if current_size > target_size:
        # Truncate the tensor
        slices           = [slice(None)] * tensor.ndimension()
        slices[dim]      = slice(0, target_size)
        truncated_tensor = tensor[tuple(slices)]
        #print(f"Truncated tensor shape: {truncated_tensor.shape}")
        return truncated_tensor
    elif current_size < target_size:
        # Pad the tensor
        pad_size                 = [0] * (2 * tensor.ndimension())
        pad_size[-(dim * 2 + 1)] = target_size - current_size
        padded_tensor            = torch.nn.functional.pad(tensor, pad_size)
        #print(f"Padded tensor shape: {padded_tensor.shape}")
        return padded_tensor
    else:
        return tensor

def get_same_dim_tensors(tensors, target_size, dim):
    """
    Process a list of tensors to ensure they all have the target size in the specified dimension.
    """
    return [pad_or_truncate(t, target_size, dim) for t in tensors]


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer. It takes a source image and a flow field and returns the warped image. The flow vectors start from the source image points and point to the target grid points. 
    """
    def __init__(self):
        """
        Args:
            size (tuple): The size of the input image.
        """
        super().__init__()

        

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        #self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow, mode='bilinear'):
        
        # create sampling grid
        size    = src.shape[2:]
        device  = src.device
        vectors = [torch.arange(0, s, device=device) for s in size]
        grids   = torch.meshgrid(vectors, indexing='ij')
        grid    = torch.stack(grids)
        grid    = torch.unsqueeze(grid, 0)
        grid    = grid.type(torch.FloatTensor).to(device)
        #self.register_buffer('grid', grid, persistent=False)
        
        # new locations
        new_locs = grid + flow
        shape    = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=False, mode=mode)



class CLMorph(nn.Module):
    """
    A PyTorch implementation of the CLMorph network. The network is a UNet.

    Args:
        im_size (tuple): The size of the input image.
        flow_multiplier (float): The flow multiplier.
        channels (int): The number of channels in the first convolution. The following convolution channels will be [2x, 4x, 8x, 16x] of this value.
        in_channels (int): The number of input channels.
    """
    def __init__(self, im_size=(128, 128, 128), flow_multiplier=1., channels=16, in_channels=1, hyper_net=None):
        super(CLMorph, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels        = channels
        self.dim = dim       = len(im_size)
        device               = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1   = convolveLeakyReLU(  in_channels,      channels, 3, 2, dim=dim)
        self.conv2   = convolveLeakyReLU(     channels, 2  * channels, 3, 2, dim=dim)
        self.conv3   = convolveLeakyReLU(2  * channels, 4  * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4  * channels, 4  * channels, 3, 1, dim=dim)
        self.conv4   = convolveLeakyReLU(4  * channels, 8  * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8  * channels, 8  * channels, 3, 1, dim=dim)
        self.conv5   = convolveLeakyReLU(8  * channels, 16 * channels, 3, 2, dim=dim)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1, dim=dim)
        self.conv6   = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2, dim=dim)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1, dim=dim)

        #self.conv_ls = convolveLeakyReLU(  in_channels*2, 32*channels, 3, 2, dim=dim)
        
        self.pred6      = convolve(2 * 32 * channels, dim, 3, 1, dim=dim)
        self.upsamp6to5 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv5    = upconvolveLeakyReLU(2 * 32 * channels, 16 * channels, 4, 2, dim=dim)

        self.pred5      = convolve(3 * 16 * channels + dim, dim, 3, 1, dim=dim)  # 514 = 32 * channels + 1 + 1
        self.upsamp5to4 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv4    = upconvolveLeakyReLU(3 * 16 * channels + dim, 8 * channels, 4, 2, dim=dim)

        self.pred4      = convolve(3 * 8 * channels + dim, dim, 3, 1, dim=dim)  # 258 = 64 * channels + 1 + 1
        self.upsamp4to3 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv3    = upconvolveLeakyReLU(3 * 8 * channels + dim,  4 * channels, 4, 2, dim=dim)

        self.pred3      = convolve(3 * 4 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp3to2 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv2    = upconvolveLeakyReLU(3 * 4 * channels + dim, 2 * channels, 4, 2, dim=dim)

        self.pred2      = convolve(3 * 2 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp2to1 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv1    = upconvolveLeakyReLU(3 * 2 * channels + dim, channels, 4, 2, dim=dim)

        self.pred0      = upconvolve(3 * channels + dim, dim, 4, 2, dim=dim)
        
        # Flow
        self.create_flow    = self.cr_flow
        self.reconstruction = SpatialTransformer().to(device)
        
        
    def cr_flow(self, theta, size):
        shape = size[2:]
        flow  = nnf.affine_grid(theta-torch.eye(len(shape), len(shape)+1, device=theta.device), size, align_corners=False)
        if len(shape) == 2:
            flow = flow[..., [1, 0]]
            flow = flow.permute(0, 3, 1, 2)
        elif len(shape) == 3:
            flow = flow[..., [2, 1, 0]]
            flow = flow.permute(0, 4, 1, 2, 3)
        flow = flow*flow.new_tensor(shape).view(-1, *[1 for _ in shape])/2
        return flow
    
    def forward(self, fixed, moving, theta, return_neg = False, hyp_tensor=None):
        
        #concat_image = torch.cat((fixed, moving), dim=1)    # 2 x 512 x 512         #   2 x 192 x 192 x 208
        # Fixed image features
        
        f1   = self.conv1(fixed)                            # 16 x 256 x 256        #  16 x  96 x  96 x 104
        f2   = self.conv2(f1)                               # 32 x 128 x 128        #  32 x  48 x  48 x  52
        f3   = self.conv3(f2)                               # 64 x 64 x 64          #  64 x  24 x  24 x  26 
        f3_1 = self.conv3_1(f3)                             # 64 x 64 x 64          #  64 x  24 x  24 x  26 
        f4   = self.conv4(f3_1)                             # 128 x 32 x 32         # 128 x  12 x  12 x  13
        f4_1 = self.conv4_1(f4)                             # 128 x 32 x 32         # 128 x  12 x  12 x  13
        f5   = self.conv5(f4_1)                             # 256 x 16 x 16         # 256 x   6 x   6 x   7
        f5_1 = self.conv5_1(f5)                             # 256 x 16 x 16         # 256 x   6 x   6 x   7
        f6   = self.conv6(f5_1)                             # 512 x 8 x 8           # 512 x   3 x   3 x   4
        f6_1 = self.conv6_1(f6)                             # 512 x 8 x 8           # 512 x   3 x   3 x   4
        
        
        
        # Moving image features
        m1   = self.conv1(moving)                            # 16 x 256 x 256        #  16 x  96 x  96 x 104
        m2   = self.conv2(m1)                               # 32 x 128 x 128        #  32 x  48 x  48 x  52
        m3   = self.conv3(m2)                               # 64 x 64 x 64          #  64 x  24 x  24 x  26 
        m3_1 = self.conv3_1(m3)                             # 64 x 64 x 64          #  64 x  24 x  24 x  26 
        m4   = self.conv4(m3_1)                             # 128 x 32 x 32         # 128 x  12 x  12 x  13
        m4_1 = self.conv4_1(m4)                             # 128 x 32 x 32         # 128 x  12 x  12 x  13
        m5   = self.conv5(m4_1)                             # 256 x 16 x 16         # 256 x   6 x   6 x   7
        m5_1 = self.conv5_1(m5)                             # 256 x 16 x 16         # 256 x   6 x   6 x   7
        m6   = self.conv6(m5_1)                             # 512 x 8 x 8           # 512 x   3 x   3 x   4
        m6_1 = self.conv6_1(m6)                             # 512 x 8 x 8           # 512 x   3 x   3 x   4
        
               
        # Concatenation of fixed and moving 
        xy   = torch.cat((f6_1, m6_1), dim=1)                                       # 1024 x   3 x   3 x   4
        pred6      = self.pred6(xy)                               # 2 x 8 x 8         #   3 x 3 x 3 x 4
        upsamp6to5 = self.upsamp6to5(pred6)                         # 2 x 16 x 16     #   3 x 6 x 6 x 8
        deconv5    = self.deconv5(xy)                             # 256 x 16 x 16     # 256 x 6 x 6 x 8
        
        
        # Funtion to get the same size in dimension 4 
        # in order to be able to concat the tensors. 
        #tensors    = get_same_dim_tensors([f5_1, m5_1, deconv5, upsamp6to5], m5_1.size(-1), -1)  # Lung
        tensors    = get_same_dim_tensors([f5_1, m5_1, deconv5, upsamp6to5], m5_1.size(-2), -2)   # Abdomen
        concat5    = torch.cat(tensors, dim=1)                      # 514 x 16 x 16     # 771 x 6 x 6 x 7

        
        pred5      = self.pred5(concat5)                            # 2 x 16 x 16       #  3 x 6 x 6 x 7
        upsamp5to4 = self.upsamp5to4(pred5)                         # 2 x 32 x 32       #  3 x 12 x 12 x 14
        deconv4    = self.deconv4(concat5)                          # 2 x 32 x 32       #  128 x 12 x 12 x 14
        #tensors    = get_same_dim_tensors([f4_1, m4_1, deconv4, upsamp5to4], m4_1.size(-1), -1) # Lung
        tensors    = get_same_dim_tensors([f4_1, m4_1, deconv4, upsamp5to4], m4_1.size(-2), -2)  # Abdomen
        concat4    = torch.cat(tensors, dim=1)                      # 258 x 32 x 32     # 387 x 12 x 12 x 13

        pred4      = self.pred4(concat4)                            # 2 x 32 x 32       # 3 x 12 x 12 x 13
        upsamp4to3 = self.upsamp4to3(pred4)                         # 2 x 64 x 64       # 3 x 24 x 24 x 26
        deconv3    = self.deconv3(concat4)                          # 64 x 64 x 64      # 64 x 24 x 24 x 26
        concat3    = torch.cat([f3_1, m3_1, deconv3, upsamp4to3], dim=1)  # 130 x 64 x 64 # 195 x 24 x 24 x 26
        
        pred3      = self.pred3(concat3)                            # 2 x 63 x 64           # 3 x 24 x 24 x 26
        upsamp3to2 = self.upsamp3to2(pred3)                         # 2 x 128 x 128         # 3 x 48 x 48 x 52
        deconv2    = self.deconv2(concat3)                          # 32 x 128 x 128        # 32 x 48 x 48 x 52
        concat2    = torch.cat([f2, m2, deconv2, upsamp3to2], dim=1)    # 66 x 128 x 128    # 99 x 48 x 48 x 52
        
        pred2      = self.pred2(concat2)                            # 2 x 128 x 128         # 3 x 48 x 48 x 52
        upsamp2to1 = self.upsamp2to1(pred2)                         # 2 x 256 x 256         # 3 x 96 x 96 x 104
        deconv1    = self.deconv1(concat2)                          # 16 x 256 x 256        # 16 x 96 x 96 x 104
        concat1    = torch.cat([f1, m1, deconv1, upsamp2to1], dim=1)    # 34 x 256 x 256    # 51 x 96 x 96 x 104

        pred0      = self.pred0(concat1)                            # 2 x 512 x 512
        #import pdb; pdb.set_trace()
        return pred0 * self.flow_multiplier                    
    

if __name__ == "__main__":
    model = CLMorph(im_size=(192, 160, 256))
    x   = torch.randn(1, 1, 192, 160, 256)
    y   = torch.randn(1, 1, 192, 160, 256)
    t   = torch.randn(1, 3, 4)
    out = model(x, x, t)
    print('Output shape: ', out.shape)