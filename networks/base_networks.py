import torch
import numpy    as np
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn import ReLU, LeakyReLU
from   torch.distributions.normal import Normal
from   . import layers
from   . import hyper_net as hn


from networks.TSM.TransMorph            import CONFIGS as cfg_tsm,   TransMorph as tsm
from networks.TSM_A.TransMorph_affine   import CONFIGS as cfg_tsm_a, SwinAffine as tsm_a
from networks.CLMorph.CLM_affine        import CLMorphAffineStem
from networks.CLMorph.CLM_elastic       import CLMorph

BASE_NETWORK = ['VTN', 'VXM', 'TSM', 'CLM']

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


def convolveReLU(in_channels, out_channels, kernel_size, stride, dim=2):
    return nn.Sequential(ReLU, convolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def convolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=2, leakyr_slope=0.1):
    return nn.Sequential(LeakyReLU(leakyr_slope), convolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def hyp_convolve(in_channels, out_channels, kernel_size, stride, dim=2, hyp_unit=128):
    return hn.HyperConv(rank=dim, hyp_units=hyp_unit, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1)


def hyp_convolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=2, leakyr_slope=0.1, hyp_unit=128):
    return nn.Sequential(LeakyReLU(leakyr_slope), hyp_convolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def upconvolve(in_channels, out_channels, kernel_size, stride, dim=2):
    return trans_conv(dim=dim)(in_channels, out_channels, kernel_size, stride, padding=1)


def upconvolveReLU(in_channels, out_channels, kernel_size, stride, dim=2):
    return nn.Sequential(ReLU, upconvolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def upconvolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=2):
    return nn.Sequential(LeakyReLU(0.1), upconvolve(in_channels, out_channels, kernel_size, stride, dim=dim))


# This function requires improvement to handle different sizes in different dimensions!
def pad_tensors(tensors, dim_to_pad):
    # Determine the max size along the given dimension
    max_size = max(t.size(dim_to_pad) for t in tensors)
    
    # Pad the tensors to match the max size along the specified dimension
    padded_tensors = []
    for t in tensors:
        padding_size = list(t.shape)
        padding_size[dim_to_pad] = max_size - t.size(dim_to_pad)
        if padding_size[dim_to_pad] > 0:
            padding = (0, padding_size[dim_to_pad])  # Padding tuple for the given dimension
            for _ in range(len(t.shape) - dim_to_pad - 1):
                padding = (0, 0) + padding  # Pad other dimensions with 0
            t = torch.nn.functional.pad(t, padding)
        padded_tensors.append(t)
    return padded_tensors


def pad_or_truncate(tensor, target_size, dim):
    """
    Pad or truncate a tensor along the specified dimension to the target size.
    """
    current_size = tensor.size(dim)
    if current_size > target_size:
        # Truncate the tensor
        slices = [slice(None)] * tensor.ndimension()
        slices[dim] = slice(0, target_size)
        return tensor[tuple(slices)]
    elif current_size < target_size:
        # Pad the tensor
        pad_size = [(0, 0)] * tensor.ndimension()
        pad_size[dim] = (0, target_size - current_size)
        pad_size = [item for sublist in pad_size for item in sublist]  # Flatten list
        return torch.nn.functional.pad(tensor, pad_size)
    else:
        return tensor

def get_same_dim_tensors(tensors, target_size, dim):
    """
    Process a list of tensors to ensure they all have the target size in the specified dimension.
    """
    return [pad_or_truncate(t, target_size, dim) for t in tensors]



def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 16, 16]
    ]
    return nb_features



class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, in_channels=2, hyper_net=False):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """
        self.is_hyper = hyper_net

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats       = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        if self.is_hyper:
            conv_block = hyp_convolveLeakyReLU
        else:
            conv_block = convolveLeakyReLU
        prev_nf      = in_channels
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(conv_block(prev_nf, nf, dim=ndims, kernel_size=3, stride=2, leakyr_slope=0.1))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm  = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(conv_block(channels, nf, dim=ndims, kernel_size=3, stride=1, leakyr_slope=0.1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf    += in_channels
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(conv_block(prev_nf, nf, dim=ndims, kernel_size=3, stride=1, leakyr_slope=0.1))
            prev_nf = nf

    def forward(self, x, hyp_tensor=None):
        if self.is_hyper:
            for layer in self.downarm+self.uparm+self.extras:
                layer[1].build_hyp(hyp_tensor)

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x
    
    
    
class VXM(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    # @store_config_args
    def __init__(self,
        im_size,
        flow_multiplier = 1,
        nb_unet_features= None,
        nb_unet_levels  = None,
        unet_feat_mult  = 1,
        int_steps       = 7,
        int_downsize    = 2,
        in_channels     = 2,
        bidir           = False,
        use_probs       = False,
        hyper_net       = False,):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(im_size)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            im_size,
            nb_features = nb_unet_features,
            nb_levels   = nb_unet_levels,
            feat_mult   = unet_feat_mult,
            in_channels = in_channels,
            hyper_net   = hyper_net
        )

        # configure unet to flow field layer
        Conv      = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight      = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias        = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.flow.initialized = True
        setattr(self.flow, 'initialized', True)

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize        = int_steps > 0 and int_downsize > 1
        self.resize   = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape     = [int(dim / int_downsize) for dim in im_size]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.flow_multiplier = flow_multiplier

    def forward(self, source, target, return_preint=False, return_neg=False, hyp_tensor=None):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
        '''
        bidir = self.bidir or return_neg

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x, hyp_tensor=hyp_tensor)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if bidir else None

        returns = [pos_flow]
        if bidir: returns.append(neg_flow)
        if return_preint: returns.append(preint_flow)
        returns = [r*self.flow_multiplier for r in returns]
        return returns if len(returns)>1 else returns[0]


class VTN(nn.Module):
    """
    A PyTorch implementation of the VTN network. The network is a UNet.

    Args:
        im_size (tuple): The size of the input image.
        flow_multiplier (float): The flow multiplier.
        channels (int): The number of channels in the first convolution. The following convolution channels will be [2x, 4x, 8x, 16x] of this value.
        in_channels (int): The number of input channels.
    """
    def __init__(self, im_size=(128, 128, 128), flow_multiplier=1., channels=16, in_channels=2, hyper_net=None):
        super(VTN, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels        = channels
        self.dim = dim       = len(im_size)
        
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

        self.pred6      = convolve(32 * channels, dim, 3, 1, dim=dim)
        self.upsamp6to5 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv5    = upconvolveLeakyReLU(32 * channels, 16 * channels, 4, 2, dim=dim)

        self.pred5      = convolve(32 * channels + dim, dim, 3, 1, dim=dim)  # 514 = 32 * channels + 1 + 1
        self.upsamp5to4 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv4    = upconvolveLeakyReLU(32 * channels + dim, 8 * channels, 4, 2, dim=dim)

        self.pred4      = convolve(16 * channels + dim, dim, 3, 1, dim=dim)  # 258 = 64 * channels + 1 + 1
        self.upsamp4to3 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv3    = upconvolveLeakyReLU(16 * channels + dim,  4 * channels, 4, 2, dim=dim)

        self.pred3      = convolve(8 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp3to2 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv2    = upconvolveLeakyReLU(8 * channels + dim, 2 * channels, 4, 2, dim=dim)

        self.pred2      = convolve(4 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp2to1 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv1    = upconvolveLeakyReLU(4 * channels + dim, channels, 4, 2, dim=dim)

        self.pred0      = upconvolve(2 * channels + dim, dim, 4, 2, dim=dim)
        
    
    def forward(self, fixed, moving, return_neg = False, hyp_tensor=None):
        
        concat_image = torch.cat((fixed, moving), dim=1)    # 2 x 512 x 512         #   2 x 192 x 192 x 208
        x1   = self.conv1(concat_image)                     # 16 x 256 x 256        #  16 x  96 x  96 x 104
        x2   = self.conv2(x1)                               # 32 x 128 x 128        #  32 x  48 x  48 x  52
        x3   = self.conv3(x2)                               # 64 x 64 x 64          #  64 x  24 x  24 x  26 
        x3_1 = self.conv3_1(x3)                             # 64 x 64 x 64          #  64 x  24 x  24 x  26 
        x4   = self.conv4(x3_1)                             # 128 x 32 x 32         # 128 x  12 x  12 x  13
        x4_1 = self.conv4_1(x4)                             # 128 x 32 x 32         # 128 x  12 x  12 x  13
        x5   = self.conv5(x4_1)                             # 256 x 16 x 16         # 256 x   6 x   6 x   7
        x5_1 = self.conv5_1(x5)                             # 256 x 16 x 16         # 256 x   6 x   6 x   7
        x6   = self.conv6(x5_1)                             # 512 x 8 x 8           # 512 x   3 x   3 x   4
        x6_1 = self.conv6_1(x6)                             # 512 x 8 x 8           # 512 x   3 x   3 x   4

        pred6      = self.pred6(x6_1)                               # 2 x 8 x 8         #   3 x 3 x 3 x 4
        upsamp6to5 = self.upsamp6to5(pred6)                         # 2 x 16 x 16       #   3 x 6 x 6 x 8
        deconv5    = self.deconv5(x6_1)                             # 256 x 16 x 16     # 256 x 6 x 6 x 8
        # Funtion to get the same size in dimension 4 
        # in order to be able to concat the tensors. 
        #tensors    = get_same_dim_tensors([x5_1, deconv5, upsamp6to5], x5_1.size(-1), -1) # Uncomment this for lung
        tensors    = get_same_dim_tensors([x5_1, deconv5, upsamp6to5], x5_1.size(-2), -2)  # Uncomment this for abdomen... improve this part!
        concat5    = torch.cat(tensors, dim=1)                      # 514 x 16 x 16     # 515 x 6 x 6 x 7

        pred5      = self.pred5(concat5)                            # 2 x 16 x 16       #   
        upsamp5to4 = self.upsamp5to4(pred5)                         # 2 x 32 x 32       #   
        deconv4    = self.deconv4(concat5)                          # 2 x 32 x 32       #  
        #tensors    = get_same_dim_tensors([x4_1, deconv4, upsamp5to4], x4_1.size(-1), -1)
        tensors    = get_same_dim_tensors([x4_1, deconv4, upsamp5to4], x4_1.size(-2), -2)
        concat4    = torch.cat(tensors, dim=1)                      # 258 x 32 x 32     # 

        pred4      = self.pred4(concat4)                            # 2 x 32 x 32
        upsamp4to3 = self.upsamp4to3(pred4)                         # 2 x 64 x 64
        deconv3    = self.deconv3(concat4)                          # 64 x 64 x 64
        concat3    = torch.cat([x3_1, deconv3, upsamp4to3], dim=1)  # 130 x 64 x 64

        pred3      = self.pred3(concat3)                            # 2 x 63 x 64
        upsamp3to2 = self.upsamp3to2(pred3)                         # 2 x 128 x 128
        deconv2    = self.deconv2(concat3)                          # 32 x 128 x 128
        concat2    = torch.cat([x2, deconv2, upsamp3to2], dim=1)    # 66 x 128 x 128
        
        
        pred2      = self.pred2(concat2)                            # 2 x 128 x 128
        upsamp2to1 = self.upsamp2to1(pred2)                         # 2 x 256 x 256
        deconv1    = self.deconv1(concat2)                          # 16 x 256 x 256
        concat1    = torch.cat([x1, deconv1, upsamp2to1], dim=1)    # 34 x 256 x 256
        #import pdb; pdb.set_trace()
        pred0      = self.pred0(concat1)                            # 2 x 512 x 512
        return pred0 * 20 * self.flow_multiplier                    # why the 20?
        

class VTNAffineStem(nn.Module):
    """
    VTN affine stem. This is the first part of the VTN network. A multi-layer convolutional network that calculates the affine transformation parameters.

    Args:
        dim (int): Dimension of the input image.
        channels (int): Number of channels in the first convolution.
        flow_multiplier (float): Multiplier for the flow output.
        im_size (int): Size of the input image.
        in_channels (int): Number of channels in the input image.
    """
    def __init__(self, dim=1, channels=16, flow_multiplier=1., im_size=512, in_channels=2):
        super(VTNAffineStem, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels        = channels
        self.dim             = dim

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1   = convolveLeakyReLU(in_channels, channels, 3, 2, dim=self.dim)
        self.conv2   = convolveLeakyReLU(channels, 2 * channels, 3, 2, dim=dim)
        self.conv3   = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1, dim=dim)
        self.conv4   = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1, dim=dim)
        self.conv5   = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2, dim=dim)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1, dim=dim)
        self.conv6   = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2, dim=dim)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1, dim=dim)

        # I'm assuming that the image's shape is like (im_size, im_size, im_size)
        self.last_conv_size = im_size // (self.channels * 4)
        self.fc_loc         = nn.Sequential(
            nn.Linear(18432, 2048),#(512 * self.last_conv_size**dim, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 6*(dim - 1))
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        """
        Identity Matrix
            | 1 0 0 0 |
        I = | 0 1 0 0 |
            | 0 0 1 0 |
        """
        if dim == 3:
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.create_flow = self.cr_flow


    def cr_flow(self, theta, size):
        shape = size[2:]
        flow  = F.affine_grid(theta-torch.eye(len(shape), len(shape)+1, device=theta.device), size, align_corners=False)
        if len(shape) == 2:
            flow = flow[..., [1, 0]]
            flow = flow.permute(0, 3, 1, 2)
        elif len(shape) == 3:
            flow = flow[..., [2, 1, 0]]
            flow = flow.permute(0, 4, 1, 2, 3)
        flow = flow*flow.new_tensor(shape).view(-1, *[1 for _ in shape])/2
        return flow
    
    
    def wr_flow(self, theta, size):
        flow = F.affine_grid(theta, size, align_corners=False)  # batch x 512 x 512 x 2
        if self.dim == 2:
            flow = flow.permute(0, 3, 1, 2)  # batch x 2 x 512 x 512
        else:
            flow = flow.permute(0, 4, 1, 2, 3)
        return flow
    
    
    def rev_affine(self, theta, dim=2):
        b = theta[:, :, dim:]
        inv_w = torch.inverse(theta[:, :dim, :dim])
        neg_affine = torch.cat([inv_w, -inv_w@b], dim=-1)
        return neg_affine
    
    
    def neg_flow(self, theta, size):
        neg_affine = self.rev_affine(theta, dim=self.dim)
        return self.create_flow(neg_affine, size)


    def forward(self, fixed, moving):
        """
        Calculate the affine transformation parameters

        Returns:
            flow: the flow field
            theta: dict, with the affine transformation parameters
        """
        concat_image = torch.cat((fixed, moving), dim=1)  # 2 x 512 x 512  -----L #2 x 192 x 192 x 208  -----A #2 x 192 x 160 x 256
        x1   = self.conv1(concat_image)  # 16 x 256 x 256  ----- #16 x 96 x 96 x 104   -----A #16 x 96 x 80 x 128
        x2   = self.conv2(x1)  # 32 x 128 x 128  ----- #32 x 48 x 48 x 52 -----A #32 x 48 x 40 x 64
        x3   = self.conv3(x2)  # 1 x 64 x 64 x 64  ----- #64 x 24 x 24 x 26 -----A #64 x 24 x 20 x 32
        x3_1 = self.conv3_1(x3)  # 64 x 64 x 64  ----- #64 x 24 x 24 x 26 -----A #64 x 24 x 20 x 32
        x4   = self.conv4(x3_1)  # 128 x 32 x 32  ----- #128 x 12 x 12 x 13 -----A #128 x 12 x 10 x 16
        x4_1 = self.conv4_1(x4)  # 128 x 32 x 32  ----- #128 x 12 x 12 x 13 -----A #128 x 12 x 10 x 16
        x5   = self.conv5(x4_1)  # 256 x 16 x 16  ----- #256 x 6 x 6 x 7 -----A #256 x 6 x 5 x 8
        x5_1 = self.conv5_1(x5)  # 256 x 16 x 16  ----- #256 x 6 x 6 x 7 -----A #256 x 6 x 5 x 8
        x6   = self.conv6(x5_1)  # 512 x 8 x 8  ----- #512 x 3 x 3 x 4 -----A #512 x 3 x 3 x 4
        x6_1 = self.conv6_1(x6)  # 512 x 8 x 8  ----- #512 x 3 x 3 x 4  -----A #512 x 3 x 3 x 4

        # Affine transformation
        xs = x6_1.view(-1, 18432)#512 * self.last_conv_size ** self.dim)
        if self.dim == 3:
            theta = self.fc_loc(xs).view(-1, 3, 4)
        else:
            theta = self.fc_loc(xs).view(-1, 2, 3)
        flow = self.create_flow(theta, moving.size())
        # theta: the affine param
        return flow, {'theta': theta}
    


class TSMAffineStem(nn.Module):
    """
    TSM affine stem. This is the first part of the TSM network.
    Credit for https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration.git.

    Args:
        dim (int): Dimension of the input image.
        channels (int): Number of channels in the first convolution.
        flow_multiplier (float): Multiplier for the flow output.
        im_size (int): Size of the input image.
        in_channels (int): Number of channels in the input image.
    """
    def __init__(self, dim=1, channels=16, flow_multiplier=1., im_size=512, in_channels=2):
        super(TSMAffineStem, self).__init__()
        config = cfg_tsm_a['TransMorph-Affine']
        # AffInfer = .ApplyAffine()
        '''
        config = ml_collections.ConfigDict()
        config.if_transskip = True
        config.if_convskip = True
        config.patch_size = 4
        config.in_chans = 2
        config.embed_dim = 48
        config.depths = (2, 2, 4, 2)
        config.num_heads = (4, 4, 8, 8)
        config.window_size = (5, 6, 7)
        config.mlp_ratio = 4
        config.pat_merg_rf = 4
        config.qkv_bias = False
        config.drop_rate = 0
        config.drop_path_rate = 0.3
        config.ape = False
        config.spe = False
        config.rpe = True
        config.patch_norm = True
        config.use_checkpoint = False
        config.out_indices = (0, 1, 2, 3)
        config.reg_head_chan = 16
        config.img_size = (160, 192, 224)'''
        config.in_chans = in_channels
        config.img_size = im_size if isinstance(im_size, (list, tuple)) else ((im_size, im_size, im_size) if dim == 3 else (im_size, im_size))
        self.model = tsm_a(config)
        self.flow_multiplier = flow_multiplier
        vectors = [torch.arange(0, s) for s in config.img_size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        self.register_buffer('grid', grid, persistent=False)
        self.cfg = config

    def cr_flow(self, theta, size):
        shape = size[2:]
        flow = F.affine_grid(theta-torch.eye(len(shape), len(shape)+1, device=theta.device), size, align_corners=False)
        if len(shape) == 2:
            flow = flow[..., [1, 0]]
            flow = flow.permute(0, 3, 1, 2)
        elif len(shape) == 3:
            flow = flow[..., [2, 1, 0]]
            flow = flow.permute(0, 4, 1, 2, 3)
        flow = flow*flow.new_tensor(shape).view(-1, *[1 for _ in shape])/2
        return flow

    def forward(self, fixed, moving):
        """
        Calculate the affine transformation parameters

        Returns:
            flow: the flow field
            theta: dict, with the affine transformation parameters
        """
        concat_image = torch.cat((fixed, moving), dim=1)  # 2 x 512 x 512       # 2 x 192 x 192 x 208 
        mat   = self.model(concat_image)
        theta = mat
        flow  = self.cr_flow(theta, moving.size())
        # theta: the affine param
        return flow, {'theta': theta}

class TSM(nn.Module):
    '''
    TransfMorph model. Credit for https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration.git.
    '''
    def __init__(self, im_size=(128,128,128), flow_multiplier=1., in_channels=2):
        super(TSM, self).__init__()
        config = cfg_tsm['TransMorph']
        config['img_size'] = im_size
        config['in_chans'] = in_channels
        self.model = tsm(config)
        self.flow_multiplier = flow_multiplier

    def forward(self, fixed, moving, return_neg=False):
        x_in = torch.cat((fixed, moving), dim=1)
        flow = self.model(x_in)
        flow = self.flow_multiplier*flow
        return flow * self.flow_multiplier
    

class CLMAffineStem(nn.Module):
    def __init__(self, dim=1, channels=16, flow_multiplier=1., im_size=512, in_channels=2):
        super(CLMAffineStem, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels        = channels
        self.dim             = dim
        self.model           = CLMorphAffineStem(dim=self.dim)
        
    def forward(self, fixed, moving):
        """
        Calculate the affine transformation parameters

        Returns:
            flow: the flow field
            theta: dict, with the affine transformation parameters
        """
        
        output = self.model(fixed, moving)
        theta  = output[1]['theta']
        flow   = output[0]
        # theta: the affine param
        return flow, {'theta': theta}



class CLM(nn.Module):
    def __init__(self, im_size=(128, 128, 128), flow_multiplier=1., channels=16, in_channels=1, hyper_net=None):
        super(CLM, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels        = channels
        self.dim = dim       = len(im_size)
        self.model           = CLMorph(flow_multiplier=self.flow_multiplier)
                
    def forward(self, fixed, moving, theta, return_neg = False, hyp_tensor=None):
        """
        Calculate the affine transformation parameters

        Returns:
            flow: the flow field
            theta: dict, with the affine transformation parameters
        """
        
        flow = self.model(fixed, moving, theta)
        return flow
    
    
    
if __name__ == "__main__":
    model = CLM()
    x   = torch.randn(1, 1, 192, 192, 208)
    t   = torch.randn(1, 3, 4)
    out = model(x, x, t)
    print('Output shape: ', out[0].shape)
    print('Output shape: ', out[1]['theta'].shape)