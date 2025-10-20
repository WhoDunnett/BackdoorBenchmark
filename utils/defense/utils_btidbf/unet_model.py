"""
Class definitions for a standard U-Net Up-and Down-sampling blocks
http://arxiv.org/abs/1505.0.397
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """
    Instances the Encoder block that forms a part of a U-Net
    Parameters:
        in_channels (int): Depth (or number of channels) of the tensor that the block acts on
        filter_num (int) : Number of filters used in the convolution ops inside the block,
                             depth of the output of the enc block
        dropout(bool) : Flag to decide whether a dropout layer should be applied
        dropout_rate (float) : Probability of dropping a convolution output feature channel
    """
    def __init__(self, filter_num=64, in_channels=1, dropout=False, dropout_rate=0.3):

        super(EncoderBlock,self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        self.conv1 = nn.Conv2d(in_channels=self.in_channels,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv2d(in_channels=self.filter_num,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.bn_op_1 = nn.InstanceNorm2d(num_features=self.filter_num, affine=True)
        self.bn_op_2 = nn.InstanceNorm2d(num_features=self.filter_num, affine=True)

        # Use Dropout ops as nn.Module instead of nn.functional definition
        # So using .train() and .eval() flags, can modify their behavior for MC-Dropout
        if dropout is True:
            self.dropout_1 = nn.Dropout(p=dropout_rate)
            self.dropout_2 = nn.Dropout(p=dropout_rate)

    def apply_manual_dropout_mask(self, x, seed):
        # Mask size : [Batch_size, Channels, Height, Width]
        dropout_mask = torch.bernoulli(input=torch.empty(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).fill_(self.dropout_rate),
                                       generator=torch.Generator().manual_seed(seed))

        x = x*dropout_mask.to(x.device)

        return x

    def forward(self, x, seeds=None):

        if seeds is not None:
            assert(seeds.shape[0] == 2)

        x = self.conv1(x)
        x = self.bn_op_1(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            if seeds is None:
                x = self.dropout_1(x)
            else:
                x = self.apply_manual_dropout_mask(x, seeds[0].item())

        x = self.conv2(x)
        x = self.bn_op_2(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            if seeds is None:
                x = self.dropout_2(x)
            else:
                x = self.apply_manual_dropout_mask(x, seeds[1].item())

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block used in the U-Net
    Parameters:
        in_channels (int) : Number of channels of the incoming tensor for the upsampling op
        concat_layer_depth (int) : Number of channels to be concatenated via skip connections
        filter_num (int) : Number of filters used in convolution, the depth of the output of the dec block
        interpolate (bool) : Decides if upsampling needs to performed via interpolation or transposed convolution
        dropout(bool) : Flag to decide whether a dropout layer should be applied
        dropout_rate (float) : Probability of dropping a convolution output feature channel
    """
    def __init__(self, in_channels, concat_layer_depth, filter_num, interpolate=False, dropout=False, dropout_rate=0.3):

        # Up-sampling (interpolation or transposed conv) --> EncoderBlock
        super(DecoderBlock, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.concat_layer_depth = int(concat_layer_depth)
        self.interpolate = interpolate
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        # Upsample by interpolation followed by a 3x3 convolution to obtain desired depth
        self.up_sample_interpolate = nn.Sequential(nn.Upsample(scale_factor=2,
                                                               mode='bilinear',
                                                               align_corners=True),

                                                   nn.Conv2d(in_channels=self.in_channels,
                                                             out_channels=self.in_channels,
                                                             kernel_size=3,
                                                             padding=1)
                                                  )

        # Upsample via transposed convolution (know to produce artifacts)
        self.up_sample_tranposed = nn.ConvTranspose2d(in_channels=self.in_channels,
                                                      out_channels=self.in_channels,
                                                      kernel_size=3,
                                                      stride=2,
                                                      padding=1,
                                                      output_padding=1)

        self.down_sample = EncoderBlock(in_channels=self.in_channels+self.concat_layer_depth,
                                        filter_num=self.filter_num,
                                        dropout=self.dropout,
                                        dropout_rate=self.dropout_rate)

    def forward(self, x, skip_layer, seeds=None):
        if self.interpolate is True:
            up_sample_out = F.leaky_relu(self.up_sample_interpolate(x))
        else:
            up_sample_out = F.leaky_relu(self.up_sample_tranposed(x))

        merged_out = torch.cat([up_sample_out, skip_layer], dim=1)
        out = self.down_sample(merged_out, seeds=seeds)
        return out


class EncoderBlock3D(nn.Module):

    """
    Instances the 3D Encoder block that forms a part of a 3D U-Net
    Parameters:
        in_channels (int): Depth (or number of channels) of the tensor that the block acts on
        filter_num (int) : Number of filters used in the convolution ops inside the block,
                             depth of the output of the enc block
    """
    def __init__(self, filter_num=64, in_channels=1, dropout=False):

        super(EncoderBlock3D, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.dropout = dropout

        self.conv1 = nn.Conv3d(in_channels=self.in_channels,
                               out_channels=self.filter_num,
                               kernel_size=3,
                               padding=1)

        self.conv2 = nn.Conv3d(in_channels=self.filter_num,
                               out_channels=self.filter_num*2,
                               kernel_size=3,
                               padding=1)

        self.bn_op_1 = nn.InstanceNorm3d(num_features=self.filter_num)
        self.bn_op_2 = nn.InstanceNorm3d(num_features=self.filter_num*2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn_op_1(x)
        x = F.leaky_relu(x)
        if self.dropout is True:
            x = F.dropout3d(x, p=0.3)

        x = self.conv2(x)
        x = self.bn_op_2(x)
        x = F.leaky_relu(x)

        if self.dropout is True:
            x = F.dropout3d(x, p=0.3)

        return x


class DecoderBlock3D(nn.Module):
    """
    Decoder block used in the 3D U-Net
    Parameters:
        in_channels (int) : Number of channels of the incoming tensor for the upsampling op
        concat_layer_depth (int) : Number of channels to be concatenated via skip connections
        filter_num (int) : Number of filters used in convolution, the depth of the output of the dec block
        interpolate (bool) : Decides if upsampling needs to performed via interpolation or transposed convolution
    """
    def __init__(self, in_channels, concat_layer_depth, filter_num, interpolate=False, dropout=False):

        super(DecoderBlock3D, self).__init__()
        self.filter_num = int(filter_num)
        self.in_channels = int(in_channels)
        self.concat_layer_depth = int(concat_layer_depth)
        self.interpolate = interpolate
        self.dropout = dropout

        # Upsample by interpolation followed by a 3x3x3 convolution to obtain desired depth
        self.up_sample_interpolate = nn.Sequential(nn.Upsample(scale_factor=2,
                                                               mode='nearest'),

                                                  nn.Conv3d(in_channels=self.in_channels,
                                                            out_channels=self.in_channels,
                                                            kernel_size=3,
                                                            padding=1)
                                                 )

        # Upsample via transposed convolution (know to produce artifacts)
        self.up_sample_transposed = nn.ConvTranspose3d(in_channels=self.in_channels,
                                                       out_channels=self.in_channels,
                                                       kernel_size=3,
                                                       stride=2,
                                                       padding=1,
                                                       output_padding=1)

        if self.dropout is True:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channels=self.in_channels+self.concat_layer_depth,
                                                       out_channels=self.filter_num,
                                                       kernel_size=3,
                                                       padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU(),

                                            nn.Dropout3d(p=0.3),

                                            nn.Conv3d(in_channels=self.filter_num,
                                                      out_channels=self.filter_num,
                                                      kernel_size=3,
                                                      padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU(),

                                            nn.Dropout3d(p=0.3))
        else:
            self.down_sample = nn.Sequential(nn.Conv3d(in_channels=self.in_channels+self.concat_layer_depth,
                                                       out_channels=self.filter_num,
                                                       kernel_size=3,
                                                       padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU(),

                                            nn.Conv3d(in_channels=self.filter_num,
                                                      out_channels=self.filter_num,
                                                      kernel_size=3,
                                                      padding=1),

                                            nn.InstanceNorm3d(num_features=self.filter_num),

                                            nn.LeakyReLU())

    def forward(self, x, skip_layer):

        if self.interpolate is True:
            up_sample_out = F.leaky_relu(self.up_sample_interpolate(x))
        else:
            up_sample_out = F.leaky_relu(self.up_sample_transposed(x))

        merged_out = torch.cat([up_sample_out, skip_layer], dim=1)
        out = self.down_sample(merged_out)
        return out

"""
A PyTorch Implementation of a U-Net.
Supports 2D (https://arxiv.org/abs/1505.04597) and 3D(https://arxiv.org/abs/1606.06650) variants
Author: Ishaan Bhat
Email: ishaan@isi.uu.nl
"""

from math import pow


class UNet(nn.Module):
    """
     PyTorch class definition for the U-Net architecture for image segmentation
     Parameters:
         n_channels (int) : Number of image channels
         base_filter_num (int) : Number of filters for the first convolution (doubled for every subsequent block)
         num_blocks (int) : Number of encoder/decoder blocks
         num_classes(int) : Number of classes that need to be segmented
         mode (str): 2D or 3D
         use_pooling (bool): Set to 'True' to use MaxPool as downnsampling op.
                             If 'False', strided convolution would be used to downsample feature maps (http://arxiv.org/abs/1908.02182)
         dropout (bool) : Whether dropout should be added to central encoder and decoder blocks (eg: BayesianSegNet)
         dropout_rate (float) : Dropout probability
     Returns:
         out (torch.Tensor) : Prediction of the segmentation map
     """
    def __init__(self, n_channels=1, base_filter_num=64, num_blocks=4, num_classes=5, mode='2D', dropout=False, dropout_rate=0.3, use_pooling=True):

        super(UNet, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.expanding_path = nn.ModuleList()
        self.downsampling_ops = nn.ModuleList()

        self.num_blocks = num_blocks
        self.n_channels = int(n_channels)
        self.n_classes = int(num_classes)
        self.base_filter_num = int(base_filter_num)
        self.enc_layer_depths = []  # Keep track of the output depths of each encoder block
        self.mode = mode
        self.pooling = use_pooling
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        if mode == '2D':
            self.encoder = EncoderBlock
            self.decoder = DecoderBlock
            self.pool = nn.MaxPool2d

        elif mode == '3D':
            self.encoder = EncoderBlock3D
            self.decoder = DecoderBlock3D
            self.pool = nn.MaxPool3d
        else:
            print('{} mode is invalid'.format(mode))

        for block_id in range(num_blocks):
            # Due to GPU mem constraints, we cap the filter depth at 512
            enc_block_filter_num = min(int(pow(2, block_id)*self.base_filter_num), 512)  # Output depth of current encoder stage of the 2-D variant
            if block_id == 0:
                enc_in_channels = self.n_channels
            else:
                if self.mode == '2D':
                    if int(pow(2, block_id)*self.base_filter_num) <= 512:
                        enc_in_channels = enc_block_filter_num//2
                    else:
                        enc_in_channels = 512
                else:
                    enc_in_channels = enc_block_filter_num  # In the 3D UNet arch, the encoder features double in the 2nd convolution op


            # Dropout only applied to central encoder blocks -- See BayesianSegNet by Kendall et al.
            if self.dropout is True and block_id >= num_blocks-2:
                self.contracting_path.append(self.encoder(in_channels=enc_in_channels,
                                                          filter_num=enc_block_filter_num,
                                                          dropout=True,
                                                          dropout_rate=self.dropout_rate))
            else:
                self.contracting_path.append(self.encoder(in_channels=enc_in_channels,
                                                          filter_num=enc_block_filter_num,
                                                          dropout=False))
            if self.mode == '2D':
                self.enc_layer_depths.append(enc_block_filter_num)
                if self.pooling is False:
                    self.downsampling_ops.append(nn.Sequential(nn.Conv2d(in_channels=self.enc_layer_depths[-1],
                                                                         out_channels=self.enc_layer_depths[-1],
                                                                         kernel_size=3,
                                                                         stride=2,
                                                                         padding=1),
                                                                nn.InstanceNorm2d(num_features=self.filter_num),
                                                                nn.LeakyReLU()))
            else:
                self.enc_layer_depths.append(enc_block_filter_num*2) # Specific to 3D U-Net architecture (due to doubling of #feature_maps inside the 3-D Encoder)
                if self.pooling is False:
                    self.downsampling_ops.append(nn.Sequential(nn.Conv3d(in_channels=self.enc_layer_depths[-1],
                                                                         out_channels=self.enc_layer_depths[-1],
                                                                         kernel_size=3,
                                                                         stride=2,
                                                                         padding=1),
                                                                nn.InstanceNorm3d(num_features=self.enc_layer_depths[-1]),
                                                                nn.LeakyReLU()))

        # Bottleneck layer
        if self.mode == '2D':
            bottle_neck_filter_num = self.enc_layer_depths[-1]*2
            bottle_neck_in_channels = self.enc_layer_depths[-1]
            self.bottle_neck_layer = self.encoder(filter_num=bottle_neck_filter_num,
                                                  in_channels=bottle_neck_in_channels)

        else:  # Modified for the 3D UNet architecture
            bottle_neck_in_channels = self.enc_layer_depths[-1]
            bottle_neck_filter_num = self.enc_layer_depths[-1]*2
            self.bottle_neck_layer =  nn.Sequential(nn.Conv3d(in_channels=bottle_neck_in_channels,
                                                              out_channels=bottle_neck_in_channels,
                                                              kernel_size=3,
                                                              padding=1),

                                                    nn.InstanceNorm3d(num_features=bottle_neck_in_channels),

                                                    nn.LeakyReLU(),

                                                    nn.Conv3d(in_channels=bottle_neck_in_channels,
                                                              out_channels=bottle_neck_filter_num,
                                                              kernel_size=3,
                                                              padding=1),

                                                    nn.InstanceNorm3d(num_features=bottle_neck_filter_num),

                                                    nn.LeakyReLU())

        # Decoder Path
        dec_in_channels = int(bottle_neck_filter_num)
        for block_id in range(num_blocks):
            if self.dropout is True and block_id < 2:
                self.expanding_path.append(self.decoder(in_channels=dec_in_channels,
                                                        filter_num=self.enc_layer_depths[-1-block_id],
                                                        concat_layer_depth=self.enc_layer_depths[-1-block_id],
                                                        interpolate=False,
                                                        dropout=True,
                                                        dropout_rate=self.dropout_rate))
            else:
                self.expanding_path.append(self.decoder(in_channels=dec_in_channels,
                                                        filter_num=self.enc_layer_depths[-1-block_id],
                                                        concat_layer_depth=self.enc_layer_depths[-1-block_id],
                                                        interpolate=False,
                                                        dropout=False))

            dec_in_channels = self.enc_layer_depths[-1-block_id]

        # Output Layer
        if mode == '2D':
            self.output = nn.Conv2d(in_channels=int(self.enc_layer_depths[0]),
                                    out_channels=self.n_classes,
                                    kernel_size=1)
        else:
            self.output = nn.Conv3d(in_channels=int(self.enc_layer_depths[0]),
                                    out_channels=self.n_classes,
                                    kernel_size=1)

    def forward(self, x, seeds=None):

        if self.mode == '2D':
            h, w = x.shape[-2:]
        else:
            d, h, w = x.shape[-3:]

        # Encoder
        enc_outputs = []
        seed_index = 0
        for stage, enc_op in enumerate(self.contracting_path):
            if stage >= len(self.contracting_path) - 2:
                if seeds is not None:
                    x = enc_op(x, seeds[seed_index:seed_index+2])
                else:
                    x = enc_op(x)
                seed_index += 2 # 2 seeds required per block
            else:
                x = enc_op(x)
            enc_outputs.append(x)

            if self.pooling is True:
                x = self.pool(kernel_size=2)(x)
            else:
                x = self.downsampling_ops[stage](x)

        # Bottle-neck layer
        x = self.bottle_neck_layer(x)
        # Decoder
        for block_id, dec_op in enumerate(self.expanding_path):
            if block_id < 2:
                if seeds is not None:
                    x = dec_op(x, enc_outputs[-1-block_id], seeds[seed_index:seed_index+2])
                else:
                    x = dec_op(x, enc_outputs[-1-block_id])
                seed_index += 2
            else:
                x = dec_op(x, enc_outputs[-1-block_id])


        # Output
        x = self.output(x)

        return x