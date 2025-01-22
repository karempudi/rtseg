import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import List

class ConvBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int, 
            kernerl_size:int =3, padding: int = 1, stride: int = 1,
            bias: bool = True):
        """
        A conv block of a U-net module

        Takes an input tensor (N, c_in, H, W) --> (N, c_out, H, W)

        Args:
            c_in  (int) : Number of channels in the input 
            c_out (int) : Number of channels in the output
            kernel_size (int) : kernel_size of the conv block
                defualt is 3
            padding (int): padding used in conv layers
                defaults is 1
            stride (int): stride of the conv block
                defaults is 1
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernerl_size, 
                    padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=kernerl_size, 
                    padding=padding, stride=stride, bias=bias),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class UpsampleBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int, 
        upsample_type: str ='transpose_conv', feature_fusion_type: str='concat'):
        """
        An upsample block in the U-net architecture

        Args:
            c_in (int): Number of channels in the input
            c_out (int): Number of channels in the output
            upsample_type (str): upsample method, 'transpose_conv' (defualt) or 'upsample'
            feature_fusion_type (str) : 'concat' or 'add', method by which the features
                from the downsample part are added to the upsample features 
        """
        super().__init__()
        self.upsample_type = upsample_type
        self.feature_fusion_type = feature_fusion_type

        if self.upsample_type == 'transpose_conv':
            self.upsample_block = nn.ConvTranspose2d(c_in, c_in, kernel_size=2, stride=2)
        elif self.upsample_type == 'upsample':
            self.upsample_block = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.feature_fusion_type == 'concat':
            c_in = c_in + c_out
            self.conv_block = ConvBlock(c_in, c_out)
        elif self.feature_fusion_type == 'add':
            self.shrink_block = ConvBlock(c_in, c_out) # so that the channels match
            self.conv_block = ConvBlock(c_out, c_out)
        elif self.feature_fusion_type is None:
            self.conv_block = ConvBlock(c_in, c_out)

    def forward(self, x, features=None):
        if features is not None:
            # then you have something coming from the side
            x = self.upsample_block(x)
            if self.feature_fusion_type == 'concat':
                x = torch.cat([features, x], dim=1) # concat along the channel 'C' dimension
            elif self.feature_fusion_type == 'add':
                x = self.shrink_block(x) + features
            x = self.conv_block(x)
            return x
        else:
            x = self.upsample_block(x)
            x = self.conv_block(x)
            return x

class Unet(nn.Module):

    def __init__(self, channels_by_scale: List[int], num_outputs: int = 1,
            upsample_type: str ='transpose_conv', feature_fusion_type: str ='concat'):
        """
        A U-net architecture

        Args:
            channels_by_scale (List[int]): number of channels in each layers 
                of the U-net for ex: [1, 8, 16, 32, 64]. Always start with 1
            num_outputs (int): number of output channels, defuault 1
            upsample_type (str) : 'transpose_conv' or 'upsample', type of 
                upsampling method in the upward path of the network
            feature_fusion_type (str): 'concat' or 'add' how features are added
                from the downward side to the upward side
        """
        super().__init__()

        self.hparams = SimpleNamespace(channels_by_scale=channels_by_scale,
                                       num_outputs=num_outputs,
                                       upsample_type=upsample_type,
                                       feature_fusion_type=feature_fusion_type)
        # create the network
        self._create_network()
        # initalize some layers
        self._init_params()

    @classmethod
    def parse(cls, *args, **kwargs):
        return cls(*args, **kwargs)


    def _create_network(self):
        down_layers = []
        for layer_idx in range(len(self.hparams.channels_by_scale) - 1):
            down_layers.append(
                ConvBlock(self.hparams.channels_by_scale[layer_idx],
                          self.hparams.channels_by_scale[layer_idx+1])
                          )
            if layer_idx < len(self.hparams.channels_by_scale) - 2:
                down_layers.append(nn.MaxPool2d(2))

        self.down_layers = nn.Sequential(*down_layers)

        up_layers = []
        reversed_channels = self.hparams.channels_by_scale[::-1]
        for layer_idx in range(len(reversed_channels) - 2):
            up_layers.append(
                UpsampleBlock(reversed_channels[layer_idx],
                              reversed_channels[layer_idx+1],
                              upsample_type=self.hparams.upsample_type,
                              feature_fusion_type=self.hparams.feature_fusion_type)
            )
        
        self.up_layers = nn.Sequential(*up_layers)

        self.last_conv = nn.Conv2d(self.hparams.channels_by_scale[1],
                                   self.hparams.num_outputs,
                                   kernel_size=1)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # as we are using relu activations we use kaiming initialization
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        # pass down and accumulate the values needed to 
        # be used on the upsample blocks
        features = []
        for i, layer in enumerate(self.down_layers):
            #print(f"Down Input layer no : {i} - shape: {x.shape}")
            x = layer(x)
            if i%2 == 0 and i < len(self.down_layers) - 1:
                features.append(x)
        for i, layer in enumerate(self.up_layers, 1):
            #print(f"Input Up sample layers {i} -- {x.shape}, features")
            x = layer(x, features[-i])

        x = self.last_conv(x) 

        return x


############################################
############## Conv blocks with ############
#############   res connections ############
############################################

class ResConvBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int,
            kernel_size : int = 3, padding: int = 1, stride : int = 1):
        """
        A residual connnection added to the a conv block  
        of a typical U-net

        Args:
            c_in (int):  number of input channels
            c_out (int): number of output channels
            kernel_size (int) : kernel size in the convolutional layers,
                defualt is 3
            padding (int): padding used in the convolutional layers,
                defualt is 1
            stride (int) : stride used in the convolutional layers,
                defualt is 1
        """
        super().__init__()
        self.block  = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=kernel_size, padding=padding, stride=stride),
        )

        # bathc norm- relu- weight, norm relu weight,
        # add

        self.skip_connection = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, padding=padding, stride=stride)
        )

        self.nonlinearity = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.block(x) + self.skip_connection(x)
        x = self.nonlinearity(x)
        return x


class ResUpsampleBlock(nn.Module):

    def __init__(self, c_in: int, c_out: int, 
        upsample_type: str ='transpose_conv', feature_fusion_type: str='concat'):
        """
        An upsample block in the U-net architecture using residual blocks

        Args:
            c_in (int): Number of channels in the input
            c_out (int): Number of channels in the output
            upsample_type (str): upsample method, 'transpose_conv' (defualt) or 'upsample'
            feature_fusion_type (str) : 'concat' or 'add', method by which the features
                from the downsample part are added to the upsample features 
        """
        super().__init__()
        self.upsample_type = upsample_type
        self.feature_fusion_type = feature_fusion_type

        if self.upsample_type == 'transpose_conv':
            self.upsample_block = nn.ConvTranspose2d(c_in, c_in, kernel_size=2, stride=2)
        elif self.upsample_type == 'upsample':
            self.upsample_block = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        if self.feature_fusion_type == 'concat':
            c_in = c_in + c_out
            self.conv_block = ResConvBlock(c_in, c_out)
        elif self.feature_fusion_type == 'add':
            self.shrink_block = ResConvBlock(c_in, c_out) # so that the channels match
            self.conv_block = ResConvBlock(c_out, c_out)
        elif self.feature_fusion_type is None:
            self.conv_block = ResConvBlock(c_in, c_out)

    def forward(self, x, features=None):
        if features is not None:
            # then you have something coming from the side
            x = self.upsample_block(x)
            if self.feature_fusion_type == 'concat':
                x = torch.cat([features, x], dim=1) # concat along the channel 'C' dimension
            elif self.feature_fusion_type == 'add':
                x = self.shrink_block(x) + features
            x = self.conv_block(x)
            return x
        else:
            x = self.upsample_block(x)
            x = self.conv_block(x)
            return x

class ResUnet(nn.Module):

    def __init__(self, channels_by_scale: List[int], num_outputs: int = 1,
            upsample_type: str ='transpose_conv', feature_fusion_type: str ='concat'):
        """
        A U-net architecture with residual blocks

        Args:
            channels_by_scale (List[int]): number of channels in each layers 
                of the U-net for ex: [1, 8, 16, 32, 64]. Always start with 1
            num_outputs (int): number of output channels, defuault 1
            upsample_type (str) : 'transpose_conv' or 'upsample', type of 
                upsampling method in the upward path of the network
            feature_fusion_type (str): 'concat' or 'add' how features are added
                from the downward side to the upward side
        """
        super().__init__()

        self.hparams = SimpleNamespace(channels_by_scale=channels_by_scale,
                                       num_outputs=num_outputs,
                                       upsample_type=upsample_type,
                                       feature_fusion_type=feature_fusion_type)
        # create the network
        self._create_network()
        # initalize some layers
        self._init_params()

    @classmethod
    def parse(cls, *args, **kwargs):
        return cls(*args, **kwargs)

    def _create_network(self):
        down_layers = []
        for layer_idx in range(len(self.hparams.channels_by_scale) - 1):
            down_layers.append(
                ResConvBlock(self.hparams.channels_by_scale[layer_idx],
                          self.hparams.channels_by_scale[layer_idx+1])
                          )
            if layer_idx < len(self.hparams.channels_by_scale) - 2:
                down_layers.append(nn.MaxPool2d(2))

        self.down_layers = nn.Sequential(*down_layers)

        up_layers = []
        reversed_channels = self.hparams.channels_by_scale[::-1]
        for layer_idx in range(len(reversed_channels) - 2):
            up_layers.append(
                ResUpsampleBlock(reversed_channels[layer_idx],
                              reversed_channels[layer_idx+1],
                              upsample_type=self.hparams.upsample_type,
                              feature_fusion_type=self.hparams.feature_fusion_type)
            )
        
        self.up_layers = nn.Sequential(*up_layers)

        self.last_conv = nn.Conv2d(self.hparams.channels_by_scale[1],
                                   self.hparams.num_outputs,
                                   kernel_size=1)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                # as we are using relu activations we use kaiming initialization
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        # pass down and accumulate the values needed to 
        # be used on the upsample blocks
        features = []
        for i, layer in enumerate(self.down_layers):
            x = layer(x)
            if i%2 == 0 and i < len(self.down_layers) - 1:
                features.append(x)
        
        for i, layer in enumerate(self.up_layers, 1):
            x = layer(x, features[-i])

        x = self.last_conv(x) 

        return x

model_dict = {
    'Unet': Unet,
    'ResUnet': ResUnet
}