from __future__ import division

import torch
import torch.nn as nn
from collections import OrderedDict
from itertools import chain


class ConvResBlock(nn.Module):
    
    def __init__(self, n_repeats, in_channels, layer_number, return_at_end=False):
        super(ConvResBlock, self).__init__()
        number = layer_number
        # if return at end, then we will get the features before residual 
        # connection for future used
        self.return_at_end = return_at_end
        self.conv = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, 
                          kernel_size=3, stride=2, padding=1)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(in_channels, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1))
                    ]))
        number += 1
        self.repeat_list = nn.ModuleList()
        for i in range(n_repeats):
            
            repeat_block = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2,
                         kernel_size=1, stride=1, padding=0)),
                ('batch_norm_' + str(number),
                nn.BatchNorm2d(in_channels//2, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1)),
                
                ('conv_' + str(number+1), 
                nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels,
                         kernel_size=3, stride=1, padding=1)),
                ('batch_norm_' + str(number+1), 
                nn.BatchNorm2d(in_channels, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number+1), nn.LeakyReLU(0.1))
            ]))
            self.repeat_list.append(repeat_block)
            number += 2
    def forward(self, x):
        # initial skip that will be added to the block at the end 
        # of each repeat in the repeat list
        x = self.conv(x)
        x_skip = x
        feature_tensor = None
        for i, module in enumerate(self.repeat_list):
            x = module(x)
            if ((i == len(self.repeat_list) - 1) and self.return_at_end):
                feature_tensor = x
            x = x + x_skip
            x_skip = x
        if self.return_at_end:
            return x, feature_tensor
        else:
            return x

class Darknet53(nn.Module):

    def __init__(self):
        super(Darknet53, self).__init__()
        number = 1
        self.conv_1 = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=1, out_channels=32, 
                          kernel_size=3, stride=1, padding=1)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(32, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1))
                    ]))
        
        # repeated blocks of the  5 times with differet set of repeats
        self.block_1 = ConvResBlock(n_repeats=1, in_channels=64, layer_number=2)
        self.block_2 = ConvResBlock(n_repeats=2, in_channels=128, layer_number=5)
        self.block_3 = ConvResBlock(n_repeats=8, in_channels=256, layer_number=10, return_at_end=True)
        self.block_4 = ConvResBlock(n_repeats=8, in_channels=512, layer_number=27, return_at_end=True)
        self.block_5 = ConvResBlock(n_repeats=4, in_channels=1024, layer_number=44)
        
    def forward(self, x):
        x = self.conv_1(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x, features_scale_1 = self.block_3(x)
        x, features_scale_2 = self.block_4(x)
        x = self.block_5(x)
        return x, features_scale_1, features_scale_2

class AfterDarknet(nn.Module):
    
    def __init__(self, in_channels, layer_number, return_indices=[], first_channels=None):
        super(AfterDarknet, self).__init__()
        self.module_list = nn.ModuleList()
        number = layer_number
        self.return_indices = return_indices
        if first_channels is None:
            first_channels = in_channels
    
        conv_1 =  nn.Sequential(OrderedDict([
                ('conv_' + str(number) , 
                nn.Conv2d(in_channels=first_channels, out_channels=in_channels//2, 
                          kernel_size=1, stride=1, padding=0)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(in_channels//2, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1))
                    ]))
        self.module_list.append(conv_1)
        number += 1
        conv_2 = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, 
                          kernel_size=3, stride=1, padding=1)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(in_channels, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1))
                    ]))
        self.module_list.append(conv_2)
        number += 1
        conv_3 = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, 
                          kernel_size=1, stride=1, padding=0)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(in_channels//2, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1))
                    ]))
        self.module_list.append(conv_3)
        number += 1
        conv_4 = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, 
                          kernel_size=3, stride=1, padding=1)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(in_channels, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1))
                    ]))
        self.module_list.append(conv_4)
        number += 1
        conv_5 = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, 
                          kernel_size=1, stride=1, padding=0)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(in_channels//2, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1))
                    ]))
        self.module_list.append(conv_5)
        number += 1
        conv_6 = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, 
                          kernel_size=3, stride=1, padding=1)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(in_channels, momentum=0.1, eps=1e-5)),
                ('leaky_' + str(number), nn.LeakyReLU(0.1))
                    ]))
        self.module_list.append(conv_6)
        number += 1
        # no activation as it says linear
        conv_7 = nn.Sequential(OrderedDict([
                ('conv_' + str(number), 
                nn.Conv2d(in_channels=in_channels, out_channels=18, 
                          kernel_size=1, stride=1, padding=0)),
                ('batch_norm_' + str(number), 
                nn.BatchNorm2d(18, momentum=0.1, eps=1e-5)),
                    ]))
        self.module_list.append(conv_7)
    def forward(self, x):
        routed_outs = []
        for i, module in enumerate(self.module_list):
            x = module(x)
            if i in self.return_indices: # return conv_5 outputs to be fused later one
                routed_outs.append(x)
        return x, routed_outs

class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes):
        super(YOLOLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.no = num_classes + 5  # number of outputs per anchor
        self.grid = torch.zeros(1)

        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.stride = None

        
    def forward(self, x, img_size):
        # img_size is a tuple
        # make stride a tuple
        stride = torch.tensor((img_size[0] // x.size(2), img_size[1] // x.size(3)))
        self.stride = stride
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        #if not self.training:  # inference
        #    if self.grid.shape[2:4] != x.shape[2:4]:
        #    self.grid = self._make_grid(nx, ny).to(x.device)

        #    x[..., 0:2] = (x[..., 0:2].sigmoid() + self.grid) * stride  # xy
        #    x[..., 2:4] = torch.exp(x[..., 2:4]) * self.anchor_grid # wh
        #    x[..., 4:] = x[..., 4:].sigmoid()
        #    x = x.view(bs, -1, self.no)

        return x
    
    @staticmethod
    def _make_grid(nx, ny):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class YOLOv3(nn.Module):
    
    def __init__(self, anchors=[[(19, 168), (21, 168), (29, 178)],
                                [(11, 130), (11, 131), (15, 165)],
                                [(6, 115), (7, 115), (7, 125)]], num_classes=1):
        super(YOLOv3, self).__init__()
        self.dark53 = Darknet53()
        self.afterdark53_1 = AfterDarknet(in_channels=1024, layer_number=53, return_indices=[4])
        self.yolo_1 = YOLOLayer(anchors=anchors[0], num_classes=num_classes)
        
        #self.afterdark53_2 = AfterDarknet(in_channels=512, layer_number=61, return_indices=[])
        self.conv_after_yolo_1 = nn.Sequential(OrderedDict([
                ('conv_60', 
                nn.Conv2d(in_channels=512, out_channels=256, 
                          kernel_size=1, stride=1, padding=0)),
                ('batch_norm_60', 
                nn.BatchNorm2d(256, momentum=0.1, eps=1e-5)),
                ('leaky_60', nn.LeakyReLU(0.1))
                    ]))
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.afterdark53_2 = AfterDarknet(in_channels=512, layer_number=61, return_indices=[4],
                                            first_channels=768)
        self.yolo_2 = YOLOLayer(anchors=anchors[1], num_classes=num_classes)
        
        self.conv_after_yolo_2 = nn.Sequential(OrderedDict([
                ('conv_68', 
                nn.Conv2d(in_channels=256, out_channels=128, 
                          kernel_size=1, stride=1, padding=0)),
                ('batch_norm_68', 
                nn.BatchNorm2d(128, momentum=0.1, eps=1e-5)),
                ('leaky_68', nn.LeakyReLU(0.1))
                    ]))
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.afterdark53_3 = AfterDarknet(in_channels=256, layer_number=69, return_indices=[],
                                         first_channels=384)
        
        self.yolo_3 = YOLOLayer(anchors=anchors[2], num_classes=num_classes)
    
    @classmethod
    def parse(cls, *args, **kwargs):
        return cls(*args, **kwargs)
        
    def forward(self, x):
        yolo_outputs = []
        img_size = (x.shape[2], x.shape[3])
        #print(f"Before network shape: {x.shape}")
        x, features_scale_1, features_scale_2 = self.dark53(x)
        #print(f"After darknet backbone: {x.shape}, features1: {features_scale_1.shape}, features2: {features_scale_2.shape}")
        x, routed_outs1 = self.afterdark53_1(x)
        combined_outputs1 = torch.cat([out for out in routed_outs1], 1)
        #print(f"After darknet 1 shape: {x.shape}, {combined_outputs1.shape}")
        x = self.yolo_1(x, img_size)
        #print(f"After yolo_1 shape: {x.shape}")
        yolo_outputs.append(x)
        
        # we used combined outputs to get stuff at other scales and
        # combine it with something at previous scales
        x = self.conv_after_yolo_1(combined_outputs1)
        x = self.upsample1(x)
        #print(f"After first upsample shape: {x.shape}")
        
        # one more route to add filters from higher resolution
        x = torch.cat([features_scale_2, x], 1)
        #print(f"After concatenation of features: {x.shape}")
        
        #
        x, routed_outs2 = self.afterdark53_2(x)
        combined_outputs2 = torch.cat([out for out in routed_outs2], 1)
        #print(f"After darknet 2 shape: {x.shape}, {combined_outputs2.shape}")
        
        x = self.yolo_2(x, img_size)
        #print(f"After yolo_2 shape: {x.shape}")
        yolo_outputs.append(x)
        
        x = self.conv_after_yolo_2(combined_outputs2)
        x = self.upsample2(x)
        #print(f"After second upsample: {x.shape}")
        
        # add features from highest resolution
        x = torch.cat([features_scale_1, x], 1)
        #print(f"After concatenation of features 2: {x.shape}")
        
        # routed out will be empty here as we dont used
        x, routed_outs3 = self.afterdark53_3(x)
        #combined_outputs3 = torch.cat([out for out in routed_outs3], 1)
        #print(f"After darknet 3 shape: {x.shape}")
        
        x = self.yolo_3(x, img_size)
        #print(f"After yolo_3 shape: {x.shape}")
        yolo_outputs.append(x)
        return yolo_outputs

model_dict = {
    'YOLOv3' : YOLOv3
} 