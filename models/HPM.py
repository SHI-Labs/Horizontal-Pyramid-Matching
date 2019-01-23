import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
import torch.nn.functional as F
from .ResNet import resnet50
# from random_erasing import RandomErasing_vertical, RandomErasing_2x2
import math

__all__ = ['HPM']
######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        # init.constant(m.bias.data, 0.0)

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)

    

def pcb_block(num_ftrs, num_stripes, local_conv_out_channels, num_classes, avg=False):
    if avg:
        pooling_list = nn.ModuleList([nn.AdaptiveAvgPool2d(1) for _ in range(num_stripes)])
    else:
        pooling_list = nn.ModuleList([nn.AdaptiveMaxPool2d(1) for _ in range(num_stripes)])
    conv_list = nn.ModuleList([nn.Conv2d(num_ftrs, local_conv_out_channels, 1, bias=False) for _ in range(num_stripes)])
    batchnorm_list = nn.ModuleList([nn.BatchNorm2d(local_conv_out_channels) for _ in range(num_stripes)])
    relu_list = nn.ModuleList([nn.ReLU(inplace=True) for _ in range(num_stripes)])
    fc_list = nn.ModuleList([nn.Linear(local_conv_out_channels, num_classes, bias=False) for _ in range(num_stripes)])
    for m in conv_list:
        weight_init(m)
    for m in batchnorm_list:
        weight_init(m)
    for m in fc_list:
        weight_init(m)
    return pooling_list, conv_list, batchnorm_list, relu_list, fc_list


def spp_vertical(feats, pool_list, conv_list, bn_list, relu_list, fc_list, num_strides, feat_list=[], logits_list=[]):
    for i in range(num_strides):
        pcb_feat = pool_list[i](feats[:, :, i * int(feats.size(2) / num_strides): (i+1) *  int(feats.size(2) / num_strides), :])
        pcb_feat = conv_list[i](pcb_feat)
        pcb_feat = bn_list[i](pcb_feat)
        pcb_feat = relu_list[i](pcb_feat)
        pcb_feat = pcb_feat.view(pcb_feat.size(0), -1)
        feat_list.append(pcb_feat)
        logits_list.append(fc_list[i](pcb_feat))
    return feat_list, logits_list

def global_pcb(feats, pool, conv, bn, relu, fc, feat_list=[], logits_list=[]):
    global_feat = pool(feats)
    global_feat = conv(global_feat)
    global_feat = bn(global_feat)
    global_feat = relu(global_feat)
    global_feat = global_feat.view(feats.size(0), -1)
    feat_list.append(global_feat)
    logits_list.append(fc(global_feat))
    return feat_list, logits_list




class HPM(nn.Module):
    def __init__(self, num_classes, num_stripes=6, local_conv_out_channels=256, erase=0, loss={'xent'}, avg=False, **kwargs):
        super(HPM, self).__init__()
        self.erase = erase
        self.num_stripes = num_stripes
        self.loss = loss

        model_ft = resnet50(pretrained=True, remove_last=True, last_conv_stride=1)
        self.num_ftrs = list(model_ft.layer4)[-1].conv1.in_channels
        self.features = model_ft
        # PSP
        # self.psp_pool, self.psp_conv, self.psp_bn, self.psp_relu, self.psp_upsample, self.conv = psp_block(self.num_ftrs)

        # global
        self.global_pooling = nn.AdaptiveMaxPool2d(1)
        self.global_conv = nn.Conv2d(self.num_ftrs, local_conv_out_channels, 1, bias=False)
        self.global_bn = nn.BatchNorm2d(local_conv_out_channels)
        self.global_relu = nn.ReLU(inplace=True)
        self.global_fc = nn.Linear(local_conv_out_channels, num_classes, bias=False)

        weight_init(self.global_conv)
        weight_init(self.global_bn) 
        weight_init(self.global_fc)


        # 2x
        self.pcb2_pool_list, self.pcb2_conv_list, self.pcb2_batchnorm_list, self.pcb2_relu_list, self.pcb2_fc_list = pcb_block(self.num_ftrs, 2, local_conv_out_channels, num_classes, avg)
        # 4x
        self.pcb4_pool_list, self.pcb4_conv_list, self.pcb4_batchnorm_list, self.pcb4_relu_list, self.pcb4_fc_list = pcb_block(self.num_ftrs, 4, local_conv_out_channels, num_classes, avg)
        # 8x
        self.pcb8_pool_list, self.pcb8_conv_list, self.pcb8_batchnorm_list, self.pcb8_relu_list, self.pcb8_fc_list = pcb_block(self.num_ftrs, 8, local_conv_out_channels, num_classes, avg)

        

    def forward(self, x):
        feat_list = []
        logits_list = []
        feats = self.features(x) # N, C, H, W
        assert feats.size(2) == 24
        assert feats.size(-1) == 8
        assert feats.size(2) % self.num_stripes == 0
        
        if self.erase>0:
        #    print('Random Erasing')
            erasing = RandomErasing_vertical(probability=self.erase)
            feats = erasing(feats)
        
        feat_list, logits_list = global_pcb(feats, self.global_pooling, self.global_conv, self.global_bn, 
                    self.global_relu, self.global_fc, [], [])
        feat_list, logits_list = spp_vertical(feats, self.pcb2_pool_list, self.pcb2_conv_list, 
                    self.pcb2_batchnorm_list, self.pcb2_relu_list, self.pcb2_fc_list, 2, feat_list, logits_list)
        feat_list, logits_list = spp_vertical(feats, self.pcb4_pool_list, self.pcb4_conv_list, 
                    self.pcb4_batchnorm_list, self.pcb4_relu_list, self.pcb4_fc_list, 4, feat_list, logits_list)

        feat_list, logits_list = spp_vertical(feats, self.pcb8_pool_list, self.pcb8_conv_list, 
                    self.pcb8_batchnorm_list, self.pcb8_relu_list, self.pcb8_fc_list, 8, feat_list, logits_list)
    
        if not self.training:
            return torch.cat(feat_list, dim=1)

        if self.loss == {'xent'}:
            return logits_list
        elif self.loss == {'xent', 'htri'}:
            return logits_list, feat_list
        elif self.loss == {'cent'}:
            return logits_list, feat_list
        elif self.loss == {'ring'}:
            return logits_list, feat_list
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))