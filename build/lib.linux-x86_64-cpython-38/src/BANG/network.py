import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import data as d 
import matplotlib.pyplot as plt
from collections import OrderedDict
'''
This is the Residual in Residual neural network structure!
'''


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

# I'LL PUT THEM IN DIFFERENT CLASSES SUCH THAT IF WE WANT TO ADD SIMILAR LAYERS IS EASIER

# out1 = 52
# out2 = 12


class features_extraction(nn.Module):
    def __init__(self,img_fil,out1,kernel_size=5,bias=True):
        
        super(features_extraction,self).__init__()

        self.feat_ext = nn.Conv2d(img_fil,
                                  out1,
                                  kernel_size,
                                  stride=1,
                                  padding=2,
                                  bias=bias)
        self.p_RELU   = nn.PReLU()  # as leaky Relu but slope is learnable
        #self.p_RELU   = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self,x):
        return self.p_RELU(self.feat_ext(x))

class schrinking(nn.Module):
    def __init__(self,out1,out2,kernel_size=1,bias=True):

        super(schrinking,self).__init__()

        self.shrink = nn.Conv2d(out1,
                                out2,
                                kernel_size,
                                stride=1,
                                padding=0,
                                bias=bias)
        self.p_RELU   = nn.PReLU()
        #self.p_RELU   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self,x):
        return self.p_RELU(self.shrink(x))

class mapping(nn.Module):
    def __init__(self,out2,n_layers,kernel_size=3,bias=True):
        
        super(mapping,self).__init__()

        mapping_block = OrderedDict()
        for i in range(n_layers):
            mapping_block[str(i)] = nn.Conv2d(out2,
                                              out2,
                                              kernel_size,
                                              stride=1,
                                              padding=1,
                                              bias=bias)
            mapping_block[str(i)] = nn.PReLU()
            #mapping_block[str(i)] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.mapping_block = nn.Sequential(mapping_block)

    def forward(self,x):
        return self.mapping_block(x)


class expanding(nn.Module):
    def __init__(self,out2,out1,kernel_size=1,bias=True):

        super(expanding,self).__init__()

        self.exp_layer = nn.Conv2d(out2,
                                   out1,
                                   kernel_size,
                                   stride=1,
                                   padding=0,
                                   bias=bias)
        self.p_RELU    = nn.PReLU()
        #self.p_RELU   = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self,x):
        return self.p_RELU(self.exp_layer(x))

class transp_conv(nn.Module):
    def __init__(self,out1,img_filter,kernel_size=9,scaling=4,bias=True):
        super(transp_conv,self).__init__()

        self.deconv_layer = nn.ConvTranspose2d(out1,
                                               img_filter,
                                               kernel_size,
                                               stride=2,
                                               padding=4,
                                               output_padding=1,
                                               bias=bias)
    
    def forward(self,x):
        return self.deconv_layer(x)




class FSRCNN_net(nn.Module):
    def __init__(self,img_filter,out1,out2,bias=True):
        super(FSRCNN_net,self).__init__()
        self.step1 = features_extraction(img_filter,out1)
        self.step2 = schrinking(out1,out2)
        self.step3 = mapping(out2,10)
        self.step4 = expanding(out2,out1)
        self.step5 = transp_conv(out1,2*out2)
        self.step6 = transp_conv(2*out2,img_filter)

    def forward(self,x):
        return self.step6(self.step5(self.step4(self.step3(self.step2(self.step1(x))))))



def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight.data)
            
        #elif isinstance(m, nn.ConvTranspose2d):
        #    nn.init.kaiming_normal_(m.weight.data)

if __name__=='__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    batch_size,img_filter,out1,out2 = 32,1,30,10

    model = FSRCNN_net(img_filter,         # filter imager grey or RGB
                       out1,               # filter for feat extraction
                       out2)               # reduced filter for mapping 
                         

    fake_img = torch.randn((batch_size,img_filter,20,20))


    HR_img = model(fake_img)

    