# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import numpy as np
import pickle
from torch.autograd import Variable
from util import normalize
import cv2
from third_party.DPT.dpt.models import DPTDepthModel
from third_party.DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose, ToTensor, InterpolationMode, Normalize
import torchvision
from third_party.DPT.util import io
import os
import uuid
import imageio
from lib.utils.render_utils import remove_noise_in_dpt_disparity
from lib.utils.data_utils import resize_img
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

class Ruler:
    def __init__(self, config):
        self.motion_input_transform = Compose(
            [
                torchvision.transforms.Resize((config['motionH'], config['motionW']),
                                              InterpolationMode.BICUBIC),
                ToTensor(),
                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.to_tensor = ToTensor()
class DPTModel:
    def __init__(self):
        net_w, net_h = 384, 384
        self.model = DPTDepthModel(
            path='ckpts/dpt_hybrid-midas-501f0c75.pt',
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )
        self.model.eval()
        self.model.cuda()

    def run_dpt(self,image,video_out_folder):
        # get input
        img = image/255.0
        img_input = self.transform({"image": img})["image"]

        # compute
        with torch.no_grad():
            sample = torch.from_numpy(img_input).cuda().unsqueeze(0)
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        dpt_file = os.path.join(
            video_out_folder, 'dpt'
        )
        io.write_depth(dpt_file, prediction, bits=2, absolute_depth=False)
        src_disp = imageio.imread(dpt_file+'.png') / 65535.
        src_disp = remove_noise_in_dpt_disparity(src_disp)
        src_depth = 1. / np.maximum(src_disp, 1e-6)
        src_depth = resize_img(src_depth, 1)
        return src_depth

class SAM:
    def __init__(self):
        sam = sam_model_registry["vit_b"](checkpoint="ckpts/sam_vit_b_01ec64.pth").cuda()
        # sam = sam_model_registry["vit_l"](checkpoint="ckpts/sam_vit_l_0b3195.pth").cuda()
        # sam = sam_model_registry["vit_h"](checkpoint="ckpts/sam_vit_h_4b8939.pth").cuda()
        self.mask_generator = SamAutomaticMaskGenerator(sam,stability_score_thresh=0.5)
    def inference(self, image, prior_mask):
        masks = self.mask_generator.generate(image)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)[:50]
        ret_mask=torch.zeros((768,768),dtype=torch.bool)
        all_masks=F.interpolate(torch.tensor(np.array([[mask['segmentation'] for mask in masks]])).bool().float(), (768, 768), mode='area').bool()[0]
        for mask in all_masks:
            if ((mask&prior_mask).sum())/mask.sum()>0.7:
                ret_mask|=mask
        return ret_mask[None,None].float()

    def inference2(self, image):
        masks = self.mask_generator.generate(image)
        masks = sorted(masks, key=lambda x: x['area'], reverse=True)
        ret_mask=torch.zeros((768,768),dtype=torch.bool)
        all_masks=F.interpolate(torch.tensor(np.array([[mask['segmentation'] for mask in masks]])).bool().float(), (768, 768), mode='area').bool()[0]
        for mask,org_mask in zip(all_masks,masks):
            if ret_mask.sum()/(768*768)>0.5:
                break
            else:
                ret_mask|=mask
        return ret_mask[None,None].float()
class MaskFormer:
    def __init__(self):
        self.processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to('cuda')

    def inference(self, original_image, text):
        threshold = 0.8
        padding = 20
        image = Image.fromarray(original_image).resize((512, 512))
        inputs = self.processor(text=text, images=image, padding="max_length", return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
        mask = torch.sigmoid(outputs[0]).squeeze().cpu().numpy() > threshold
        area_ratio = len(np.argwhere(mask)) / (mask.shape[0] * mask.shape[1])
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        mask_array = F.interpolate(torch.tensor(mask_array[None, None]).bool().float(), (768, 768), mode='area')
        return mask_array
class OpticalPridictor:
    def __init__(self):
        self.P_m = ConditionalMotionNet()
        param = torch.load('ckpts/PMNet_weight_5000.pth')
        self.P_m.load_state_dict(param)
        self.P_m.cuda()
        with open('ckpts/codebook_m_5000.pkl', 'rb') as f:
            self.codebook_m = pickle.load(f, encoding='latin1')

    def get_hint(self, image, mask):
        w, h = 256, 256
        t_m = np.random.rand()
        id1 = int(np.floor((len(self.codebook_m) - 1) * t_m))
        id2 = int(np.ceil((len(self.codebook_m) - 1) * t_m))
        z_weight = (len(self.codebook_m) - 1) * t_m - np.floor((len(self.codebook_m) - 1) * t_m)
        z_m = (1. - z_weight) * self.codebook_m[id1:id1 + 1] + z_weight * self.codebook_m[id2:id2 + 1]
        z_m = Variable(torch.from_numpy(z_m.astype(np.float32)))
        z_m = z_m.cuda()
        with torch.no_grad():
            test_img_large = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            test_img = cv2.resize(test_img_large, (w, h))
            test_input = np.array([normalize(test_img)])
            test_input = Variable(torch.from_numpy(test_input.transpose(0, 3, 1, 2)))
            test_input = test_input.cuda()
            flow = self.P_m(test_input, z_m)[:,[1,0]]
            flow/=(flow.std()+1e-7)
            flow = F.interpolate(flow, (768, 768), mode='bilinear', align_corners=False).cpu()
            flow = flow*mask
        return flow

class ConditionalMotionNet(torch.nn.Module):
    def __init__(self, nz=8, nout=2, beta=1./64.):
        super(ConditionalMotionNet, self).__init__()
        c_num = 128
         
        # Downsampling layers
        self.conv1 = ConvLayer(3+nz, c_num, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(c_num+nz, c_num*2, kernel_size=3, stride=2)
        self.in2 = InstanceNormalization(c_num*2)
        self.conv3 = ConvLayer(c_num*2+nz, c_num*4, kernel_size=3, stride=2)
        self.in3 = InstanceNormalization(c_num*4)
 
        # Residual layers
        self.res1 = ResidualBlock(c_num*4)
        self.res2 = ResidualBlock(c_num*4)
        self.res3 = ResidualBlock(c_num*4)
        self.res4 = ResidualBlock(c_num*4)
        self.res5 = ResidualBlock(c_num*4)
 
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(c_num*4*2, c_num*2, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNormalization(c_num*2)
        self.deconv2 = UpsampleConvLayer(c_num*2*2, c_num, kernel_size=3, stride=1, upsample=2)
        self.in5 = InstanceNormalization(c_num)
        self.deconv3 = UpsampleConvLayer(c_num*2, nout, kernel_size=5, stride=1, upsample=2)

        self.relu = nn.LeakyReLU(0.1,inplace=True)
        self.beta = beta

    def forward(self, x, z, frame_size=0):
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat((x, z2D), 1)
        h1 = self.relu(self.conv1(x_with_z))
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), h1.size(2), h1.size(3))
        h1_with_z =  torch.cat((h1, z2D), 1)
        h2 = self.relu(self.in2(self.conv2(h1_with_z)))
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), h2.size(2), h2.size(3))
        h2_with_z =  torch.cat((h2, z2D), 1)
        h3 = self.relu(self.in3(self.conv3(h2_with_z)))
          
        h4 = self.res1(h3)
        h4 = self.res2(h4)
        h4 = self.res3(h4)
        h4 = self.res4(h4)
        h4 = self.res5(h4)

        h4 = torch.cat((h4,h3),1)
        h5 = self.relu(self.in4(self.deconv1(h4)))
        h5 = torch.cat((h5,h2),1)
        h6 = self.relu(self.in5(self.deconv2(h5)))
        h6 = torch.cat((h6,h1),1)
        h7 = self.deconv3(h6)
         
        y = F.tanh(h7)*self.beta
        
        return y

class ConditionalAppearanceNet(torch.nn.Module):
    def __init__(self, nz=8):
        super(ConditionalAppearanceNet, self).__init__()
        c_num = 128
          
        # Downsampling layers
        self.conv1 = ConvLayer(3+nz, c_num, kernel_size=5, stride=2)
        self.conv2 = ConvLayer(c_num+nz, c_num*2, kernel_size=3, stride=2)
        self.in2 = InstanceNormalization(c_num*2)
        self.conv3 = ConvLayer(c_num*2+nz, c_num*4, kernel_size=3, stride=2)
        self.in3 = InstanceNormalization(c_num*4)
  
        # Residual layers
        self.res1 = ResidualBlock(c_num*4)
        self.res2 = ResidualBlock(c_num*4)
        self.res3 = ResidualBlock(c_num*4)
        self.res4 = ResidualBlock(c_num*4)
        self.res5 = ResidualBlock(c_num*4)
  
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(c_num*4*2, c_num*2, kernel_size=3, stride=1, upsample=2)
        self.in4 = InstanceNormalization(c_num*2)
        self.deconv2 = UpsampleConvLayer(c_num*2*2, c_num, kernel_size=3, stride=1, upsample=2)
        self.in5 = InstanceNormalization(c_num)
        self.deconv3 = UpsampleConvLayer(c_num*2, 6, kernel_size=5, stride=1, upsample=2)

        self.fc1 = nn.Linear(c_num*4, 6)
        self.relu = nn.LeakyReLU(0.1,inplace=True)
  
    def forward(self, x, z):
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), x.size(2), x.size(3))
        x_with_z = torch.cat((x, z2D), 1)
        h1 = self.relu(self.conv1(x_with_z))
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), h1.size(2), h1.size(3))
        h1_with_z =  torch.cat((h1, z2D), 1)
        h2 = self.relu(self.in2(self.conv2(h1_with_z)))
        z2D = z.view(z.size(0), z.size(1), 1, 1).expand(z.size(0), z.size(1), h2.size(2), h2.size(3))
        h2_with_z =  torch.cat((h2, z2D), 1)
        h3 = self.relu(self.in3(self.conv3(h2_with_z)))
          
        h4 = self.res1(h3)
        h4 = self.res2(h4)
        h4 = self.res3(h4)
        h4 = self.res4(h4)
        h4 = self.res5(h4)
       
        h4 = torch.cat((h4,h3),1)
        h5 = self.relu(self.in4(self.deconv1(h4)))
        h5 = torch.cat((h5,h2),1)
        h6 = self.relu(self.in5(self.deconv2(h5)))
        h6 = torch.cat((h6,h1),1)
        h7 = self.deconv3(h6)
        al, bl = h7.split(3,dim=1)

        Y = al*x+bl
        Y = F.tanh(Y)
        return Y, al, bl
    
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = InstanceNormalization(channels)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = InstanceNormalization(channels)
        #self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.1,inplace=True)
     
    def forward(self, x, x_o=None):
        if x_o is None:
            residual = x
            out = self.conv1(self.relu(self.in1(x)))
            out = self.conv2(self.relu(self.in2(out)))
            out = out + residual
            return out
        
        residual = x
        residual_o = x_o
        out = self.conv1(self.relu(self.in1(x, x_o)))
        out_o = self.conv1(self.relu(self.in1(x_o, x_o)))
        out = self.conv2(self.relu(self.in2(out, out_o)))
        out_o = self.conv2(self.relu(self.in2(out_o, out_o)))
        out = out + residual
        out_o = out_o + residual_o
        return out, out_o


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class InstanceNormalization(torch.nn.Module):
    #Original code from https://github.com/abhiskk/fast-neural-style
    """InstanceNormalization
    Improves convergence of neural-style.
    ref: https://arxiv.org/pdf/1607.08022.pdf
    """

    def __init__(self, dim, eps=1e-9):
        super(InstanceNormalization, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor(dim))
        self.shift = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.scale.data.uniform_()
        self.shift.data.zero_()

    def forward(self, x, x_o=None):
        if x_o is None:
            x_o = x
        n = x_o.size(2) * x_o.size(3)
        t = x_o.view(x_o.size(0), x_o.size(1), n)
        mean = torch.mean(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x)
        # Calculate the biased var. torch.var returns unbiased var
        var = torch.var(t, 2).unsqueeze(2).unsqueeze(3).expand_as(x) * ((n - 1) / float(n))
        scale_broadcast = self.scale.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        scale_broadcast = scale_broadcast.expand_as(x)
        shift_broadcast = self.shift.unsqueeze(1).unsqueeze(1).unsqueeze(0)
        shift_broadcast = shift_broadcast.expand_as(x)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * scale_broadcast + shift_broadcast
        return out
    
class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = [
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        ]
        for idx, module in enumerate(self.main):
            self.add_module(str(idx), module)
    def forward(self, x):
        for layer in self.main:
            x = layer(x)
        return x