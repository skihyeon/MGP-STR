import os
import time
import string
import argparse
import re
import PIL
import math

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from matplotlib import pyplot as plt
from matplotlib import colors
import cv2
from torchvision import transforms
import torchvision.utils as vutils

from utils import TokenLabelConverter, CTCLabelConverter
from models import Model
from utils import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_model(image_tensors, model, converter, opt):
    image = image_tensors.to(device)
    batch_size = image.shape[0]
    length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
    text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
    

    if opt.Transformer == 'char-str':
        attens, preds = model(image, is_eval=True)
        pred_size = torch.IntTensor([preds.size(1)]*batch_size)
    # for index in range(image.shape[0]):
    index = 0

    _, t = preds.topk(1, dim=-1, largest=True, sorted=True)
    t = t.view(-1, opt.batch_max_length+2)
    t_s = converter.decode(t[:,0:], length_for_pred+2)
    t_p = F.softmax(preds, dim=2)
    t_p_max, _ = t_p.max(dim=2)
    t_p_max = t_p_max[:,0:]
    t_i = t[:,0:]
    t_pred = t_s[index]
    t_pred_max_prob = t_p_max[index]
    t_pred_index = t_i[index].cpu().tolist()
    print(t_pred_index)
    _, preds_index = preds.max(2)
    pred_str = converter.decode(preds_index.data, pred_size.data)

    preds_prob = F.softmax(preds, dim=2)
    preds_max_prob, _ = preds_prob.max(dim=2)
    # preds_max_prob = preds_max_prob[index,1:]
    try:
        char_confidence_score = preds_max_prob.cumprod(dim=0)[-1][0].item()
    except:
        char_confidence_score = 0.0
    print('char:', pred_str[index], char_confidence_score)

    # draw atten
    pil = transforms.ToPILImage()
    tensor = transforms.ToTensor()
    size = opt.imgH , opt.imgW
    resize = transforms.Resize(size=size, interpolation=0)
    # print(attens.shape)
    char_atten = attens[index]
    char_atten = char_atten[:,1:].view(-1, 8, 32) 
    # 각 어텐션 맵의 전체 최대값을 계산 (1차원과 2차원 동시 고려)
    char_atten_max = char_atten.view(char_atten.size(0), -1).max(dim=1)[0]

    # 최대값을 기준으로 내림차순 정렬
    _, top_indices = torch.sort(char_atten_max, descending=True)

    n = len(pred_str[index]) 
    top_n_indices = top_indices[:n]
    top_n_indices, _ = torch.sort(top_n_indices)  # index를 순서대로 정렬
    
    top_n_char_atten = char_atten[top_n_indices]
    char_atten = top_n_char_atten
    # print(f"Top {n} char_atten: {top_n_char_atten}")
    # char_atten = char_atten[1:len(pred_str[index])+1]
    # print(char_atten.shape) # (batch_max_length+2, imgH/patch_size, imgW/patch_size)
    # char_atten = char_atten.unsqueeze(0)  # [257] -> [1, 257]
    # char_atten = char_atten.unsqueeze(-1)  # [1, 257] -> [1, 257, 1]

    # 어텐션 맵 크기 조정
    # target_height = opt.imgH // 2
    # target_width = opt.imgW // 2
    # char_atten = F.interpolate(char_atten.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False).squeeze(0)
    # char_atten = (char_atten - char_atten.min()) / (char_atten.max() - char_atten.min())
    # if opt.imgW == 224:
    #     char_atten = char_atten[:, 1:].view(-1, 4, 28)
    # else:
    #     char_atten = char_atten[:,1:].view(-1, 8, 32)
    # char_atten = char_atten[1:char_pred_EOS+1]
    draw_atten(opt.demo_imgs, pred_str[index], char_atten, pil, tensor, resize, flag='char')

# class NormalizePAD(object):

#     def __init__(self, max_size, PAD_type='right'):
#         self.toTensor = transforms.ToTensor()
#         self.max_size = max_size
#         self.max_width_half = math.floor(max_size[2] / 2)
#         self.PAD_type = PAD_type

#     def __call__(self, img):
#         c, h, w = img.size()
#         Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
#         Pad_img[:, :, :w] = img  # right pad
#         return Pad_img

# def Pad(img, w=128, h=32):
#     input_channel = 1
#     transform = (NormalizePAD((input_channel, h, w)))
#     i_w, i_h = img.size
#     ratio = i_w / float(i_h)
#     if math.ceil(32 * ratio) > 128:
#             resized_w = 128
#     else:
#         resized_w = math.ceil(32 * ratio)

#     resized_image = img.resize((resized_w, 32), Image.BICUBIC)

#     return transform(resized_image)

def load_img(img_path, opt):
    img = Image.open(img_path).convert('L')
    # img = img.resize((opt.imgW, opt.imgH), Image.BICUBIC)
# 
    # img_arr = np.array(img)
    # img_tensor = transforms.ToTensor()(img)
    # img = np.array(img)
    # # Adaptive Thresholding
    # _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # Convert back to PIL Image
    # img = Image.fromarray(img)

    i_w, i_h = img.size
    ratio = i_w / float(i_h)
    
    if math.ceil(opt.imgH * ratio) > opt.imgW:
        resized_w = opt.imgW
    else:
        resized_w = math.ceil(opt.imgH * ratio)
    
    resized_image = img.resize((resized_w, opt.imgH), Image.BICUBIC)
    
    # Padding
    input_channel = 1
    max_size = (input_channel, opt.imgH, opt.imgW)
    Pad_img = torch.FloatTensor(*max_size).fill_(0)
    img_tensor = transforms.ToTensor()(resized_image)
    Pad_img[:, :, :resized_w] = img_tensor


    image_tensor = Pad_img.unsqueeze(0)
    return image_tensor




def draw_atten(img_path, pred, attn, pil, tensor, resize, flag=''):
    image = PIL.Image.open(img_path).convert('RGB')
    # image = cv2.resize(np.array(image), (128, 32))
    
    # image = tensor(image)
    # image_np = np.array(pil(image))

    i_w, i_h = image.size
    ratio = i_w / float(i_h)
    
    if math.ceil(opt.imgH * ratio) > opt.imgW:
        resized_w = opt.imgW
    else:
        resized_w = math.ceil(opt.imgH * ratio)
    
    resized_image = image.resize((resized_w, opt.imgH), Image.BICUBIC)
    
    # Padding
    input_channel = 3
    max_size = (input_channel, opt.imgH, opt.imgW)
    Pad_img = torch.FloatTensor(*max_size).fill_(0)
    img_tensor = tensor(resized_image)
    Pad_img[:, :, :resized_w] = img_tensor
    image = Pad_img
    image_np = np.array(pil(image))

    attn_pil = [pil(a) for a in attn[:, None, :, :]]
    attn = [tensor(resize(a)).repeat(3, 1, 1) for a in attn_pil]
    attn_sum = np.array([np.array(a) for a in attn_pil[:len(pred)]]).sum(axis=0)
    blended_sum = tensor(blend_mask(image_np, attn_sum))
    blended = [tensor(blend_mask(image_np, np.array(a))) for a in attn_pil]
    save_image = torch.stack([image] + attn + [blended_sum] + blended)
    save_image = save_image.view(2, -1, *save_image.shape[1:])
    save_image = save_image.permute(1, 0, 2, 3, 4).flatten(0, 1)
    
    gt = os.path.basename(img_path).split('.')[0]
    vutils.save_image(save_image, f'demo_imgs/attens/{gt}_{pred}_{flag}.jpg', nrow=2, normalize=True, scale_each=True)

def blend_mask(image, mask, alpha=0.5, cmap='jet', color='b', color_alpha=1.0):
    # normalize mask
    mask = (mask-mask.min()) / (mask.max() - mask.min() + np.finfo(float).eps)
    if mask.shape != image.shape:
        mask = cv2.resize(mask,(image.shape[1], image.shape[0]))
    # get color map
    color_map = plt.get_cmap(cmap)
    mask = color_map(mask)[:,:,:3]
    # convert float to uint8
    mask = (mask * 255).astype(dtype=np.uint8)

    # set the basic color
    basic_color = np.array(colors.to_rgb(color)) * 255 
    basic_color = np.tile(basic_color, [image.shape[0], image.shape[1], 1]) 
    basic_color = basic_color.astype(dtype=np.uint8)
    # blend with basic color
    blended_img = cv2.addWeighted(image, color_alpha, basic_color, 1-color_alpha, 0)
    # blend with mask
    blended_img = cv2.addWeighted(blended_img, alpha, mask, 1-alpha, 0)

    return blended_img


def test(opt):
    """ model configuration """
    # converter = TokenLabelConverter(opt)
    converter = CTCLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))

    # load img
    if os.path.isdir(opt.demo_imgs):
        imgs = [os.path.join(opt.demo_imgs, fname) for fname in os.listdir(opt.demo_imgs)]
        imgs = [img for img in imgs if img.endswith('.jpg') or img.endswith('.png')]
    else:
        imgs = [opt.demo_imgs]
    
    for img in imgs:
        opt.demo_imgs = img
        img_tensor = load_img(opt.demo_imgs, opt)
        print('imgs:', img)

        """ evaluation """
        model.eval()
        opt.eval = True
        with torch.no_grad():
            run_model(img_tensor, model, converter, opt)
        print('============================================================================')


if __name__ == '__main__':
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    # if opt.sensitive:
        # opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).
    import pickle
    if 'pkl' in opt.character:
        with open(opt.character, 'rb') as f:
            extended_char = pickle.load(f)
        ## '金','整','公','簿' 추후 추가
        extended_char.extend(['±',' ','△','※','☑','☐','⓪','①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩','⑪','⑫','⑬','⑭','⑮','⑯','⑰','⑱','⑲','⑳','@'])
        opt.character = ''.join(extended_char)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    opt.saved_model = opt.model_dir
    test(opt)
