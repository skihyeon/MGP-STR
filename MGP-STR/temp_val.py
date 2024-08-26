from dataclasses import dataclass
import torch
from typing import List
import pickle

import os
import time
import string
import argparse
import re
import PIL

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from PIL import Image
import torch.nn.functional as F
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from utils import  TokenLabelConverter
from dataset import hierarchical_dataset, AlignCollate, ImgDataset
from models import Model
from utils import get_args

import cv2
@dataclass
class Opt:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_class: int = None
    character: str = None
    language: str = "kor"
    ign_char_idx: List[int] = None
    canvas_size: str = "medium"
    response_type: str = "basic"
    orientation: bool = False
    workers: int = 4
    batch_size: int = 100
    batch_max_length: int = 30
    imgH: int = 32
    imgW: int = 320
    rgb: bool = False
    sensitive: bool = False
    PAD: bool = True
    Transformation: str = "TPS"
    FeatureExtraction: str = "ResNet"
    SequenceModeling: str = "BiLSTM"
    Prediction: str = "CTC"
    num_fiducial: int = 20
    input_channel: int = 1
    output_channel: int = 512
    hidden_size: int = 256
    Transformer: str = 'mgp-str'
    TransformerModel: str = 'mgp_str_base_patch4_3_32_128'

    @classmethod
    def read_pkl(cls, path: str):
        with open(path, 'rb') as f:
            char = pickle.load(f)
        char.extend([' ', '(', ')'])
        return len(char), char

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__annotations__:
                setattr(self, k, v)
        
        self.batch_max_length = 100
        self.num_class, self.character = self.read_pkl('korean_dict.pkl')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.imgH, self.imgW = 32, 128
        self.input_channel = 3
        self.output_channel = 512
        self.hidden_size = 256
        self.num_fiducial = 20
        self.num_class = len(self.character) + 2
        self.ign_char_idx = None
        self.PAD = True
        self.batch_size = 200
        self.workers = 4 
        self.rgb = True
        self.eval= True
        self.eval_img = False

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None
    
def load_images_from_path(files, type="pillow"):
    # files = sorted([x for x in os.listdir(folder_path) if x.endswith(('png', 'jpeg', 'jpg'))])
    img_list = []
    for fname in files:
        print("fname :", fname)
        # img_path = os.path.join(folder_path, fname)
        if type == "pillow":
            img = np.array(Image.open(fname).convert('RGB'))
        elif type == "cv":
            img = imread(fname)
        img_list.append(img)
    return img_list, files


def extract_number(file_name):
    match = re.search(r'(\d+)', file_name)
    return int(match.group(1)) if match else -1

class MGPDataset(Dataset):
    def __init__(self, img_list):
        self.img_list = img_list
        self.nSamples = len(img_list)
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        img = Image.fromarray(self.img_list[index]).convert('RGB')
        return img, str(index)

import time

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def recog_mgp():
    opt = Opt()
    
    converter = TokenLabelConverter(opt)
    model = Model(opt)
    model.load_state_dict(copyStateDict(torch.load('/mnt/hdd1/sgh/MGP-STR/MGP-STR/model_files/mgp_str_base_patch4_3_32_128-Seed226_aug-Seed226/iter_370000.pth', map_location=opt.device)))
    model.eval()
    device = 'cuda'
    images_path = '/mnt/hdd1/seungchan/test_dataset/random_crop_text_10000'
    files = [x for x in os.listdir(images_path) if x.endswith('png') or x.endswith('jpeg') or x.endswith('jpg')]
    sorted_files = sorted(files, key=lambda x: extract_number(x))

    sorted_files = [os.path.join(images_path, i) for i in sorted_files]
    img_list, file_names = load_images_from_path(sorted_files, type="cv")

    def collate_fn(batch):
        images, indices = zip(*batch)
        images = torch.stack(images)
        return images, indices
    
    def img_list_prediction_mgpstr(img_list, model, opt):
        model.to(device)
        model.eval()
        converter = TokenLabelConverter(opt)

        dataset = MGPDataset(img_list)
        AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD, opt=opt)

        data_loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_demo,
            pin_memory=True
        )

        out_result = []
        confidence_score_list = []

        start_time = time.time()
        with torch.no_grad():
            for batch in data_loader:
                images, _, _ = batch
                images = images.to(device)
                batch_size = images.size(0)

                attens, char_preds, bpe_preds, wp_preds = model(images, is_eval=True)
                _, char_pred_index = char_preds.topk(1, dim=-1, largest=True, sorted=True)
                char_pred_index = char_pred_index.view(-1, converter.batch_max_length)
                length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
                char_preds_str = converter.char_decode(char_pred_index[:, 1:], length_for_pred)
                char_pred_prob = F.softmax(char_preds, dim=2)
                char_pred_max_prob, _ = char_pred_prob.max(dim=2)
                char_preds_max_prob = char_pred_max_prob[:, 1:]

                for i in range(batch_size):
                    char_pred = char_preds_str[i]
                    char_pred_max_prob = char_preds_max_prob[i]
                    char_pred_EOS = char_pred.find('[s]')
                    char_pred = char_pred[:char_pred_EOS]
                    if '0' in char_pred and ')' in char_pred and '(' not in char_pred[:char_pred.find(')')]:
                        char_pred = char_pred.replace('0', '(', 1)
                    char_pred_max_prob = char_pred_max_prob[:char_pred_EOS+1]
                    try:
                        char_confidence_score = char_pred_max_prob.cumprod(dim=0)[-1].cpu().tolist()
                    except:
                        char_confidence_score = 0.0
                    out_result.append(char_pred)
                    confidence_score_list.append(char_confidence_score)

        end_time = time.time()
        print(f"Time taken: {(end_time - start_time) * 1000} ms")

        return out_result, confidence_score_list
    preds, _  = img_list_prediction_mgpstr(img_list, model, opt)
    with open('prediction.txt', 'w') as f:
        for i, pred in enumerate(preds):
            f.write(f'{i} {pred}\n')
    print(preds)

if __name__ == "__main__":
    recog_mgp()