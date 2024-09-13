'''
Implementation of MGP-STR based on ViTSTR.

Copyright 2022 Alibaba
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch 
import torch.nn as nn
import logging
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

from copy import deepcopy
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models import create_model
from .token_learner import TokenLearner

_logger = logging.getLogger(__name__)

__all__ = [
    'char_str_base_patch4_3_32_128',
    'char_str_large_patch8_1_32_224'
]

def create_char_str(batch_max_length, num_tokens, model=None, checkpoint_path=''):
    char_str = create_model(
        model,
        pretrained=True,
        num_classes=num_tokens,
        checkpoint_path=checkpoint_path,
        batch_max_length=batch_max_length)

    # might need to run to get zero init head for transfer learning
    char_str.reset_classifier(num_classes=num_tokens)

    return char_str
    
class CHARSTR(VisionTransformer):
    def __init__(self, batch_max_length, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.batch_max_length = batch_max_length
        self.char_tokenLearner = TokenLearner(self.embed_dim, self.batch_max_length)
        self.num_classes = num_classes
        self.char_head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        char_attn, char_x = self.char_tokenLearner(x)
        char_x = self.char_head(char_x)
        char_out = F.log_softmax(char_x, dim=-1)
        
        return char_attn, char_out

    def forward(self, x, is_eval=False):
        char_attn, char_out = self.forward_features(x)
        if is_eval:
            return [char_attn, char_out]
        else:
            return char_out

def load_pretrained(model, cfg=None, num_classes=1000, in_chans=1, filter_fn=None, strict=True):
    '''
    Loads a pretrained checkpoint
    From an older version of timm
    '''
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    state_dict = model_zoo.load_url(cfg['url'], progress=True, map_location='cpu')
    if "model" in state_dict.keys():
        state_dict = state_dict["model"]

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        key = conv1_name + '.weight'
        if key in state_dict.keys():
            _logger.info('(%s) key found in state_dict' % key)
            conv1_weight = state_dict[conv1_name + '.weight']
        else:
            _logger.info('(%s) key NOT found in state_dict' % key)
            return
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    print("Loading pre-trained vision transformer weights from %s ..." % cfg['url'])
    model.load_state_dict(state_dict, strict=strict)

def _conv_filter(state_dict):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if not 'patch_embed' in k and  not 'pos_embed' in k :
            out_dict[k] = v
        else:
            print("not load",k) 
    return out_dict

@register_model
def char_str_base_patch4_3_32_128(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = CHARSTR(
        img_size=(32,128), patch_size=4, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            #url='https://github.com/roatienza/public/releases/download/v0.1-deit-base/deit_base_patch16_224-b5f2ef4d.pth'
            url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

@register_model
def char_str_large_patch8_1_32_224(pretrained=False, **kwargs):
    kwargs['in_chans'] = 1
    model = CHARSTR(
        img_size=(32,224), patch_size=8, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    model.default_cfg = _cfg(
            #url='https://github.com/roatienza/public/releases/download/v0.1-deit-base/deit_base_patch16_224-b5f2ef4d.pth'
            url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth'
    )
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model