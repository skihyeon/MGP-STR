'''
Implementation of A3 module based on TokenLearner.

Copyright 2022 Alibaba
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLearner(nn.Module):
    
    def __init__(self, input_embed_dim, out_token=30):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner = nn.Sequential(nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size = (1,1), stride=1, groups=8, bias=False),
                                          nn.Conv2d(input_embed_dim, out_token, kernel_size = (1,1), stride=1, bias=False))
        self.feat = nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size = (1,1), stride=1, groups=8, bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.token_norm(x) # [bs, 257, 768]
        x = x.transpose(1, 2).unsqueeze(-1) # [bs, 768, 257, 1]
        selected = self.tokenLearner(x) # [bs, 27, 257, 1].
        selected = selected.flatten(2)  # [bs, 27, 257].
        selected = F.softmax(selected, dim=-1) 
        feat = self.feat(x) #  [bs, 768, 257, 1].
        feat = feat.flatten(2).transpose(1,2)  # [bs, 257, 768]
        x = torch.einsum('...si,...id->...sd', selected, feat) # [bs, 27, 768]
        
        x = self.norm(x)
        return selected, x
    
# class TokenLearner(nn.Module):
#     def __init__(self, input_embed_dim, out_token):
#         super().__init__()
#         self.token_norm = nn.LayerNorm(input_embed_dim)
#         self.tokenLearner = nn.Sequential(
#             nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False),
#             nn.ReLU(),
#             nn.Conv2d(input_embed_dim, out_token, kernel_size=(1,1), stride=1, bias=False)
#         )
#         self.feat = nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False)
#         self.norm = nn.LayerNorm(input_embed_dim)

#     def forward(self, x):
#         B = x.shape[0]
#         x = self.token_norm(x)
#         x = x.transpose(1, 2).unsqueeze(-1)
#         selected = self.tokenLearner(x)
#         selected = selected.flatten(2)
#         selected = F.softmax(selected, dim=-1)
#         feat = self.feat(x)
#         feat = feat.flatten(2).transpose(1,2)
#         x = torch.einsum('...si,...id->...sd', selected, feat)
#         x = self.norm(x)
#         return selected, x
    




# 글자간 경계 학습 레이어 추가
class ITokenLearner(nn.Module):
    def __init__(self, input_embed_dim, out_token):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner = nn.Sequential(
            nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False),
            nn.ReLU(),
            nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False),
            nn.ReLU(),
            nn.Conv2d(input_embed_dim, out_token, kernel_size=(1,1), stride=1, bias=False)
        )
        self.feat = nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)
        
        # 글자 경계 인식을 위한 추가 레이어
        self.boundary_detector = nn.Sequential(
            nn.Conv2d(input_embed_dim, input_embed_dim // 4, kernel_size=(1,1), stride=1),
            nn.ReLU(),
            nn.Conv2d(input_embed_dim // 4, 1, kernel_size=(1,1), stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.token_norm(x)
        x = x.transpose(1, 2).unsqueeze(-1)
        
        # 글자 경계 감지
        boundaries = self.boundary_detector(x)
        
        selected = self.tokenLearner(x)
        selected = selected.flatten(2)
        selected = F.softmax(selected, dim=-1)
        
        # 글자 경계 정보를 토큰 선택에 반영
        selected = selected * boundaries.flatten(2)
        selected = F.normalize(selected, p=1, dim=-1)
        
        feat = self.feat(x)
        feat = feat.flatten(2).transpose(1,2)
        x = torch.einsum('...si,...id->...sd', selected, feat)
        x = self.norm(x)
        return selected, x
    

# 어텐션 추가
class AttentionTokenLearner(nn.Module):
    def __init__(self, input_embed_dim, out_token):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner = nn.Sequential(
            nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False),
            nn.ReLU(),
            nn.Conv2d(input_embed_dim, out_token, kernel_size=(1,1), stride=1, bias=False)
        )
        self.feat = nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)
        
        # 어텐션 레이어 추가
        self.attention = nn.MultiheadAttention(input_embed_dim, num_heads=8, batch_first=True)

    def forward(self, x):
        B = x.shape[0]
        x = self.token_norm(x)
        x_2d = x.transpose(1, 2).unsqueeze(-1)
        
        selected = self.tokenLearner(x_2d)
        selected = selected.flatten(2)
        selected = F.softmax(selected, dim=-1)
        
        feat = self.feat(x_2d)
        feat = feat.flatten(2).transpose(1,2)
        
        # 어텐션 적용
        x, _ = self.attention(feat, feat, feat)
        
        x = torch.einsum('...si,...id->...sd', selected, x)
        x = self.norm(x)
        return selected, x
    

# 동적 토큰 수 조정
class DynamicTokenLearner(nn.Module):
    def __init__(self, input_embed_dim, max_tokens):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner = nn.Sequential(
            nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False),
            nn.ReLU(),
            nn.Conv2d(input_embed_dim, max_tokens, kernel_size=(1,1), stride=1, bias=False)
        )
        self.feat = nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)
        
        # 토큰 수 예측기
        self.token_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.token_norm(x)
        x_2d = x.transpose(1, 2).unsqueeze(-1)
        
        # 동적으로 토큰 수 결정
        token_ratio = self.token_predictor(x_2d)
        num_tokens = int(token_ratio.item() * self.tokenLearner[-1].out_channels)
        
        selected = self.tokenLearner(x_2d)
        selected = selected[:, :num_tokens, :]
        selected = selected.flatten(2)
        selected = F.softmax(selected, dim=-1)
        
        feat = self.feat(x_2d)
        feat = feat.flatten(2).transpose(1,2)
        x = torch.einsum('...si,...id->...sd', selected, feat)
        x = self.norm(x)
        return selected, x



# 계층적 토큰 학습
class HierarchicalTokenLearner(nn.Module):
    def __init__(self, input_embed_dim, out_token):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner1 = nn.Sequential(
            nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False),
            nn.ReLU(),
            nn.Conv2d(input_embed_dim, out_token * 2, kernel_size=(1,1), stride=1, bias=False)
        )
        self.tokenLearner2 = nn.Sequential(
            nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False),
            nn.ReLU(),
            nn.Conv2d(input_embed_dim, out_token, kernel_size=(1,1), stride=1, bias=False)
        )
        self.feat = nn.Conv2d(input_embed_dim, input_embed_dim, kernel_size=(1,1), stride=1, groups=16, bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x = self.token_norm(x)
        x_2d = x.transpose(1, 2).unsqueeze(-1)
        
        # 첫 번째 레벨 토큰 학습
        selected1 = self.tokenLearner1(x_2d)
        selected1 = selected1.flatten(2)
        selected1 = F.softmax(selected1, dim=-1)
        
        feat = self.feat(x_2d)
        feat = feat.flatten(2).transpose(1,2)
        x1 = torch.einsum('...si,...id->...sd', selected1, feat)
        
        # 두 번째 레벨 토큰 학습
        x1_2d = x1.unsqueeze(-1)
        selected2 = self.tokenLearner2(x1_2d)
        selected2 = selected2.flatten(2)
        selected2 = F.softmax(selected2, dim=-1)
        
        x = torch.einsum('...si,...id->...sd', selected2, x1)
        x = self.norm(x)
        return selected2, x













import math

class CTCTokenLearner(nn.Module):
    def __init__(self, input_embed_dim, out_token=30):
        super().__init__()
        self.token_norm = nn.LayerNorm(input_embed_dim)
        self.tokenLearner = nn.Sequential(
            nn.Conv1d(input_embed_dim, input_embed_dim, kernel_size=3, padding=1, groups=8, bias=False),
            nn.ReLU(),
            nn.Conv1d(input_embed_dim, out_token, kernel_size=1, bias=False)
        )
        self.feat = nn.Conv1d(input_embed_dim, input_embed_dim, kernel_size=1, groups=8, bias=False)
        self.norm = nn.LayerNorm(input_embed_dim)
        self.pos_encoder = PositionalEncoding(input_embed_dim, max_len=1000)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.token_norm(x)  # [bs, 257, 768]
        x = x.transpose(1, 2)  # [bs, 768, 257]
        
        # 위치 인코딩 추가
        x = self.pos_encoder(x.transpose(1, 2)).transpose(1, 2)
        
        selected = self.tokenLearner(x)  # [bs, out_token, 257]
        selected = F.softmax(selected, dim=2)
        
        feat = self.feat(x)  # [bs, 768, 257]
        x = torch.bmm(feat, selected.transpose(1, 2))  # [bs, 768, out_token]
        x = x.transpose(1, 2)  # [bs, out_token, 768]
        
        x = self.norm(x)
        return selected.transpose(1, 2), x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]



