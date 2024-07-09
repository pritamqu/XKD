# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary  import summary
from einops.layers.torch import Rearrange
import math
from timm.models.layers import trunc_normal_
try:
    from .vit2 import ViT_Backbone
except:
    from vit2 import ViT_Backbone
    
encoder_dict = {
    'tiny_encoder' : {'embed_dim':192, 'depth':12, 'num_heads':3},
    'tiny_encoder_h6' : {'embed_dim':192, 'depth':12, 'num_heads':6},
    'small_encoder' : {'embed_dim':384, 'depth':12, 'num_heads':6},
    'small_encoder_h12' : {'embed_dim':384, 'depth':12, 'num_heads':12},
    'base_encoder' : {'embed_dim':768, 'depth':12, 'num_heads':12},
    'large_encoder' : {'embed_dim':1024, 'depth':24, 'num_heads':16},
    }


decoder_dict = {
    'large_decoder' : {'embed_dim':512, 'depth':8, 'num_heads':16},
    'base_decoder' : {'embed_dim':384, 'depth':4, 'num_heads':12},
    'base_decoder_d1' : {'embed_dim':384, 'depth':1, 'num_heads':12},
    'small_decoder' : {'embed_dim':192, 'depth':4, 'num_heads':6},
    'small_decoder_d1' : {'embed_dim':192, 'depth':1, 'num_heads':6},
    'tiny_decoder' : {'embed_dim':96, 'depth':4, 'num_heads':3},   
    'tiny_decoder_d1' : {'embed_dim':96, 'depth':1, 'num_heads':3},   

    }

class Cuboids(nn.Module):
    """
    the frame sequences are transformed into smaller 3d cuboids;
    3d cuboids are directly projected to linear embeddings;
    op dimension: batch size x number of cuboids x embed dimension
    """
    def __init__(self, embedding_dim, tubelet_t, tubelet_h, tubelet_w, in_channels):
        super(Cuboids, self).__init__()
        tubelet_dim = in_channels * tubelet_h * tubelet_w * tubelet_t
        self.patches = Rearrange("b  c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)",
                pt=tubelet_t,
                ph=tubelet_h,
                pw=tubelet_w,
            )
        self.proj = nn.Linear(tubelet_dim, embedding_dim)

    def forward(self, x, return_embeds=False):
        patches = self.patches(x)
        embeddings = self.proj(patches)
        if return_embeds:
            return embeddings
        return patches, embeddings

class PosEmbedding(nn.Module):
    # copied from https://github.com/SforAiDl/vformer/blob/main/vformer/encoder/embedding/pos_embedding.py#L77

    def __init__(self, shape, dim, drop=None, sinusoidal=False, std=0.02):
        super(PosEmbedding, self).__init__()

        if not sinusoidal:
            if isinstance(shape, int):
                shape = [1, shape, dim]
            else:
                shape = [1] + list(shape) + [dim]
            self.pos_embed = nn.Parameter(torch.zeros(shape))

        else:
            pe = torch.FloatTensor(
                [
                    [p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)]
                    for p in range(shape)
                ]
            )
            pe[:, 0::2] = torch.sin(pe[:, 0::2])
            pe[:, 1::2] = torch.cos(pe[:, 1::2])
            self.pos_embed = pe
            self.pos_embed.requires_grad = False
        trunc_normal_(self.pos_embed, std=std)
        self.pos_drop = nn.Dropout(drop) if drop is not None else nn.Identity()

    def forward(self, x, cls_token=False):
        if cls_token:
            x + self.pos_embed[:, 1:, :]
        else:
            x = x + self.pos_embed
        return self.pos_drop(x)


class VideoViT(nn.Module):
    def __init__(self,
                 frame_size = (3, 224, 224),
                 num_frames = 8,
                 patch_spatial = (16, 16),
                 patch_temporal = 2,
                 encoder_dim = 1024,
                 encoder = None,
                 apply_cls_token=True,
                 norm_layer=nn.LayerNorm,
                 num_classes=0,
                 ):
        super().__init__()

        self.frame_size = frame_size
        self.num_frames = num_frames
        self.patch_temporal = patch_temporal
        self.patch_spatial = patch_spatial
        self.encoder_dim = encoder_dim
        self.apply_cls_token = apply_cls_token
        self.patch_dim = patch_spatial[0] * patch_spatial[1] * patch_temporal * frame_size[0]
        self.num_cuboids = (frame_size[1]//patch_spatial[0]) * (frame_size[2]//patch_spatial[1]) * (num_frames//patch_temporal) # number of smaller cuboids
        self.cuboid_embed = Cuboids(embedding_dim=encoder_dim,
                                         tubelet_t=patch_temporal,
                                         tubelet_h=patch_spatial[1],
                                         tubelet_w=patch_spatial[0],
                                         in_channels=frame_size[0])

        if self.apply_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
            self.enc_pos_embed = PosEmbedding(self.num_cuboids+1, encoder_dim).pos_embed            
        else:
            self.enc_pos_embed = PosEmbedding(self.num_cuboids, encoder_dim).pos_embed

        self.encoder = encoder
        self.encoder_norm = norm_layer(encoder_dim)
        self.head = nn.Linear(self.encoder_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize cuboid_embed like nn.Linear following MAE by K. HE
        w = self.cuboid_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if self.apply_cls_token:
            torch.nn.init.normal_(self.cls_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'enc_pos_embed', 'cls_token'}
            
    def prepare_token(self, x):
               
        B, nc, t, w, h = x.shape
        x = self.cuboid_embed(x, return_embeds=True)
        pos_embed = self.interpolate_pos_encoding(x, t, w, h)
        
        # add pos embed w/o cls token
        if self.apply_cls_token:
            x = x + pos_embed[:, 1:, :]
        else:
            x = x + pos_embed
            
        # append cls token
        if self.apply_cls_token:
            cls_token = self.cls_token + pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        return x

    def interpolate_pos_encoding(self, x, t, w, h):
        
        npatch = x.shape[1]
        if self.apply_cls_token:
            N = self.enc_pos_embed.shape[1] - 1
        else:
            N = self.enc_pos_embed.shape[1]
            
        Ns = N//self.patch_temporal
        Nt = self.patch_temporal
        if npatch == N and w == h:
            return self.enc_pos_embed

        if self.apply_cls_token:
            class_pos_embed = self.enc_pos_embed[:, 0]
            patch_pos_embed = self.enc_pos_embed[:, 1:]
        else:
            patch_pos_embed = self.enc_pos_embed
        
        dim = x.shape[-1]
        w0 = w // self.patch_spatial[0]
        h0 = h // self.patch_spatial[1]
        t0 = t // self.patch_temporal
        w0, h0, t0 = w0 + 0.1, h0 + 0.1, t0 + 0.1
        # FIXME: debug
        # print(patch_pos_embed.shape, '->', (1, (Nt), (math.sqrt(Ns)), (math.sqrt(Ns)), dim))
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, 
                                    int(Nt),
                                    int(math.sqrt(Ns)), 
                                    int(math.sqrt(Ns)), 
                                    dim).permute(0, 4, 1, 2, 3),
            scale_factor=(t0 / Nt, w0 / math.sqrt(Ns), h0 / math.sqrt(Ns)),
            mode='trilinear',
        ) # for videos
        assert int(t0) == patch_pos_embed.shape[-3] and int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1] 
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
        if self.apply_cls_token:
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        else:
            return patch_pos_embed

    def forward(self, x, **kwargs):
        # simply extract feature for downstream task
        x = self.prepare_token(x)
        # apply Transformer blocks
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x

    def forward_last_layer_attn(self, x, **kwargs):
        x = self.prepare_token(x)
        attn, x = self.encoder.get_last_selfattention(x)
        x = self.encoder_norm(x)
        return attn, x


def vid_vit(
            frame_size= (3, 224, 224),
            num_frames=16,
            patch_spatial=(16, 16),
            patch_temporal=2,
            apply_cls_token=True,
            encoder_cfg='base_encoder',
            norm_layer=nn.LayerNorm,
            num_classes=0,
        ):
    
    encoder=ViT_Backbone(**encoder_dict[encoder_cfg], mlp_ratio=4)
    
    model = VideoViT(
                 frame_size = frame_size,
                 num_frames = num_frames,
                 patch_spatial = patch_spatial,
                 patch_temporal = patch_temporal,
                 encoder_dim = encoder_dict[encoder_cfg]['embed_dim'],
                 encoder = encoder,
                 apply_cls_token=apply_cls_token,
                 norm_layer=norm_layer,
                 num_classes=num_classes,
                 )
    
    return model



# if __name__=='__main__':

       
#     model = vid_vit(frame_size = (3, 224, 224),
#                             num_frames = 16,
#                             patch_spatial = (16, 16),
#                             patch_temporal = 2,
#                             encoder_cfg = 'tiny_encoder',
#                             apply_cls_token=True,
#                             )
    
#     frames0 = torch.randn(2, 3, 16, 224, 224)
#     feat0 = model(frames0)
    