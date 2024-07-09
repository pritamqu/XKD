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
    from .vit2 import Block
except:
    from vit2 import Block
    
def return_vit_config(vit_config):
    # return  embed_dim, depth, num_heads
    if vit_config=='tiny':
        return 192, 12, 3
    elif vit_config=='small':
        return 384, 12, 6
    elif vit_config=='base':
        return 768, 12, 12
    elif vit_config=='large':
        return 1024, 24, 16
    else:
        raise NotImplementedError(f'vit_config: {vit_config} is not available')
    

class SpecPatches(nn.Module):
    """
    the specs are transformed into smaller 2d patches;
    2d patches are then projected to linear embeddings;
    op dimension: batch size x number of 2d patches x embed dimension
    """
    def __init__(self, embedding_dim, patch_spatial, in_channels):
        super(SpecPatches, self).__init__()
        _dim = patch_spatial[0] * patch_spatial[1] * in_channels
        self.patches = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)",
                ph=patch_spatial[0],
                pw=patch_spatial[1],
                c=in_channels,
            )
        self.proj = nn.Linear(_dim, embedding_dim)

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


class AudioViT(nn.Module):
    def __init__(self,
                 spec_size = (80, 224),
                 patch_spatial = (5, 16),
                 apply_cls_token=True,
                 vit_config='base',
                 mlp_ratio=4,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, num_classes=0,
                 **kwargs
                 ):
        super().__init__()
                
        assert len(spec_size)==2, " spec_size = (freq height x time width)"
        assert spec_size[0] % patch_spatial[0]==0 and  spec_size[1] % patch_spatial[1]==0, "spec_size dims should be divisible by patch_spatial dims"

        embed_dim, depth, num_heads = return_vit_config(vit_config) 

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.spec_size = spec_size
        self.patch_spatial = patch_spatial
        self.embed_dim = embed_dim
        self.apply_cls_token = apply_cls_token
        self.patch_dim = patch_spatial[0] * patch_spatial[1]
        self.num_patches = (spec_size[1]//patch_spatial[1]) * (spec_size[0]//patch_spatial[0])
        
        self.patch_embed = SpecPatches(embedding_dim=embed_dim,
                                         patch_spatial=patch_spatial,
                                         in_channels=1)

        if self.apply_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = PosEmbedding(self.num_patches+1, embed_dim).pos_embed
        else:
            self.pos_embed = PosEmbedding(self.num_patches, embed_dim).pos_embed

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()       
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize patch_embed like nn.Linear following MAE by K. HE
        w = self.patch_embed.proj.weight.data
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
        return {'pos_embed', 'cls_token'}

    def prepare_token(self, x):

        B, nc, w, h = x.shape
        x = self.patch_embed(x, return_embeds=True)
        pos_embed = self.interpolate_pos_encoding(x, w, h)

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
    
    def interpolate_pos_encoding(self, x, w, h):
        
        npatch = x.shape[1]
        if self.apply_cls_token:
            N = self.pos_embed.shape[1] - 1
        else:
            N = self.pos_embed.shape[1]
            
        if npatch == N: # and w == h: this is not applicable for spec as w!=h
            return self.pos_embed
        
        assert w == self.spec_size[0], f'mel-spectrogram freq side should remain same; should be {self.spec_size[0]} but got {w}'        

        if self.apply_cls_token:
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
        else:
            patch_pos_embed = self.pos_embed
            
        dim = x.shape[-1]
        # w0 = w // self.patch_embed.patch_spatial[0]
        h0 = h // self.patch_spatial[1]
        
        Nw = self.spec_size[0]//self.patch_spatial[0]
        Nh = N//Nw
        h0 = h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(Nw), int(Nh), dim).permute(0, 3, 1, 2),
            scale_factor=(1, h0 / Nh),
            mode='bicubic', align_corners=False,
        )
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        if self.apply_cls_token:
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        else:
            return patch_pos_embed

    def forward_features(self, x, **kwargs):
        """ pass features w/o head """
        # simply extract feature for downstream task
        x = self.prepare_token(x)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, feat_op, **kwargs):
        """ fwd pass through head. ability to choose which feature to pass """
        assert feat_op in ['pool', 'cls']
        # simply extract feature for downstream task
        x = self.prepare_token(x)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        if feat_op =='pool':
            if self.apply_cls_token:
                x = x[:, 1:, :].mean(dim=1)
            else:
                x = x.mean(dim=1)
        elif self.feat_op =='cls':
            assert self.apply_cls_token
            x = x[:, 0]
        else:
            raise ValueError(f'feat_op should be either pool or cls; given {feat_op}')
            
        x = self.head(x)
        return x
    
    def get_last_selfattention(self, x):
        x = self.prepare_token(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        # we return the output tokens from the `n` last blocks
        output = []
        x = self.prepare_token(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
    

# def aud_vit(
#             spec_size,
#             patch_spatial,
#             apply_cls_token,
#             encoder_cfg,
#             norm_layer=nn.LayerNorm,
#             num_classes=0,
#         ):
    
#     encoder=ViT_Backbone(**encoder_dict[encoder_cfg], mlp_ratio=4)
    
#     model = AudioViT(
#                  spec_size = spec_size,
#                  patch_spatial = patch_spatial,
#                  encoder_dim = encoder_dict[encoder_cfg]['embed_dim'],
#                  encoder = encoder,
#                  apply_cls_token=apply_cls_token,
#                  norm_layer=norm_layer,
#                  num_classes=num_classes,
#                  )
#     return model



# if __name__=='__main__':

#     model = aud_vit(spec_size = (80, 224),
#                     patch_spatial = (5, 16),
#                     encoder_cfg = 'base_encoder',
#                     apply_cls_token=True,
#                     )
    
#     specs0 = torch.randn(2, 1, 80, 224)
#     feat0 = model(specs0)
    