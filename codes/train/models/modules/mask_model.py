import torch
import torch.nn as nn
import math
try:
    from .vit_video import VideoViT, PosEmbedding, ViT_Backbone, encoder_dict, decoder_dict
except:
    from vit_video import VideoViT, PosEmbedding, ViT_Backbone, encoder_dict, decoder_dict
    
class VideoMAEWrapper(nn.Module):
    def __init__(self,
                 # frame_size = (3, 224, 224),
                 # num_frames = 8,
                 # patch_spatial = (16, 16),
                 # patch_temporal = 2,
                 encoder_dim = 1024,
                 decoder_dim = 512,
                 encoder = None,
                 decoder = None,
                 masking_fn=None,
                 # apply_cls_token=True,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=True):
        super().__init__()

        self.frame_size = encoder.frame_size
        self.num_frames = encoder.num_frames
        self.patch_temporal = encoder.patch_temporal
        self.patch_spatial = encoder.patch_spatial
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.apply_cls_token = encoder.apply_cls_token
        self.masking_fn = masking_fn
        self.patch_dim = encoder.patch_dim
        self.num_cuboids = encoder.num_cuboids # number of smaller cuboids

        if self.apply_cls_token:
            self.dec_pos_embed = PosEmbedding(self.num_cuboids+1, decoder_dim).pos_embed
        else:
            self.dec_pos_embed = PosEmbedding(self.num_cuboids, decoder_dim).pos_embed

        self.encoder = encoder
        self.decoder = decoder
        self.mid_proj = nn.Linear(encoder_dim, decoder_dim, bias=True) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.out_proj = nn.Linear(decoder_dim, self.patch_dim) if decoder_dim != self.patch_dim else nn.Identity()

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        torch.nn.init.normal_(self.mask_token, std=.02)
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
            
    def prepare_token(self, x, mask_ratio=None, **kwargs):
               
        B, nc, t, w, h = x.shape
        target, x = self.encoder.cuboid_embed(x)
        pos_embed = self.encoder.interpolate_pos_encoding(x, t, w, h)
        
        # add pos embed w/o cls token
        if self.apply_cls_token:
            x = x + pos_embed[:, 1:, :]
        else:
            x = x + pos_embed
        
        # apply mask
        x, mask, ids_restore = self.masking_fn(x, mask_ratio, **kwargs)
            
        # append cls token
        if self.apply_cls_token:
            cls_token = self.encoder.cls_token + pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        return x, target, mask, ids_restore
    
    def interpolate_pos_encoding_decoder(self, x, t, w, h):
        
        # npatch = x.shape[1]
        # here x is already added with cls_token
        if self.apply_cls_token:
            npatch = x.shape[1] - 1
            N = self.dec_pos_embed.shape[1] - 1
        else:
            npatch = x.shape[1]
            N = self.dec_pos_embed.shape[1]
            
        Ns = N//self.patch_temporal
        Nt = self.patch_temporal
        if npatch == N and w == h:
            return self.dec_pos_embed
    
        if self.apply_cls_token:
            class_pos_embed = self.dec_pos_embed[:, 0]
            patch_pos_embed = self.dec_pos_embed[:, 1:]
        else:
            patch_pos_embed = self.dec_pos_embed
        
        dim = x.shape[-1]
        w0 = w // self.patch_spatial[0]
        h0 = h // self.patch_spatial[1]
        t0 = t // self.patch_temporal
        
        w0, h0, t0 = w0 + 0.1, h0 + 0.1, t0 + 0.1
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
        
    def forward_encoder(self, x, mask_ratio, **kwargs):
        x, target, mask, ids_restore = self.prepare_token(x, mask_ratio, **kwargs)
        x = self.encoder.encoder(x) # calling ViT_Backbone
        x = self.encoder.encoder_norm(x)
        return x, target, mask, ids_restore

    def forward_decoder(self, x, t, w, h, ids_restore):
        # embed tokens
        x = self.mid_proj(x)
        # append mask tokens to sequence
        if self.apply_cls_token:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # tackling cls token # appending mask token with encoder op
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # appending mask token with encoder op

        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        if self.apply_cls_token:
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            x = x_

        # add pos embed to tackle size mismatch
        # x = x + self.dec_pos_embed
        # or
        pos_embed = self.interpolate_pos_encoding_decoder(x, t, w, h)
        x = x + pos_embed
        
        x = self.decoder(x) # apply Transformer blocks
        x = self.out_proj(x) # predictor projection
        if self.apply_cls_token:
            x = x[:, 1:, :] # remove cls token
        
        return x

    def forward_loss(self, target, pred, mask):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, frames, mask_ratio, return_latent=False, **kwargs):
        _, _, t, w, h = frames.shape
        latent, target, mask, ids_restore = self.forward_encoder(frames, mask_ratio, **kwargs)
        pred = self.forward_decoder(latent, t, w, h, ids_restore)
        loss = self.forward_loss(target, pred, mask)
        if return_latent:
            return latent, loss
        return loss


class AudioMAEWrapper(nn.Module):
    def __init__(self,
                 # frame_size = (3, 224, 224),
                 # num_frames = 8,
                 # patch_spatial = (16, 16),
                 # patch_temporal = 2,
                 encoder_dim = 1024,
                 decoder_dim = 512,
                 encoder = None,
                 decoder = None,
                 masking_fn=None,
                 # apply_cls_token=True,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=True):
        super().__init__()

        self.spec_size = encoder.spec_size
        self.patch_spatial = encoder.patch_spatial
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.apply_cls_token = encoder.apply_cls_token
        self.masking_fn = masking_fn
        self.patch_dim = encoder.patch_dim
        self.num_patches = encoder.num_patches # number of smaller patches

        if self.apply_cls_token:
            self.dec_pos_embed = PosEmbedding(self.num_patches+1, decoder_dim).pos_embed
        else:
            self.dec_pos_embed = PosEmbedding(self.num_patches, decoder_dim).pos_embed

        self.encoder = encoder
        self.decoder = decoder
        self.mid_proj = nn.Linear(encoder_dim, decoder_dim, bias=True) if encoder_dim != decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.out_proj = nn.Linear(decoder_dim, self.patch_dim) if decoder_dim != self.patch_dim else nn.Identity()

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        torch.nn.init.normal_(self.mask_token, std=.02)
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
            
    def prepare_token(self, x, mask_ratio=None, **kwargs):
               
        B, nc, w, h = x.shape
        target, x = self.encoder.patch_embed(x)
        pos_embed = self.encoder.interpolate_pos_encoding(x, w, h)
        
        # add pos embed w/o cls token
        if self.apply_cls_token:
            x = x + pos_embed[:, 1:, :]
        else:
            x = x + pos_embed
        
        # apply mask
        x, mask, ids_restore = self.masking_fn(x, mask_ratio, **kwargs)
            
        # append cls token
        if self.apply_cls_token:
            cls_token = self.encoder.cls_token + pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        
        return x, target, mask, ids_restore
    
    def interpolate_pos_encoding_decoder(self, x, w, h):
        
        # npatch = x.shape[1]
        # here x is already added with cls_token
        if self.apply_cls_token:
            npatch = x.shape[1] - 1
            N = self.dec_pos_embed.shape[1] - 1
        else:
            npatch = x.shape[1]
            N = self.dec_pos_embed.shape[1]
            
        if npatch == N: # and w == h: this is not applicable for spec as w!=h
            return self.dec_pos_embed
        
        assert w == self.spec_size[0], f'mel-spectrogram freq side should remain same; should be {self.spec_size[0]} but got {w}'        

        if self.apply_cls_token:
            class_pos_embed = self.dec_pos_embed[:, 0]
            patch_pos_embed = self.dec_pos_embed[:, 1:]
        else:
            patch_pos_embed = self.dec_pos_embed
            
        dim = x.shape[-1]
        h0 = h // self.patch_spatial[1]
        
        Nw = self.spec_size[0]//self.patch_spatial[0]
        Nh = N//Nw
        h0 = h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(Nw), int(Nh), dim).permute(0, 3, 1, 2),
            scale_factor=(1, h0 / Nh),
            mode='bicubic',
        )
        assert int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        if self.apply_cls_token:
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
        else:
            return patch_pos_embed
        
    def forward_encoder(self, x, mask_ratio, **kwargs):
        x, target, mask, ids_restore = self.prepare_token(x, mask_ratio, **kwargs)
        x = self.encoder.encoder(x) # calling ViT_Backbone
        x = self.encoder.encoder_norm(x)
        return x, target, mask, ids_restore

    def forward_decoder(self, x, w, h, ids_restore):
        # embed tokens
        x = self.mid_proj(x)
        # append mask tokens to sequence
        if self.apply_cls_token:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # tackling cls token # appending mask token with encoder op
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)  # appending mask token with encoder op

        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        if self.apply_cls_token:
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        else:
            x = x_

        # add pos embed to tackle size mismatch
        # x = x + self.dec_pos_embed
        # or
        pos_embed = self.interpolate_pos_encoding_decoder(x, w, h)
        x = x + pos_embed
        
        x = self.decoder(x) # apply Transformer blocks
        x = self.out_proj(x) # predictor projection
        if self.apply_cls_token:
            x = x[:, 1:, :] # remove cls token
        
        return x

    def forward_loss(self, target, pred, mask):
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, specs, mask_ratio, return_latent=False, **kwargs):
        _, _, w, h = specs.shape
        latent, target, mask, ids_restore = self.forward_encoder(specs, mask_ratio, **kwargs)
        pred = self.forward_decoder(latent, w, h, ids_restore)
        loss = self.forward_loss(target, pred, mask)
        if return_latent:
            return latent, loss
        return loss

