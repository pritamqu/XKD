# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""


import torch
import numpy as np
import torch.nn as nn

try:
    from models.modules import DINOHead, MultiViewWrapper, DINOLossX, \
        AudioViT, VideoViT, ViT_Backbone, AudioMAEWrapper, VideoMAEWrapper, \
                    encoder_dict, decoder_dict, projector_dict, vid_vit, aud_vit, \
                        MMD_Loss
    from .mask import random_masking
except:
    from modules import DINOHead, MultiViewWrapper, DINOLossX, \
        AudioViT, VideoViT, ViT_Backbone, AudioMAEWrapper, VideoMAEWrapper, \
                    encoder_dict, decoder_dict, projector_dict, vid_vit, aud_vit, \
                        MMD_Loss 
    from mask import random_masking

class XKD_Net(nn.Module):
    def __init__(self,
                 # video stuff
                 frame_size = (3, 224, 224),
                 num_frames = 16,
                 vid_patch_spatial = (16, 16),
                 vid_patch_temporal = 2,
                 # audio stuff
                 spec_size = (80, 224),
                 spec_patch_spatial = (4, 16),
                 # models
                 teacher_cfg=None,
                 student_cfg=None, 
                 decoder_cfg=None,
                 projector_cfg=None,
                 # others
                 center_momentum=0.9,
                 masking_fn=None,
                 align_loss=None,
                 apply_cls_token=True,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=True,
                 **kwargs,
                 ):
        super().__init__()
        
        """ 
        masked data modelling + local-global correspondence through knowledge distillation;
        modality specific student and teacher
        """
        
        teacher_dim = encoder_dict[teacher_cfg]['embed_dim']
        student_dim = encoder_dict[student_cfg]['embed_dim']
        decoder_dim = decoder_dict[decoder_cfg]['embed_dim']
        proj_out_dim = projector_dict[projector_cfg]['out_dim']
        
        # ---------- MA Backbone for Teacher
        ma_backbone_teacher = ViT_Backbone(**encoder_dict[teacher_cfg], mlp_ratio=4)
        # ---------- MA Backbone for Studnet
        ma_backbone_student = ViT_Backbone(**encoder_dict[student_cfg], mlp_ratio=4)

        # ---------- video               
        # agnostic teacher
        self.vid_teacher = MultiViewWrapper(
            backbone= VideoViT(
                         frame_size = frame_size,
                         num_frames = num_frames,
                         patch_spatial = vid_patch_spatial,
                         patch_temporal = vid_patch_temporal,
                         encoder_dim = teacher_dim,
                         encoder = ma_backbone_teacher,
                         apply_cls_token=apply_cls_token,
                         norm_layer=norm_layer,
                         ),
            head = DINOHead(in_dim=teacher_dim,
                **projector_dict[projector_cfg]) if projector_cfg != 'none' else nn.Identity(),
            )
        
        # agnostic student
        # this backbone is shared b/w kd and autoencoder
        vid_backbone=VideoViT(
                     frame_size = frame_size,
                     num_frames = num_frames,
                     patch_spatial = vid_patch_spatial,
                     patch_temporal = vid_patch_temporal,
                     encoder_dim = student_dim,
                     encoder = ma_backbone_student,
                     apply_cls_token=apply_cls_token,
                     norm_layer=norm_layer,
                     )
        self.vid_student = MultiViewWrapper(
            backbone=vid_backbone,
            head = DINOHead(in_dim=student_dim,
                **projector_dict[projector_cfg]) if projector_cfg != 'none' else nn.Identity(),
            )
        
        self.vid_ae = VideoMAEWrapper(
            encoder=vid_backbone,
            decoder = ViT_Backbone(**decoder_dict[decoder_cfg], mlp_ratio=4),
            encoder_dim = student_dim,
            decoder_dim = decoder_dim,
            masking_fn=masking_fn,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
            )
        
        # ---------- audio
        # agnostic teacher
        self.aud_teacher = MultiViewWrapper(
            backbone=AudioViT(
                         spec_size = spec_size,
                         patch_spatial = spec_patch_spatial,
                         encoder_dim = teacher_dim,
                         encoder = ma_backbone_teacher,
                         apply_cls_token=apply_cls_token,
                         norm_layer=norm_layer,
                         ),
            head = DINOHead(in_dim=teacher_dim,
                **projector_dict[projector_cfg]) if projector_cfg != 'none' else nn.Identity(),
            )
        
        # agnostic student
        # this backbone is shared b/w kd and autoencoder
        aud_backbone=AudioViT(
                     spec_size = spec_size,
                     patch_spatial = spec_patch_spatial,
                     encoder_dim = student_dim,
                     encoder = ma_backbone_student,
                     apply_cls_token=apply_cls_token,
                     norm_layer=norm_layer,
                     )
        self.aud_student = MultiViewWrapper(
            backbone=aud_backbone,
            head = DINOHead(in_dim=student_dim,
                **projector_dict[projector_cfg]) if projector_cfg != 'none' else nn.Identity(),
            )
        
        self.aud_ae = AudioMAEWrapper(
            encoder=aud_backbone,
            decoder = ViT_Backbone(**decoder_dict[decoder_cfg], mlp_ratio=4),
            encoder_dim = student_dim,
            decoder_dim = decoder_dim,
            masking_fn=masking_fn,
            norm_layer=norm_layer,
            norm_pix_loss=norm_pix_loss,
            )
        
        if isinstance(center_momentum, list):
            self.vt2as_ce_loss = DINOLossX(out_dim=proj_out_dim, center_momentum=center_momentum[0])
            self.at2vs_ce_loss = DINOLossX(out_dim=proj_out_dim, center_momentum=center_momentum[1])
        else:
            self.vt2as_ce_loss = DINOLossX(out_dim=proj_out_dim, center_momentum=center_momentum)
            self.at2vs_ce_loss = DINOLossX(out_dim=proj_out_dim, center_momentum=center_momentum)
  
        self.masking_fn = masking_fn
        self.align_loss = align_loss
        self.encoder_dim = student_dim  # using later for feature extraction
        self.apply_cls_token = apply_cls_token
        self.kld_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def save_state_dicts(self, model_path):
        """ custom function to save teacher and student states for easy load.
        """
        torch.save(self.vid_student.backbone.state_dict(), model_path+'vid_student.pth.tar')
        torch.save(self.vid_teacher.backbone.state_dict(), model_path+'vid_teacher.pth.tar')
        torch.save(self.aud_student.backbone.state_dict(), model_path+'aud_student.pth.tar')
        torch.save(self.aud_teacher.backbone.state_dict(), model_path+'aud_teacher.pth.tar')

    def clip_gradients(self, clip):
        # adopted from DINO
        norms = []
        # vid
        for name, p in self.vid_student.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                norms.append(param_norm.item())
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)
        # aud
        for name, p in self.aud_student.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                norms.append(param_norm.item())
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)                    
                    
        return norms
    
    def cancel_gradients_last_layer(self, epoch, freeze_last_layer):
        # on projector # adopted from DINO
        if epoch >= freeze_last_layer:
            return
        # vid
        for n, p in self.vid_student.head.named_parameters():
            if "last_layer" in n:
                p.grad = None
        # aud
        for n, p in self.aud_student.head.named_parameters():
            if "last_layer" in n:
                p.grad = None
    
    def _forward_teacher(self, x, mode):
        """ feature extraction directly from the backbone """
        if mode == 'frames':
            return self.vid_teacher.backbone(x)  
        elif mode == 'specs':
            return self.aud_teacher.backbone(x) 
        else:
            raise ValueError(f'{mode} unknown mode')  
    
    def _forward_student(self, x, mode):
        """ feature extraction directly from the backbone """
        if mode == 'frames':
            return self.vid_student.backbone(x)  
        elif mode == 'specs':
            return self.aud_student.backbone(x) 
        else:
            raise ValueError(f'{mode} unknown mode')

    def forward_features(self, x, mode, key='teacher'):
        """ feature extraction directly from the backbone """
        if key=='teacher':
            return self._forward_teacher(x, mode)
        elif key=='student':
            return self._forward_student(x, mode)

    @torch.no_grad()
    def update_teacher(self, vm, am):
        # θt = m ∗ θt + (1 − m) ∗ θs
        # with torch.no_grad():
        
        # intra-modality update
        # vid updates 
        for param_v, param_k in zip(
                                    self.vid_student.parameters(), 
                                    self.vid_teacher.parameters()):

            param_k.data.mul_(vm).add_((1 - vm) * param_v.detach().data)
        # aud updates
        for param_a, param_k in zip(
                                    self.aud_student.parameters(), 
                                    self.aud_teacher.parameters()):
            
            param_k.data.mul_(am).add_((1 - am) * param_a.detach().data)

    def init_teacher_student_same_weights(self):
        self.vid_teacher.load_state_dict(self.vid_student.state_dict(), strict=False)
        self.aud_teacher.load_state_dict(self.aud_student.state_dict(), strict=False)

    def set_teacher_no_grad(self):
        for p in self.vid_teacher.parameters():
            p.requires_grad = False
        for p in self.aud_teacher.parameters():
            p.requires_grad = False

    def cross_modal_attn(self, vattn, aattn, mode):
        
        def _cal_attn_a2b_mode0(attna, attnb):
            attna = attna[:, :, None, :]
            attnb = attnb[:, :, None, :].transpose(-2, -1)
            cm_attn = attna*attnb 
            return cm_attn
                
        if mode=='mean':
            cm_attn =  _cal_attn_a2b_mode0(vattn, aattn)
            cvattn = (cm_attn).mean(dim=-2)/vattn.mean(dim=-1, keepdim=True) 
            caattn = (cm_attn).mean(dim=-1)/aattn.mean(dim=-1, keepdim=True) 
        elif mode=='softmax':
            cm_attn =  _cal_attn_a2b_mode0(vattn, aattn)
            cvattn = (cm_attn).mean(dim=-2).softmax(dim=-1)
            caattn = (cm_attn).mean(dim=-1).softmax(dim=-1)
                             
        else:
            raise NotImplementedError()

        return cvattn, caattn
    
    def align_feats(self, feat, attn):
        attn = attn.mean(dim=1).unsqueeze(dim=-1)
        eng_org = ((feat.mean(dim=1))**2).sum(dim=1, keepdim=True)  # taking the patch mean; then calculate energy; [B, 1]
        eng_aft = (((feat*attn).mean(dim=1))**2).sum(dim=1, keepdim=True)  # multiply the patch attn; patch mean; calculate energy; # [B, 1]
        scalar = (eng_org / eng_aft).sqrt().unsqueeze(dim=-1)
        new_feat = scalar * attn * feat
        
        return new_feat
    
    
    def forward(self, 
                frames, specs,
                teacher_temp, student_temp, 
                global_views_number,
                vid_mask_ratio, aud_mask_ratio,
                align_coeff,
                cmkd_coeff, recon_coeff, 
                cm_attn_mode,align_mode,
                ):
        """ self-supervised main training loop 
        cross-modal kd + reconstruction
        """
        all_loss = {}
        if isinstance(global_views_number, list):
            global_views_number_vid, global_views_number_aud = global_views_number
        else:
            global_views_number_vid = global_views_number_aud = global_views_number
            
            
            
        """ reconstruction """
        
        # reconstruction to learn intra-modal representations
        # student performing reconstruction

        vid_student_output_g, vid_recon_loss = self.vid_ae(torch.cat(frames[:global_views_number_vid]), 
                                               mask_ratio=vid_mask_ratio, return_latent=True)
        aud_student_output_g, aud_recon_loss = self.aud_ae(torch.cat(specs[:global_views_number_aud]), 
                                               mask_ratio=aud_mask_ratio, return_latent=True)
        
        recon_loss = (vid_recon_loss + aud_recon_loss)/2
        all_loss['vid_recon_loss']=vid_recon_loss
        all_loss['aud_recon_loss']=aud_recon_loss
        
        """ knowledge distillation """
        
        # get global views
        # teacher op on global views
        with torch.no_grad():
            vattn, vid_teacher_output = self.vid_teacher.backbone.forward_last_layer_attn(frames[global_views_number_vid])
            aattn, aud_teacher_output = self.aud_teacher.backbone.forward_last_layer_attn(specs[global_views_number_aud])
            
            if self.apply_cls_token:
                vattn = vattn[:, :, 0, 1:] # -> B, nH, num_of_cuboids
                aattn = aattn[:, :, 0, 1:] # -> B, nH, num_of_patches
            else:
                vattn = vattn.mean(dim=-2)
                aattn = aattn.mean(dim=-2)
            
            if self.apply_cls_token: # remove cls if presents
                vid_teacher_output = vid_teacher_output[:, 1:, ::]
                aud_teacher_output = aud_teacher_output[:, 1:, ::]
                
            cvattn, caattn = self.cross_modal_attn(vattn=vattn, aattn=aattn, mode=cm_attn_mode)            
            vid_teacher_output = self.align_feats(vid_teacher_output, cvattn)
            aud_teacher_output = self.align_feats(aud_teacher_output, caattn)
            # pass to the heads 
            vid_teacher_output = self.vid_teacher.head(vid_teacher_output.mean(dim=1)) # patch mean
            aud_teacher_output = self.aud_teacher.head(aud_teacher_output.mean(dim=1)) # patch mean
            
        # student op on local views
        vid_student_output_l = self.vid_student.forward_last_layer_attn(frames[global_views_number_vid:], 
                                                                        pass_patch_mean=True, 
                                                                        cls_token_present=self.apply_cls_token,
                                                                        return_last_layer_attn=False,)
        aud_student_output_l = self.aud_student.forward_last_layer_attn(specs[global_views_number_aud:], 
                                                                        pass_patch_mean=True, 
                                                                        cls_token_present=self.apply_cls_token,
                                                                        return_last_layer_attn=False,)
        
        # taking patch mean than cls which is default
        if self.apply_cls_token:
            # further passing the student global view representations (coming from ae) through the student head
            vid_student_output_g = self.vid_student.head(vid_student_output_g[:, 1:, ::].mean(dim=1))
            aud_student_output_g = self.aud_student.head(aud_student_output_g[:, 1:, ::].mean(dim=1))
        else:
            # further passing the student global view representations (coming from ae) through the student head
            vid_student_output_g = self.vid_student.head(vid_student_output_g.mean(dim=1))
            aud_student_output_g = self.aud_student.head(aud_student_output_g.mean(dim=1))

        vid_student_output = torch.cat([vid_student_output_g, vid_student_output_l])
        aud_student_output = torch.cat([aud_student_output_g, aud_student_output_l])
        
        # FIXME if student_output==student_output_l then num_student_views=len(frames)-global_views_number
        # assert len(frames) == len(specs)
        num_student_views_vid = len(frames) 
        num_student_views_aud = len(specs)
        num_teacher_views_vid = global_views_number_vid
        num_teacher_views_aud = global_views_number_aud
        
                
        #------------------ cross-modal stuff
        # earlier we modified the teacher representations to align with the cross-modal student representations
        as2vs_align_loss = self.align_loss(vid_student_output, aud_student_output)
        vt2at_align_loss = self.align_loss(vid_teacher_output, aud_teacher_output)
        align_loss = (as2vs_align_loss+vt2at_align_loss)/2
        all_loss['as2vs_align_loss']=as2vs_align_loss
        all_loss['vt2at_align_loss']=vt2at_align_loss
 
        # calculate cross-modal kd loss
        vt2as_kd_loss = self.vt2as_ce_loss(aud_student_output, vid_teacher_output, 
                    student_temp['audio'], teacher_temp['video'], 
                    num_student_views_aud, num_teacher_views_vid)
        at2vs_kd_loss = self.at2vs_ce_loss(vid_student_output, aud_teacher_output, 
                    student_temp['audio'], teacher_temp['audio'], 
                    num_student_views_vid, num_teacher_views_aud)
        
        cmkd_loss = (vt2as_kd_loss + at2vs_kd_loss)/2
        all_loss['vt2as_kd_loss']=vt2as_kd_loss
        all_loss['at2vs_kd_loss']=at2vs_kd_loss
        
        loss = (align_loss*align_coeff +
                    cmkd_loss*cmkd_coeff + 
                        recon_loss*recon_coeff ) / (align_coeff+cmkd_coeff+recon_coeff)
        
        all_loss['recon_loss'] = recon_loss
        all_loss['cmkd_loss'] = cmkd_loss
        all_loss['alignment_loss'] = align_loss
        all_loss['loss'] = loss
                
        return all_loss

def XKD_MATS(
            frame_size,
            num_frames,
            vid_patch_spatial,
            vid_patch_temporal,
            spec_size,
            spec_patch_spatial,
            teacher_cfg,
            student_cfg, 
            decoder_cfg,
            projector_cfg,
            center_momentum,
            masking_fn,
            align_loss,
            apply_cls_token,
            norm_layer=nn.LayerNorm,
            norm_pix_loss=True,
        ):

    masking_fn = eval(masking_fn)
    if align_loss=='mmd':
        align_loss = MMD_Loss(kernel_type = 'gaussian')
    else:
        raise NotImplementedError()


    model = XKD_Net(
                 frame_size = frame_size,
                 num_frames = num_frames,
                 vid_patch_spatial = vid_patch_spatial,
                 vid_patch_temporal = vid_patch_temporal,
                 spec_size = spec_size,
                 spec_patch_spatial = spec_patch_spatial,
                 teacher_cfg=teacher_cfg,
                 student_cfg=student_cfg, 
                 decoder_cfg=decoder_cfg,
                 projector_cfg=projector_cfg,
                 center_momentum=center_momentum,
                 masking_fn=masking_fn,
                 align_loss=align_loss,
                 apply_cls_token=apply_cls_token,
                 norm_layer=norm_layer,
                 norm_pix_loss=norm_pix_loss,
                 )
    
    return model
