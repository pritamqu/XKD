from .vit2 import ViT_Backbone
from .projector import DINOHead, MultiViewWrapper
from .loss import MMD_Loss, DINOLossX
from .vit_video import VideoViT, vid_vit
from .vit_audio import AudioViT, aud_vit
from .mask_model import VideoMAEWrapper, AudioMAEWrapper


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
    'base_decoder_d2' : {'embed_dim':384, 'depth':2, 'num_heads':12},
    'base_decoder_d3' : {'embed_dim':384, 'depth':3, 'num_heads':12},
    'small_decoder' : {'embed_dim':192, 'depth':4, 'num_heads':6},
    'small_decoder_d1' : {'embed_dim':192, 'depth':1, 'num_heads':6},
    'small_decoder_d2' : {'embed_dim':192, 'depth':2, 'num_heads':6},
    'small_decoder_d3' : {'embed_dim':192, 'depth':3, 'num_heads':6},
    'tiny_decoder' : {'embed_dim':96, 'depth':4, 'num_heads':3},   
    'tiny_decoder_d1' : {'embed_dim':96, 'depth':1, 'num_heads':3},   

    }

projector_dict = {
    # '2048-gelu-3-2048' : {'out_dim': 2048, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
    #                    'norm': None, 'last_norm': None, 'bottleneck_dim': 0, 'norm_last_layer': True},
    # '2048-gelu-3-2048-bn' : {'out_dim': 2048, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
    #                    'norm': 'bn', 'last_norm': None, 'bottleneck_dim': 0, 'norm_last_layer': True},
    # '2048-gelu-3-2048-syncbn' : {'out_dim': 2048, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
    #                     'norm': 'syncbn', 'last_norm': None, 'bottleneck_dim': 0, 'norm_last_layer': True},
    # '2048-gelu-3-8192' : {'out_dim': 8192, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
    #                    'norm': None, 'last_norm': None, 'bottleneck_dim': 0, 'norm_last_layer': True},
    '2048-gelu-3-256-8192' : {'out_dim': 8192, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': None, 'last_norm': None, 'bottleneck_dim': 256, 'norm_last_layer': True}, # this cause collapse in cm; this is best in im;
    '2048-gelu-3-256-4096' : {'out_dim': 4096, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': None, 'last_norm': None, 'bottleneck_dim': 256, 'norm_last_layer': True},
    # trying cmkd
    '2048-gelu-3-256-8192-norm1' : {'out_dim': 8192, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': None, 'last_norm': None, 'bottleneck_dim': 256, 'norm_last_layer': False},
    '2048-gelu-3-256-8192-norm2' : {'out_dim': 8192, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': None, 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True},    
    '2048-gelu-3-256-8192-norm3' : {'out_dim': 8192, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True},  
    '2048-gelu-3-256-8192-norm4' : {'out_dim': 8192, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': None, 'bottleneck_dim': 256, 'norm_last_layer': True},   
    '2048-gelu-3-256-8192-norm3bn' : {'out_dim': 8192, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'bn', 'bottleneck_dim': 256, 'norm_last_layer': True},
    
    '2048-gelu-3-256-2048-norm3' : {'out_dim': 2048, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True}, 
    '2048-gelu-3-256-4096-norm3' : {'out_dim': 4096, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True},  
    '2048-gelu-3-256-16384-norm3' : {'out_dim': 16384, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True},  
    '2048-gelu-3-256-32768-norm3' : {'out_dim': 32768, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True},  
    '2048-gelu-3-256-65536-norm3' : {'out_dim': 65536, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True}, 
                       
    '2048-gelu-4-256-4096-norm3' : {'out_dim': 4096, 'act': 'gelu', 'nlayers':4, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True},  
    '2048-gelu-2-256-4096-norm3' : {'out_dim': 4096, 'act': 'gelu', 'nlayers':2, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True}, 
    
    '2048-gelu-4-256-8192-norm3' : {'out_dim': 8192, 'act': 'gelu', 'nlayers':4, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True},      
    '2048-gelu-2-256-8192-norm3' : {'out_dim': 8192, 'act': 'gelu', 'nlayers':2, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True},     
    
    }


