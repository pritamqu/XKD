from .vit2 import ViT_Backbone
from .projector import DINOHead, MultiViewWrapper
from .loss import MMD_Loss, DINOLossX
from .vit_video import VideoViT, vid_vit
from .vit_audio import AudioViT, aud_vit
from .mask_model import VideoMAEWrapper, AudioMAEWrapper


encoder_dict = {
    'base_encoder' : {'embed_dim':768, 'depth':12, 'num_heads':12},
    'large_encoder' : {'embed_dim':1024, 'depth':24, 'num_heads':16},
    }

decoder_dict = {
    'large_decoder' : {'embed_dim':512, 'depth':8, 'num_heads':16},
    'base_decoder' : {'embed_dim':384, 'depth':4, 'num_heads':12},
    }

projector_dict = {    
    '2048-gelu-3-256-8192-norm3' : {'out_dim': 8192, 'act': 'gelu', 'nlayers':3, 'hidden_dim': 2048, 
                       'norm': 'ln', 'last_norm': 'ln', 'bottleneck_dim': 256, 'norm_last_layer': True},  
    }


