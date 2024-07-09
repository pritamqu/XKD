import os
import time
import torch
import warnings
import torch.multiprocessing as mp
import yaml
from tools import environment as environ
import argparse
from tools.utils import sanity_check, set_deterministic
from tools import paths

def get_args():
        
    parser = argparse.ArgumentParser()
    
    # some stuff
    parser.add_argument("--parent_dir", default="Transformer", help="output folder name",)    
    parser.add_argument("--sub_dir", default="svm", help="output folder name",)    
    parser.add_argument("--job_id", type=str, default='00', help="jobid=%j")

    ## debug mode
    parser.add_argument('--quiet', action='store_true')

    ## dir stuff
    parser.add_argument('--data_dir', type=str, default='D:\\datasets\\Vision\\image')
    parser.add_argument("--output_dir", default="D:\\projects\\OUTPUTS", help="path where to save")
    parser.add_argument("--resume", default="", help="path where to resume")
    parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    parser.add_argument('--ckpt_dir', type=str, default=os.getenv('CHECKPOINT'))

    ## dist training stuff
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use., default to 0 while using 1 gpu')    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dist-url', default="env://", type=str, help='url used to set up distributed training, change to; "tcp://localhost:15475"')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    
    parser.add_argument("--checkpoint_path", default="", help="checkpoint_path for system restoration")
    parser.add_argument('--checkpoint_interval', default=3600, type=int, help='checkpoint_interval')
    
    ## pretrained model
    parser.add_argument("--weight_path", default="/path", help="checkpoint_path for backbone restoration.")

    ## dataset and config
    parser.add_argument("--db", default="fsd50", help="target db", choices=['esc50', 'dcase', 'kinetics_sound', 'fsd50'])  
    parser.add_argument('-c', '--config-file', type=str, help="pass config file name w/o extension", default="fold1_4s_16x4_vj_base_pool")

    ## sanity
    args = parser.parse_args()
    args = sanity_check(args)
    set_deterministic(args.seed)
    torch.backends.cudnn.benchmark = True 
    
    return args


def main(args):
       
    cfg = yaml.safe_load(open(args.config_file))
    
    print(args)
    print(cfg)
    
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print(f'number of gpus per node {ngpus_per_node} - Rank {args.rank}')
    
    if args.multiprocessing_distributed:
        print('mp.spawn calling main_worker')
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        print('direct calling main_worker')
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, cfg)


def main_worker(gpu, ngpus_per_node, args, cfg):
        
    # Setup environment
    args.gpu = gpu
    args = environ.initialize_distributed_backend(args, ngpus_per_node)
    logger, tb_writter, wandb_writter = environ.prep_environment_ddp(args, cfg) # prep_environment
    path = args.weight_path
    if os.path.isfile(path):
        if args.gpu is None:
            state = torch.load(path)
        else:
            # Map model to be loaded to specified single gpu.
            state = torch.load(path, map_location='cuda:{}'.format(args.gpu))
    else:
        raise FileNotFoundError (f'weight is not found at {path}')
        
    from engine.audio.audio_svm import linear_svm as worker
    worker(args, cfg, state, logger=logger, tb_writter=tb_writter, wandb_writter=wandb_writter)
    
if __name__ == "__main__":
    args = get_args()
    main(args=args)

