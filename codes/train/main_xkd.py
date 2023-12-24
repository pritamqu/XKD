import os
import time
import math
import sys
import torch
import warnings
import argparse
import yaml
import torch.multiprocessing as mp
from datasets.augmentations import get_aud_aug, get_vid_aug
from datasets import get_dataset, dataloader, FetchSubset
from tools import environment as environ
from models import get_model, has_batchnorms
from optimizers import get_optimizer, cosine_scheduler
from tools import AverageMeter, ProgressMeter, sanity_check, set_deterministic
from tools.utils import resume_model, save_checkpoint # general use
import torchvision
import numpy as np
GB = (1024*1024*1024)


def get_args(mode='default'):

    # mode will take care specific arguments for specific cases

    parser = argparse.ArgumentParser()

    # some stuff
    parser.add_argument("--parent_dir", default="Transformer", help="output folder name",)
    parser.add_argument("--sub_dir", default="pretext", help="output folder name",)
    parser.add_argument("--job_id", type=str, default='00', help="jobid=%j")
    parser.add_argument("--server", type=str, default="local", help="location of server",
                        choices=["ingenuity", "vector", "local", "scinet", "narval"])
    parser.add_argument("--db", default="kinetics400", help="target db",
                        choices=['kinetics400', 'audioset', 'kinetics_sound'])
    parser.add_argument('-c', '--config-file', type=str, help="config", default="xkd.yaml")

    ## debug mode
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=2)

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

    if mode=='grid':
        # grid job
        parser.add_argument("--hyper_params_search", default=False, type=bool)
        parser.add_argument("--grid_cfg", default='cfg_proj_layer', type=str, help='config dict for hyperparameter search')
        parser.add_argument("--search_num", default=0, type=int)

    args = parser.parse_args()
    args.mode = mode
    args = sanity_check(args)
    set_deterministic(args.seed)
    torch.backends.cudnn.benchmark = True

    return args

def main(args):

    cfg = yaml.safe_load(open(args.config_file))
    print(args)
    print(cfg)

    # #------------ curves ---------------
    # if args.server == 'vector':
    #     cfg['progress']['log2tb']=False
    #     cfg['progress']['wandb']=True
    # elif args.server == 'scinet':
    #     cfg['progress']['log2tb']=True
    #     cfg['progress']['wandb']=False

    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)
        # small model for debug
        cfg['model']['kwargs']['teacher_cfg']='small_encoder'
        cfg['model']['kwargs']['student_cfg']='small_encoder'
        cfg['model']['kwargs']['decoder_cfg']='small_decoder'
        cfg['model']['video_temp_kwargs']['warmup_teacher_temp_epochs'] = 1
        cfg['model']['audio_temp_kwargs']['warmup_teacher_temp_epochs'] = 1
        # cfg['model']['kwargs']['teacher_cfg']=['small_encoder', 'tiny_encoder']
        # cfg['model']['kwargs']['student_cfg']=['small_encoder', 'tiny_encoder']
        # cfg['model']['kwargs']['decoder_cfg']=['small_decoder', 'tiny_decoder']
        # cfg['model']['temp_kwargs']['warmup_teacher_temp_epochs'] = 1           

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

    #---------------------- Setup environment
    args.gpu = gpu
    args = environ.initialize_distributed_backend(args, ngpus_per_node)
    logger, tb_writter, wandb_writter = environ.prep_environment_ddp(args, cfg)
    # use apex for mixed precision training
    amp = torch.cuda.amp.GradScaler() if cfg['apex'] else None
    
    #---------------------- define model
    model = get_model(cfg['model'])
    # synchronize batch norm
    if args.distributed and cfg['sync_bn']:
        if has_batchnorms(model):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(args.gpu)
    # wrap in ddp
    model, args, cfg['dataset']['batch_size'], cfg['num_workers'] = environ.distribute_model_to_cuda(models=model,
                                                                         args=args,
                                                                         batch_size=cfg['dataset']['batch_size'],
                                                                         num_workers=cfg['num_workers'],
                                                                         ngpus_per_node=ngpus_per_node)
    
    # initialize student and teacher with same params
    model.module.init_teacher_student_same_weights()
    # teacher no grad required
    model.module.set_teacher_no_grad()
    # model size
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.add_line('number of params (M): %.2f' % (n_parameters / 1.e6))
    
    logger.add_line(f"effecting batch size: {cfg['dataset']['batch_size']*args.world_size}")

    # transformations
    if cfg['dataset']['train']['mode'] == 'global_local':
        # TODO: remove later
        video_frames=[cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'], # global duration
                    cfg['dataset']['clip_duration']*cfg['dataset']['video_fps']/cfg['dataset']['local2global_ratio'] # local duration
                    ]
        audio_duration=[cfg['dataset']['audio_clip_duration'], # global duration
                        cfg['dataset']['audio_clip_duration']/cfg['dataset']['local2global_ratio'] # local duration
                        ]
    else:
        video_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps']
        audio_duration=cfg['dataset']['audio_clip_duration']

    vid_transformations = get_vid_aug(name=cfg['dataset']['vid_transform'],
                                    crop_size=cfg['dataset']['crop_size'],
                                    num_frames=video_frames,
                                    mode=cfg['dataset']['train']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['train']['vid_aug_kwargs']) if cfg['dataset']['return_video'] else None

    aud_transformations = get_aud_aug(name=cfg['dataset']['aud_transform'],
                                    audio_fps=cfg['dataset']['audio_fps'],
                                    n_fft=cfg['dataset']['n_fft'] if 'n_fft' in cfg['dataset'] else None,
                                    n_mels=cfg['dataset']['n_mels'] if 'n_mels' in cfg['dataset'] else None,
                                    duration=audio_duration,
                                    hop_length=cfg['dataset']['hop_length'] if 'hop_length' in cfg['dataset'] else None,
                                    mode=cfg['dataset']['train']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['train']['aud_aug_kwargs']) if cfg['dataset']['return_audio'] else None

    # dataset
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=vid_transformations,
                                audio_transform=aud_transformations,
                                split='train')

    if args.debug:
        train_dataset = FetchSubset(train_dataset, cfg['dataset']['batch_size']*ngpus_per_node*args.debug_subset_size)

    # dataloader
    train_loader = dataloader.make_dataloader(dataset=train_dataset,
                                              batch_size=cfg['dataset']['batch_size'],
                                              use_shuffle=cfg['dataset']['train']['use_shuffle'],
                                              drop_last=cfg['dataset']['train']['drop_last'],
                                              num_workers=cfg['num_workers'],
                                              distributed=args.distributed)
    
    # define optimizer
    optimizer = get_optimizer(name= cfg['hyperparams']['optimizer']['name'],
                              model=model, lr=1e-3, # this is overwritten by lr-scheduler
                              weight_decay=None, #cfg['hyperparams']['optimizer']['weight_decay'],
                              betas=cfg['hyperparams']['optimizer']['betas'])
    
    # define EMA Scheduler
    ema_scheduler={}
    if cfg['hyperparams']['vid_ema']['name'] == 'cosine':
        ema_scheduler['vid_ema'] = cosine_scheduler(base_value=cfg['hyperparams']['vid_ema']['base'], 
                                        final_value=cfg['hyperparams']['vid_ema']['final'], 
                                        epochs=cfg['hyperparams']['num_epochs'], 
                                        niter_per_ep=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['vid_ema']['warmup_epochs'], 
                                        start_warmup_value=cfg['hyperparams']['vid_ema']['warmup'])
    else:
        raise NotImplementedError(print(f"{cfg['hyperparams']['vid_ema']['name']} not implemented"))
        
    if cfg['hyperparams']['aud_ema']['name'] == 'cosine':
        ema_scheduler['aud_ema'] = cosine_scheduler(base_value=cfg['hyperparams']['aud_ema']['base'], 
                                        final_value=cfg['hyperparams']['aud_ema']['final'], 
                                        epochs=cfg['hyperparams']['num_epochs'], 
                                        niter_per_ep=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['aud_ema']['warmup_epochs'], 
                                        start_warmup_value=cfg['hyperparams']['aud_ema']['warmup'])
    else:
        raise NotImplementedError(print(f"{cfg['hyperparams']['aud_ema']['name']} not implemented"))
        
    # define lr scheduler
    if cfg['hyperparams']['lr']['name'] == 'cosine':
        lr_scheduler = cosine_scheduler(base_value=cfg['hyperparams']['lr']['base_lr'], 
                                        final_value=cfg['hyperparams']['lr']['final_lr'], 
                                        epochs=cfg['hyperparams']['num_epochs'], 
                                        niter_per_ep=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['lr']['warmup_epochs'], 
                                        start_warmup_value=cfg['hyperparams']['lr']['warmup_lr'])

    else:
        raise NotImplementedError(print(f"{cfg['hyperparams']['weight_decay']['name']} not implemented"))

    # define wd scheduler
    if cfg['hyperparams']['weight_decay']['name'] == 'cosine':
        wd_scheduler = cosine_scheduler(base_value=cfg['hyperparams']['weight_decay']['base'], 
                                        final_value=cfg['hyperparams']['weight_decay']['final'], 
                                        epochs=cfg['hyperparams']['num_epochs'], 
                                        niter_per_ep=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['weight_decay']['warmup_epochs'], 
                                        start_warmup_value=cfg['hyperparams']['weight_decay']['warmup'])
    
    else:
        raise NotImplementedError(print(f"{cfg['hyperparams']['weight_decay']['name']} not implemented"))


    # temperature schedule
    video_temp_kwargs = cfg['model']['video_temp_kwargs']
    video_teacher_temp_schedule = np.concatenate((
            np.linspace(video_temp_kwargs['warmup_teacher_temp'], video_temp_kwargs['teacher_temp'], video_temp_kwargs['warmup_teacher_temp_epochs']),
            np.ones(cfg['hyperparams']['num_epochs'] - video_temp_kwargs['warmup_teacher_temp_epochs']) * video_temp_kwargs['teacher_temp']
        ))
    video_student_temp_schedule = np.ones(cfg['hyperparams']['num_epochs']) * video_temp_kwargs['student_temp']

    audio_temp_kwargs = cfg['model']['audio_temp_kwargs']
    audio_teacher_temp_schedule = np.concatenate((
            np.linspace(audio_temp_kwargs['warmup_teacher_temp'], audio_temp_kwargs['teacher_temp'], audio_temp_kwargs['warmup_teacher_temp_epochs']),
            np.ones(cfg['hyperparams']['num_epochs'] - audio_temp_kwargs['warmup_teacher_temp_epochs']) * audio_temp_kwargs['teacher_temp']
        ))
    audio_student_temp_schedule = np.ones(cfg['hyperparams']['num_epochs']) * audio_temp_kwargs['student_temp']



    ## try loading from checkpoint

    ## manual resume
    model, optimizer, start_epoch, amp = resume_model(args, model, optimizer, amp, logger)

    end_epoch = cfg['hyperparams']['num_epochs']
    logger.add_line('='*30 + ' Training Started' + '='*30)

    for epoch in range(start_epoch, end_epoch):
               
        
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        fwd_kwargs = cfg['model']['fwd_kwargs']
        fwd_kwargs['teacher_temp'] = {'video': video_teacher_temp_schedule[epoch], 'audio': audio_teacher_temp_schedule[epoch]}
        fwd_kwargs['student_temp'] = {'video': video_student_temp_schedule[epoch], 'audio': audio_student_temp_schedule[epoch]}
        
        train_one_epoch(args, model, optimizer, 
                        lr_scheduler, ema_scheduler, wd_scheduler,
                        train_loader, 
                        logger, tb_writter, wandb_writter, 
                        epoch, cfg['progress']['print_freq'], amp, 
                        fwd_kwargs)

        # Save checkpoint
        if args.rank==0:
            ## normal checkpoint
            save_checkpoint(args, model, optimizer, epoch, amp, logger)
            
        # Save just the backbone for further use
        if args.rank==0 and ((epoch+1==end_epoch) or (epoch+1)%50==0):
        # if args.rank==0 and (epoch+1==end_epoch):
            model_path = os.path.join(args.ckpt_dir, f"{cfg['model']['name']}_{args.sub_dir}_{args.db}_ep{epoch}")
            model.module.save_state_dicts(model_path)
            # model_path = os.path.join(args.ckpt_dir, f"{cfg['model']['name']}_{args.sub_dir}_{args.db}_ep{epoch}.pth.tar")
            # torch.save(model.module.state_dict(), model_path)
            print(f"model is saved to \n{args.ckpt_dir}")            
                
        if args.distributed:
            torch.distributed.barrier() # check this

    # finish logging for this run
    if wandb_writter is not None:
        wandb_writter.finish()
    return

def train_one_epoch(args, model, optimizer, 
                    lr_scheduler, ema_scheduler, wd_scheduler,
                    train_loader, 
                    logger, tb_writter, wandb_writter, 
                    epoch, print_freq, amp, fwd_kwargs):

    model.train()
    batch_size = train_loader.batch_size
    logger.add_line('[Train] Epoch {}'.format(epoch))
    batch_time = AverageMeter('Time', ':6.3f', window_size=100)
    data_time = AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = AverageMeter('Loss', ':.3e')
    gpu_meter = AverageMeter('GPU', ':4.2f')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, loss_meter, 
                                                 gpu_meter,
                                                 ],
                                          phase='pretext-iter', epoch=epoch, logger=logger, tb_writter=None)
    device = args.gpu if args.gpu is not None else 0
    
    clip_grad = fwd_kwargs.pop('clip_grad')
    freeze_last_layer = fwd_kwargs.pop('freeze_last_layer')
    
    end = time.time()
    for i, sample in enumerate(train_loader):
        # break
    
        # update lr & weight decay
        step = epoch * len(train_loader) + i
        for pi, param_group in enumerate(optimizer.param_groups):

            param_group["lr"] = lr_scheduler[step]            
            # param_group["weight_decay"] = wd_scheduler[step]
            if pi == 1:  # only the second group is regularized; first group has bias and norms
                param_group["weight_decay"] = wd_scheduler[step]
    
    
        # measure data loading time
        data_time.update(time.time() - end)
        if train_loader.dataset.return_video:
            if isinstance(sample, dict):
                frames = [k.cuda(device, non_blocking=True) for k in sample['frames']]
            else:
                frames = sample['frames'].cuda(device, non_blocking=True)
            # batch_size = frames.size(0)
        if train_loader.dataset.return_audio:
            if isinstance(sample, dict):
                specs = [k.cuda(device, non_blocking=True) for k in sample['audio']]
            else:
                specs = sample['audio'].cuda(device, non_blocking=True)
            # batch_size = specs.size(0)

        optimizer.zero_grad()
        param_norms = None
        
        if amp is not None: # mixed precision
            with torch.cuda.amp.autocast():
                data_dict = model.forward(frames, specs, **fwd_kwargs)
        else:
            data_dict = model.forward(frames, specs, **fwd_kwargs)

        loss = data_dict.pop('loss')
        loss_meter.update(loss, batch_size)
        data_dict.update({'lr':optimizer.param_groups[0]["lr"]})
        data_dict.update({'wd':optimizer.param_groups[1]["weight_decay"]})
        # data_dict.update({'vid_lr':optimizer.param_groups[0]["lr"]})
        # data_dict.update({'vid_wd':optimizer.param_groups[1]["weight_decay"]})
        # data_dict.update({'aud_lr':optimizer.param_groups[2]["lr"]})
        # data_dict.update({'aud_wd':optimizer.param_groups[3]["weight_decay"]})
        data_dict.update({'vid_ema':ema_scheduler['vid_ema'][step]})
        data_dict.update({'aud_ema':ema_scheduler['aud_ema'][step]})
        data_dict.update({'video_teacher_temp':fwd_kwargs['teacher_temp']['video']})
        data_dict.update({'video_student_temp':fwd_kwargs['student_temp']['video']})
        data_dict.update({'audio_teacher_temp':fwd_kwargs['teacher_temp']['audio']})
        data_dict.update({'audio_student_temp':fwd_kwargs['student_temp']['audio']})

        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training") # for log
            logger.add_line(f"Loss is {loss.item()}, stopping training") # for logger
            sys.exit(1)

        if amp is not None:
            amp.scale(loss).backward()
            # -- copied from DINO to stabilize loss         
            if clip_grad:
                amp.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = model.module.clip_gradients(clip_grad)
            model.module.cancel_gradients_last_layer(epoch, freeze_last_layer)
            
            amp.step(optimizer)
            amp.update()
        else:
            loss.backward()
            # -- copied from DINO to stabilize loss
            if clip_grad:
                amp.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = model.module.clip_gradients(clip_grad)
            model.module.cancel_gradients_last_layer(epoch, freeze_last_layer)
            
            optimizer.step()

        # update teachers
        vm = ema_scheduler['vid_ema'][step] # video
        am = ema_scheduler['aud_ema'][step] # audio
        model.module.update_teacher(vm=vm, am=am)

        # measure elapsed time
        batch_time.update(time.time() - end)
        # measure gpu usage
        # if args.server=='scinet':
        gpu_meter.update(torch.cuda.max_memory_allocated()/GB)        

        # print to terminal and tensorboard
        step = epoch * len(train_loader) + i
        if (i+1) % print_freq == 0 or i == 0 or i+1 == len(train_loader):
            progress.display(i+1)

            if tb_writter is not None:
                for kk in data_dict.keys():
                    tb_writter.add_scalar(f'pretext-iter/{kk}', data_dict[kk], step)
                for meter in progress.meters:
                    tb_writter.add_scalar(f'pretext-iter/{meter.name}', meter.val, step)

            if wandb_writter is not None:
                for kk in data_dict.keys():
                    wandb_writter.log({f'pretext-iter/{kk}': data_dict[kk], 'custom_step': step})
                for meter in progress.meters:
                     wandb_writter.log({f'pretext-iter/{meter.name}': meter.val, 'custom_step': step})

        end = time.time()

    # Sync metrics across all GPUs and print final averages
    if args.distributed:
        # progress.synchronize_meters(args.gpu)
        progress.synchronize_meters_custom(args.gpu)

    if tb_writter is not None:
        tb_writter.add_scalar('pretext-epoch/Epochs', epoch, epoch)
        for meter in progress.meters:
            if meter.name == 'Time' or meter.name == 'Data': # printing total time
                tb_writter.add_scalar(f'pretext-epoch/{meter.name}', meter.sum, epoch)
            else:
                tb_writter.add_scalar(f'pretext-epoch/{meter.name}', meter.avg, epoch)

    if wandb_writter is not None:
        wandb_writter.log({'pretext-epoch/Epochs': epoch, 'custom_step': epoch})
        for meter in progress.meters:
            if meter.name == 'Time' or meter.name == 'Data':
                wandb_writter.log({f'pretext-epoch/{meter.name}': meter.sum, 'custom_step': epoch})
            else:
                wandb_writter.log({f'pretext-epoch/{meter.name}': meter.avg, 'custom_step': epoch})
                
    # quick fix
    fwd_kwargs.update({'clip_grad':clip_grad, 'freeze_last_layer':freeze_last_layer}) 
    
    return


if __name__ == "__main__":

    args = get_args()
    if args.server =='local':
        args.debug=True
    main(args=args)