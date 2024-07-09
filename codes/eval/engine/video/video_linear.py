# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import os
import torch
import time
import numpy as np
from engine import AverageMeter, ProgressMeter, accuracy, warmup_cosine_scheduler, warmup_multistep_scheduler, warmup_fixed_scheduler, Classifier
from datasets.augmentations import get_vid_aug
from datasets import get_dataset, dataloader, FetchSubset        
from tools import environment as environ
# from tools.utils import resume_model, save_checkpoint
from checkpointing import commit_state3, create_or_restore_training_state3
from models.modules.vit_video2 import VideoViT
from models import has_batchnorms
GB = (1024*1024*1024)
from collections import OrderedDict
import copy
from einops import rearrange
import torch.nn as nn

def finetune(args, cfg, backbone_state_dict, ngpus_per_node, logger, tb_writter, wandb_writter):
    
    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)
                
    # get the model
    model=VideoViT(**cfg['model']['backbone'])    
    # fix the names to the more standard one.
    state = OrderedDict()
    for key in backbone_state_dict:
        val = backbone_state_dict[key]
        if key.startswith('encoder.'):
            state[key.replace('encoder.', '')] = val
        elif key.startswith('encoder_'):
            state[key.replace('encoder_', '')] = val
        elif key == 'enc_pos_embed':
            state['pos_embed'] = val
        elif key.startswith('cuboid_embed'):
            state[key.replace('cuboid_embed', 'patch_embed')] = val
        else:
            state[key] = val
               
    # load weights
    model.load_state_dict(state, strict=True)
    # del backbone_state_dict
    # configure head.
    # model.head = nn.Linear(model.embed_dim, cfg['model']['classifier']['num_classes']) 
    # model.head.weight.data.normal_(mean=0.0, std=0.01)
    # model.head.bias.data.zero_()     
    # TODO: setup cfg to pass **classifier
    model.head = Classifier(feat_dim=model.embed_dim, **cfg['model']['classifier'])

    
    # freeze all but the classifier head for linear eval
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True
     
    # check no. of model params
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.add_line('number of params (M): %.2f' % (n_parameters / 1.e6))
    # sync bn if used
    if args.distributed and cfg['sync_bn']:
        if has_batchnorms(model):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # distribute the model
    model = model.cuda(args.gpu)
    model, _, cfg['dataset']['batch_size'], cfg['num_workers'] = environ.distribute_model_to_cuda(models=model, 
                                                                     args=args, 
                                                                     batch_size=cfg['dataset']['batch_size'], 
                                                                     num_workers=cfg['num_workers'], 
                                                                     ngpus_per_node=ngpus_per_node)

    # transformations
    train_transformations = get_vid_aug(name=cfg['dataset']['vid_transform'],
                                    crop_size=cfg['dataset']['crop_size'],
                                    num_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'],
                                    mode=cfg['dataset']['train']['aug_mode'],                                    
                                    aug_kwargs=cfg['dataset']['train']['vid_aug_kwargs'], 
                                    batch_multiplier=cfg['dataset']['batch_multiplier'] if 'batch_multiplier' in cfg['dataset'] else 0, 
                                    )

    val_transformations = get_vid_aug(name=cfg['dataset']['vid_transform'],
                                    crop_size=cfg['dataset']['crop_size'],
                                    num_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'],
                                    mode=cfg['dataset']['test']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['test']['vid_aug_kwargs'])  

    # dataset
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=train_transformations, 
                                split='train')

    val_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=val_transformations, 
                                split='test')
                
    if args.debug:
        train_dataset = FetchSubset(train_dataset, cfg['dataset']['batch_size']*64)
        val_dataset = FetchSubset(val_dataset, cfg['dataset']['batch_size']*8)
        
    # adjusting as test is done in dense mode
    test_batch_size = max(cfg['dataset']['batch_size'] // (3*cfg['dataset']['test']['clips_per_video']), 1)
    logger.add_line(f'test batch size is {test_batch_size*args.world_size}')
    args.effective_batch = cfg['dataset']['batch_size']*args.world_size
    logger.add_line(f"train batch size is {args.effective_batch}")
    logger.add_line(f'Training dataset size: {len(train_dataset)} - Validation dataset size: {len(val_dataset)}')
              
    # dataloader
    train_loader = dataloader.make_dataloader(dataset=train_dataset, 
                                              batch_size=cfg['dataset']['batch_size'],
                                              use_shuffle=cfg['dataset']['train']['use_shuffle'],
                                              drop_last=cfg['dataset']['train']['drop_last'],
                                              num_workers=cfg['num_workers'],
                                              distributed=args.distributed)
    
    test_loader = dataloader.make_dataloader(dataset=val_dataset, 
                                          batch_size=test_batch_size,
                                          use_shuffle=cfg['dataset']['test']['use_shuffle'],
                                          drop_last=cfg['dataset']['test']['drop_last'],
                                          num_workers=cfg['num_workers'],
                                          distributed=args.distributed)
        
        
    # optim; pass just classifier params
    if cfg['hyperparams']['optimizer']['name']=='sgd':
        optimizer = torch.optim.SGD(model.module.head.parameters(), 
                                    lr=1e-3, # setting lr through lr scheduler
                                    momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
        
    elif cfg['hyperparams']['optimizer']['name']=='adam':
        optimizer = torch.optim.Adam(model.module.head.parameters(), 
                                    lr=1e-3, # setting lr through lr scheduler
                                    betas=cfg['hyperparams']['optimizer']['betas'],
                                    # momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
        
    elif cfg['hyperparams']['optimizer']['name']=='adamw':
        optimizer = torch.optim.AdamW(model.module.head.parameters(), 
                                    lr=1e-3, # setting lr through lr scheduler
                                    betas=cfg['hyperparams']['optimizer']['betas'],
                                    # momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
            
        
        
    else:
        raise NotImplementedError()
    # lr scheduler
    if cfg['hyperparams']['lr']['name'] == 'cosine':
        lr_scheduler = warmup_cosine_scheduler(base_lr=cfg['hyperparams']['lr']['base_lr'], 
                                        final_lr=cfg['hyperparams']['lr']['final_lr'], 
                                        num_epochs=cfg['hyperparams']['num_epochs'], 
                                        iter_per_epoch=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['lr']['warmup_epochs'], 
                                        warmup_lr=cfg['hyperparams']['lr']['warmup_lr'])
    elif cfg['hyperparams']['lr']['name'] == 'fixed':
        # iters = cfg['hyperparams']['num_epochs'] * len(train_loader)
        # lr_scheduler = np.ones(iters) * cfg['hyperparams']['lr']['base_lr']
        lr_scheduler = warmup_fixed_scheduler(warmup_epochs=cfg['hyperparams']['lr']['warmup_epochs'], 
                                              warmup_lr=cfg['hyperparams']['lr']['warmup_lr'], 
                                              num_epochs=cfg['hyperparams']['num_epochs'], 
                                              base_lr=cfg['hyperparams']['lr']['base_lr'], 
                                              iter_per_epoch=len(train_loader))
    elif cfg['hyperparams']['lr']['name'] == 'step':
        lr_scheduler = warmup_multistep_scheduler(base_lr=cfg['hyperparams']['lr']['base_lr'], 
                                        milestones=cfg['hyperparams']['lr']['milestones'], 
                                        gamma=cfg['hyperparams']['lr']['gamma'],                                         
                                        num_epochs=cfg['hyperparams']['num_epochs'], 
                                        iter_per_epoch=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['lr']['warmup_epochs'], 
                                        warmup_lr=cfg['hyperparams']['lr']['warmup_lr'])
    else:
        raise NotImplementedError()

    # use apex for mixed precision training
    if cfg['apex']:
        amp = torch.cuda.amp.GradScaler() 
    else:
        amp=None
    
    model, optimizer, start_epoch, amp, rng = create_or_restore_training_state3(args, model, optimizer, logger, amp)
        
    # Start training
    end_epoch = cfg['hyperparams']['num_epochs']
    logger.add_line('='*30 + ' Training Started Finetune' + '='*30)
    
    best_top1, best_top5=0, 0
    best_epoch=start_epoch
    tr_top1, tr_top5 = 0, 0
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        fwd_kwargs = cfg['model']['fwd_kwargs']
            
        # train
        tr_top1, tr_top5 = run_phase('train', train_loader, model, fwd_kwargs,
                                     optimizer, lr_scheduler, amp,
                                     epoch, args, logger, tb_writter, wandb_writter, cfg['progress']['print_freq'])
        
        
        if args.rank==0:
            logger.add_line('saving model')    
            commit_state3(args, model, optimizer, epoch, amp, rng, logger)
            
        # test
        acc_top1, acc_top5 = run_phase('test_dense', test_loader, model, fwd_kwargs,
                                       optimizer, lr_scheduler, amp,
                                       epoch, args, logger, tb_writter, wandb_writter, print_freq=cfg['progress']['print_freq'])
        top1, top5 = acc_top1, acc_top5

        if top1>best_top1:
            best_top1=top1
            best_top5=top5
            best_epoch=epoch
            # # Save checkpoint
            # if args.rank==0:
            #     torch.save(model, os.path.join(args.ckpt_dir, "best_model.pth.tar"))
           
        if tb_writter is not None:
            tb_writter.add_scalar('train/tr_top1', tr_top1, epoch)
            tb_writter.add_scalar('eval/te_top1', acc_top1, epoch)
            tb_writter.add_scalar('eval/te_top5', acc_top5, epoch)
                        
        if wandb_writter is not None:
            wandb_writter.log({'train/tr_top1': tr_top1, 'custom_step': epoch})
            wandb_writter.log({'eval/te_top1': acc_top1, 'custom_step': epoch})
            wandb_writter.log({'eval/te_top5': acc_top5, 'custom_step': epoch})
                      
    if args.rank==0:
        logger.add_line(f'Final Acc - top1: {top1} - top5: {top5}')
        logger.add_line(f'Best Acc at epoch {best_epoch} - best_top1: {best_top1} - best_top5: {best_top5}')
    if wandb_writter is not None:
        wandb_writter.log({'eval/acc_top1_best': best_top1, 'custom_step': 0})
        wandb_writter.log({'eval/acc_top5_best': best_top5, 'custom_step': 0})

    torch.cuda.empty_cache()
    if wandb_writter is not None:
        wandb_writter.finish()
        
    return

def run_phase(phase, loader, model, fwd_kwargs, optimizer, lr_scheduler, amp,
              epoch, args, logger, tb_writter, wandb_writter, print_freq):
    
    logger.add_line('\n {}: Epoch {}'.format(phase, epoch))
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    gpu_meter = AverageMeter('GPU', ':4.2f')
    progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, top1_meters, top5_meters, gpu_meter,], phase=phase, epoch=epoch, logger=logger)


    if phase == 'train':
        model.train()
        LOG_HEAD='train'
    else:
        model.eval()
        LOG_HEAD='eval'

    end = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)
        
        # update lr during training
        if phase =='train':
            step = epoch * len(loader) + it
            for pi, param_group in enumerate(optimizer.param_groups):
                param_group["lr"] = lr_scheduler[step]
                # if pi == 1:  # only the second group is regularized; first group has bias and norms
                #     param_group["weight_decay"] = wd_scheduler[step]
                    

        # prepare data
        video = sample['frames']
        target = sample['label'].cuda()
        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)

        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[0], video.shape[1]
            video = video.flatten(0, 1).contiguous()
            
        # elif phase == 'train':
        #     if len(video.shape)==6: # tackling batch_multiplier
        #         batch_size, batch_multiplier = video.shape[0], video.shape[1]
        #         video = video.flatten(0, 1).contiguous()
        #         target = target.unsqueeze(1).repeat(1, batch_multiplier).view(-1)

        # compute outputs
        if phase == 'train':
            optimizer.zero_grad()
            if amp is not None:
                with torch.cuda.amp.autocast():
                    logits = model(video, **fwd_kwargs)
            else:
                logits = model(video, **fwd_kwargs)
        else:
            with torch.no_grad():
                if amp is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(video, **fwd_kwargs)
                else:
                    logits = model(video, **fwd_kwargs)
                        
        # compute loss and measure accuracy
        if phase == 'test_dense':
            confidence = softmax(logits).view(batch_size, clips_per_sample, -1).mean(1)
            target_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
            loss = criterion(logits, target_tiled)
        else:
            confidence = softmax(logits)
            loss = criterion(logits, target)

        with torch.no_grad():
            acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
            loss_meters.update(loss.item(), target.size(0))
            top1_meters.update(acc1[0].item(), target.size(0))
            top5_meters.update(acc5[0].item(), target.size(0))

        # compute gradient
        if phase == 'train':
            if amp is not None:
                amp.scale(loss).backward()
                amp.step(optimizer)
                amp.update()
            else:
                loss.backward()
                optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure gpu usage
        gpu_meter.update(torch.cuda.max_memory_allocated()/GB) 
        
        # log
        step = epoch * len(loader) + it
        if (it + 1) % print_freq == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)
            if tb_writter is not None:
                tb_writter.add_scalar(f'{LOG_HEAD}/LR', optimizer.param_groups[0]['lr'], step)
                for meter in progress.meters:
                    tb_writter.add_scalar(f'{LOG_HEAD}/{meter.name}', meter.val, step)
            
            if wandb_writter is not None and phase == 'train':
                wandb_writter.log({f'{LOG_HEAD}/LR': optimizer.param_groups[0]['lr'], 'custom_step': step})
                for meter in progress.meters:
                     wandb_writter.log({f'{LOG_HEAD}/{meter.name}': meter.val, 'custom_step': step})
            

    if args.distributed:
        progress.synchronize_meters_custom(args.gpu)
        progress.display(len(loader) * args.world_size)
            
    torch.cuda.empty_cache()
    return top1_meters.avg, top5_meters.avg