# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import os
import torch
import time
import numpy as np
import torch.nn as nn
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
import math
import sys
from engine import AverageMeter, ProgressMeter, accuracy, cosine_scheduler, get_params_groups
from datasets.augmentations import get_aud_aug
from datasets import get_dataset, dataloader, FetchSubset        
from tools import environment as environ
# from tools.utils import resume_model, save_checkpoint
from checkpointing import create_or_restore_training_state3, commit_state3
from models import LinearClassifier, has_batchnorms
from models.modules.vit_audio2 import AudioViT
GB = (1024*1024*1024)
from collections import OrderedDict
import copy
from einops import rearrange


def finetune(args, cfg, backbone_state_dict, ngpus_per_node, logger, tb_writter, wandb_writter):
    
    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)
                
    # get the model
    model=AudioViT(**cfg['model']['backbone'])    
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
        # elif key.startswith('cuboid_embed'):
        #     state[key.replace('cuboid_embed', 'patch_embed')] = val
        else:
            state[key] = val
               
    # load weights
    model.load_state_dict(state, strict=True)
    # del backbone_state_dict
    # configure head.
    if cfg['model']['classifier']['use_bn']:
        bn = nn.BatchNorm1d(model.embed_dim, affine=False, eps=1e-6)
        linear = nn.Linear(model.embed_dim, cfg['model']['classifier']['num_classes'])
        linear.weight.data.normal_(mean=0.0, std=0.01)
        linear.bias.data.zero_()
        model.head = nn.Sequential(bn, linear)
    else:
        model.head = nn.Linear(model.embed_dim, cfg['model']['classifier']['num_classes']) 
        model.head.weight.data.normal_(mean=0.0, std=0.01)
        model.head.bias.data.zero_()
    
    # # freeze all but the classifier head for linear eval
    # for _, p in model.named_parameters():
    #     p.requires_grad = False
    # for _, p in model.head.named_parameters():
    #     p.requires_grad = True
     
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
    train_transformations = get_aud_aug(name=cfg['dataset']['aud_transform'],
                                    audio_fps=cfg['dataset']['audio_fps'],
                                    n_mels=cfg['dataset']['n_mels'],
                                    duration=cfg['dataset']['audio_clip_duration'],
                                    n_fft=cfg['dataset']['n_fft'],
                                    hop_length=cfg['dataset']['hop_length'],
                                    mode=cfg['dataset']['train']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['train']['aud_aug_kwargs'])

    val_transformations = get_aud_aug(name=cfg['dataset']['aud_transform'],
                                    audio_fps=cfg['dataset']['audio_fps'],
                                    n_mels=cfg['dataset']['n_mels'],
                                    duration=cfg['dataset']['audio_clip_duration'],
                                    n_fft=cfg['dataset']['n_fft'],
                                    hop_length=cfg['dataset']['hop_length'],
                                    mode=cfg['dataset']['test']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['test']['aud_aug_kwargs'])

    # dataset
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                audio_transform=train_transformations, 
                                split='train')

    val_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                audio_transform=val_transformations, 
                                split='test')
                
    if args.debug:
        train_dataset = FetchSubset(train_dataset, cfg['dataset']['batch_size']*64)
        val_dataset = FetchSubset(val_dataset, cfg['dataset']['batch_size']*8)
        
    # adjusting as test is done in dense mode
    test_batch_size = max(cfg['dataset']['batch_size'] // cfg['dataset']['test']['clips_per_video'], 2)
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
    # mixup
    mixup_fn = None
    mixup_active = cfg['hyperparams']['mixup'] > 0 or cfg['hyperparams']['cutmix'] > 0. or cfg['hyperparams']['cutmix_minmax'] is not None
    if mixup_active:
        logger.add_line("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=cfg['hyperparams']['mixup'], 
            cutmix_alpha=cfg['hyperparams']['cutmix'], 
            cutmix_minmax=cfg['hyperparams']['cutmix_minmax'],
            prob=cfg['hyperparams']['mixup_prob'], switch_prob=cfg['hyperparams']['mixup_switch_prob'], mode=cfg['hyperparams']['mixup_mode'],
            label_smoothing=cfg['hyperparams']['label_smoothing'], num_classes=cfg['model']['classifier']['num_classes'])
        
    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif cfg['hyperparams']['label_smoothing'] > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=cfg['hyperparams']['label_smoothing'])
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    test_criterion = torch.nn.CrossEntropyLoss()
    	
	# optim
    param_groups = get_params_groups(model.module, 
                                     weight_decay=cfg['hyperparams']['weight_decay'], 
                                     no_weight_decay_list=model.module.no_weight_decay(), 
                                     layer_decay=cfg['hyperparams']['layer_decay'])
        
    if cfg['hyperparams']['optimizer']['name']=='sgd':
        optimizer = torch.optim.SGD(param_groups, 
                                    lr=1e-3, # setting lr through lr scheduler
                                    momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
        
    elif cfg['hyperparams']['optimizer']['name']=='adam':
        optimizer = torch.optim.Adam(param_groups, 
                                    lr=1e-3, # setting lr through lr scheduler
                                    betas=cfg['hyperparams']['optimizer']['betas'],
                                    # momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
        
    elif cfg['hyperparams']['optimizer']['name']=='adamw':
        optimizer = torch.optim.AdamW(param_groups, 
                                    lr=1e-3, # setting lr through lr scheduler
                                    betas=cfg['hyperparams']['optimizer']['betas'],
                                    # momentum=cfg['hyperparams']['optimizer']['momentum'],
                                    weight_decay=cfg['hyperparams']['optimizer']['weight_decay']) 
            
        
        
    else:
        raise NotImplementedError()
    # lr scheduler
    if cfg['hyperparams']['lr']['name'] == 'cosine':
        lr_scheduler = cosine_scheduler(base_value=cfg['hyperparams']['lr']['base_lr'], 
                                        final_value=cfg['hyperparams']['lr']['final_lr'], 
                                        epochs=cfg['hyperparams']['num_epochs'], 
                                        niter_per_ep=len(train_loader), 
                                        warmup_epochs=cfg['hyperparams']['lr']['warmup_epochs'], 
                                        start_warmup_value=cfg['hyperparams']['lr']['warmup_lr'])
    elif cfg['hyperparams']['lr']['name'] == 'fixed':
        iters = cfg['hyperparams']['num_epochs'] * len(train_loader)
        lr_scheduler = np.ones(iters) * cfg['hyperparams']['lr']['base_lr']
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
        marker = time.time()
        tr_top1, tr_top5 = run_phase(phase='train', 
                                      loader=train_loader, 
                                      model=model,
                                      optimizer=optimizer, 
                                      criterion=criterion, 
                                      mixup_fn=mixup_fn,
                                      fwd_kwargs=fwd_kwargs,
                                      lr_scheduler=lr_scheduler, 
                                      wd_scheduler=None, 
                                      amp=amp,
                                      epoch=epoch, 
                                      args=args, logger=logger, 
                                      tb_writter=tb_writter, wandb_writter=wandb_writter, 
                                      print_freq=cfg['progress']['print_freq'])     
        
        logger.add_line(f'trainng took {time.time()-marker} seconds')
        marker = time.time()

        if args.rank==0:
            logger.add_line('saving model')    
            commit_state3(args, model, optimizer, epoch, amp, rng, logger)

        logger.add_line(f'model saving took {time.time()-marker} seconds')
        marker = time.time()
        
        if tb_writter is not None:
            tb_writter.add_scalar('fine_tune_epoch/tr_top1', tr_top1, epoch)
            tb_writter.add_scalar('fine_tune_epoch/tr_top5', tr_top5, epoch)
        if wandb_writter is not None:
            wandb_writter.log({'fine_tune_epoch/tr_top1': tr_top1, 'custom_step': epoch})
            wandb_writter.log({'fine_tune_epoch/tr_top5': tr_top5, 'custom_step': epoch})
            
        # test
        if (epoch+1) % cfg['eval_freq'] == 0 or (epoch+1) == end_epoch:
            vid_top1, vid_top5 = run_phase(phase='test_dense', 
                                         loader=test_loader, 
                                         model=model,
                                         optimizer=None, 
                                         criterion=test_criterion, # criterion
                                         mixup_fn=None,
                                         fwd_kwargs=fwd_kwargs,
                                         lr_scheduler=None, 
                                         wd_scheduler=None, 
                                         amp=amp,
                                         epoch=epoch, 
                                         args=args, logger=logger, 
                                         tb_writter=tb_writter, wandb_writter=wandb_writter, 
                                         print_freq=cfg['progress']['print_freq'])

            logger.add_line(f'validation took {time.time()-marker} seconds')
            top1, top5 = vid_top1, vid_top5
            if top1>best_top1:
                best_top1=top1
                best_top5=top5
                best_epoch=epoch
                # # Save checkpoint
                # if args.rank==0:
                #     torch.save(model, os.path.join(args.ckpt_dir, "best_model.pth.tar"))
                
            if tb_writter is not None:
                tb_writter.add_scalar('fine_tune_epoch/vid_top1', vid_top1, epoch)
                tb_writter.add_scalar('fine_tune_epoch/vid_top5', vid_top5, epoch)
            if wandb_writter is not None:
                wandb_writter.log({'fine_tune_epoch/vid_top1': vid_top1, 'custom_step': epoch})
                wandb_writter.log({'fine_tune_epoch/vid_top5': vid_top5, 'custom_step': epoch})
                  
    # --------- end log               
    if args.rank==0:
        logger.add_line(f'Final Acc - top1: {vid_top1} - top5: {vid_top5}')
        logger.add_line(f'Best Acc at epoch {best_epoch} - best_top1: {best_top1} - best_top5: {best_top5}')
    if wandb_writter is not None:
        wandb_writter.log({'fine_tune_epoch/vid_top1_best': best_top1, 'custom_step': 0})
        wandb_writter.log({'fine_tune_epoch/vid_top5_best': best_top5, 'custom_step': 0})
    if tb_writter is not None:
        tb_writter.add_scalar('fine_tune_epoch/vid_top1_best', best_top1, 0)
        tb_writter.add_scalar('fine_tune_epoch/vid_top5_best', best_top5, 0)

    torch.cuda.empty_cache()
    if wandb_writter is not None:
        wandb_writter.finish()
        
    return

def run_phase(phase, loader, model, optimizer, criterion, mixup_fn, fwd_kwargs,
              lr_scheduler, wd_scheduler, amp,
              epoch, args, logger, tb_writter, wandb_writter, print_freq):
    
    logger.add_line('\n {}: Epoch {}'.format(phase, epoch))
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    max_lr_meter = AverageMeter('Max_LR', ':.4e', 0)
    min_lr_meter = AverageMeter('Min_LR', ':.4e', 0)
    weight_decay_meter = AverageMeter('WD', ':.4e', 0)
    top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    gpu_meter = AverageMeter('GPU', ':4.2f')
    if phase=='train':
        progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, 
                                                      max_lr_meter, min_lr_meter, weight_decay_meter,
                                                      top1_meters, top5_meters, gpu_meter,], phase=phase, epoch=epoch, logger=logger)
    else:
        progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, 
                                                      top1_meters, top5_meters,], phase=phase, epoch=epoch, logger=logger)


    if phase == 'train':
        model.train()
    else:
        model.eval()

    end = time.time()
    softmax = torch.nn.Softmax(dim=1)
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)
        
        step = epoch * len(loader) + it
        # update lr during training
        if phase =='train':
            
            for param_group in optimizer.param_groups:
                if "lr_scale" in param_group:
                    param_group["lr"] = lr_scheduler[step] * param_group["lr_scale"]
                else:
                    param_group["lr"] = lr_scheduler[step]
                if wd_scheduler is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_scheduler[step]

        # prepare data
        spec = sample['audio']
        target = sample['label'].cuda()
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)

        if phase == 'test_dense':
            batch_size, clips_per_sample = spec.shape[0], spec.shape[1]
            spec = spec.flatten(0, 1).contiguous()

        elif phase == 'train':
            if mixup_fn is not None:
                spec, target = mixup_fn(spec, target)
                
        # compute outputs
        if phase == 'train':
            optimizer.zero_grad()
            if amp is not None:
                with torch.cuda.amp.autocast():
                    logits = model(spec, **fwd_kwargs)
            else:
                logits = model(spec, **fwd_kwargs)
        else:
            with torch.no_grad():
                if amp is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(spec, **fwd_kwargs)
                else:
                    logits = model(spec, **fwd_kwargs)
                        
        # compute loss and measure accuracy
        if phase == 'test_dense':
            confidence = softmax(logits).view(batch_size, clips_per_sample, -1).mean(1)
            target_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
            if criterion is not None:
                loss = criterion(logits, target_tiled)
        else:
            confidence = softmax(logits)
            loss = criterion(logits, target)
            
        if phase == 'train':
            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training") # for log
                logger.add_line(f"Loss is {loss.item()}, stopping training") # for logger
                sys.exit(1)
            
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
        
        # with torch.no_grad():
        if phase == 'train':
            loss_meters.update(loss.item(), target.size(0))
            if mixup_fn is not None:
                # if mixup is enabled, skipping acc calculation during training
                acc1, acc5 = torch.tensor([0.]).cuda(args.gpu), torch.tensor([0.]).cuda(args.gpu)
            else:
                acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
        elif phase == 'test_dense':
            if criterion is not None:
                loss_meters.update(loss.item(), target.size(0))
            acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
        
        top1_meters.update(acc1[0].item(), target.size(0))
        top5_meters.update(acc5[0].item(), target.size(0))

        if phase == 'train':
            # lr meter
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
    
            max_lr_meter.update(max_lr, target.size(0))
            min_lr_meter.update(min_lr, target.size(0))
            
            # wd meter
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            # metric_logger.update(weight_decay=weight_decay_value)
            # metric_logger.update(grad_norm=grad_norm)
            weight_decay_meter.update(weight_decay_value, target.size(0))
        
        # log
        if (it + 1) % print_freq == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)
            if tb_writter is not None :
                if phase == 'train':
                #     tb_writter.add_scalar(f'{phase}_fine_tune_iter/LR', optimizer.param_groups[0]['lr'], step)
                #     tb_writter.add_scalar(f'{phase}_fine_tune_iter/WD', optimizer.param_groups[1]['weight_decay'], step)
                    for meter in progress.meters:
                        tb_writter.add_scalar(f'{phase}_fine_tune_iter/{meter.name}', meter.val, step)
            
            if wandb_writter is not None : 
                if phase == 'train':
                #     wandb_writter.log({f'{phase}_fine_tune_iter/LR': optimizer.param_groups[0]['lr'], 'custom_step': step})
                #     wandb_writter.log({f'{phase}_fine_tune_iter/WD': optimizer.param_groups[1]['weight_decay'], 'custom_step': step})
                    for meter in progress.meters:
                         wandb_writter.log({f'{phase}_fine_tune_iter/{meter.name}': meter.val, 'custom_step': step})
            
            
    if args.distributed:
        progress.synchronize_meters_custom(args.gpu)
        progress.display(len(loader) * args.world_size)
            
    torch.cuda.empty_cache()
    return top1_meters.avg, top5_meters.avg