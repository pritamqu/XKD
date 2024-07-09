# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import os
import torch
import time
import numpy as np
from engine import AverageMeter, ProgressMeter, accuracy, warmup_multistep_scheduler, warmup_cosine_scheduler, Classifier
from tools import mean_ap_metric, synchronize_holder
from datasets.augmentations import get_aud_aug
from datasets import get_dataset, dataloader, FetchSubset        
from tools import environment as environ
# from tools.utils import resume_model, save_checkpoint
from checkpointing import commit_state3, create_or_restore_training_state3
from models.modules.vit_audio2 import AudioViT
from models import has_batchnorms
GB = (1024*1024*1024)
from collections import OrderedDict
import copy
from einops import rearrange
import torch.nn as nn
from sklearn import metrics


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
    # model.head = nn.Linear(model.embed_dim, cfg['model']['classifier']['num_classes']) 
    # model.head.weight.data.normal_(mean=0.0, std=0.01)
    # model.head.bias.data.zero_()   
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
                
    # if args.debug:
    #     train_dataset = FetchSubset(train_dataset, cfg['dataset']['batch_size']*4)
    #     val_dataset = FetchSubset(val_dataset, cfg['dataset']['batch_size']*2)
        
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
        iters = cfg['hyperparams']['num_epochs'] * len(train_loader)
        lr_scheduler = np.ones(iters) * cfg['hyperparams']['lr']['base_lr']
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
    
    best_map=0
    best_map_dict=None
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        fwd_kwargs = cfg['model']['fwd_kwargs']
            
        tr_mAP_dict = run_phase('train', train_loader, model, fwd_kwargs,
                                        optimizer, lr_scheduler, amp,
                                        epoch, args, logger, tb_writter, wandb_writter, cfg['progress']['print_freq'], 
                                        )
            
        
        if args.rank==0:
            logger.add_line('saving model')
            commit_state3(args, model, optimizer, epoch, amp, rng, logger)
            

        te_mAP_dict = run_phase('test_dense', test_loader, model, fwd_kwargs,
                                        optimizer, lr_scheduler, amp,
                                        epoch, args, logger, tb_writter, wandb_writter, print_freq=cfg['progress']['print_freq'], 
                                        ensemble=cfg['dataset']['test']['ensemble'])

        if te_mAP_dict is not None:
            for k in te_mAP_dict:
                logger.add_line(f'fine_tune_epoch- {epoch}: {k}:{te_mAP_dict[k]}')

        if te_mAP_dict['mean_ap']>best_map:
            best_map_dict=te_mAP_dict
            # # Save checkpoint
            # if args.rank==0:
            #     torch.save(model, os.path.join(args.ckpt_dir, "best_model.pth.tar"))
           
        if tb_writter is not None:
            if tr_mAP_dict is not None:
                for k in tr_mAP_dict:
                    tb_writter.add_scalar(f'train/{k}', tr_mAP_dict[k], epoch)
            if te_mAP_dict is not None:
                for k in te_mAP_dict:
                    tb_writter.add_scalar(f'eval/{k}', te_mAP_dict[k], epoch)
                        
        if wandb_writter is not None:
            if tr_mAP_dict is not None:
                for k in tr_mAP_dict:
                    wandb_writter.log({f'train/{k}': tr_mAP_dict[k], 'custom_step': epoch})
            if te_mAP_dict is not None:
                for k in te_mAP_dict:
                    wandb_writter.log({f'eval/{k}': te_mAP_dict[k], 'custom_step': epoch})
                      
    if args.rank==0:
        for k in best_map_dict:
            logger.add_line(f'Best {k}: {best_map_dict[k]}')
            if wandb_writter is not None:
                wandb_writter.log({f'eval/{k}_best': best_map_dict[k], 'custom_step': 0})
            if tb_writter is not None:
                tb_writter.add_scalar(f'eval/{k}_best', best_map_dict[k], 0)

    torch.cuda.empty_cache()
    if wandb_writter is not None:
        wandb_writter.finish()
        
    return

def run_phase(phase, loader, model, fwd_kwargs, optimizer, lr_scheduler, amp,
              epoch, args, logger, tb_writter, wandb_writter, print_freq,
              ensemble=None):
    
    logger.add_line('\n {}: Epoch {}'.format(phase, epoch))
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    # top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    # top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    gpu_meter = AverageMeter('GPU', ':4.2f')
    progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, 
                                                  # top1_meters, top5_meters, 
                                                  gpu_meter,], phase=phase, epoch=epoch, logger=logger)


    if phase == 'train':
        model.train()
        LOG_HEAD='train'
    else:
        model.eval()
        LOG_HEAD='eval'
        
    mAP_dict=None
    pred_holder = []
    target_holder = []

    end = time.time()
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCEWithLogitsLoss()
    sigmoid = torch.nn.Sigmoid()
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
        spec = sample['audio']
        target = sample['label'].cuda()
        if args.gpu is not None:
            spec = spec.cuda(args.gpu, non_blocking=True)

        if phase == 'test_dense':
            batch_size, clips_per_sample = spec.shape[0], spec.shape[1]
            spec = spec.flatten(0, 1).contiguous()

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
                        
        # making sure their types are same.
        target = target.type(logits.dtype)
        # compute loss and measure accuracy
        if phase == 'test_dense':
            confidence = sigmoid(logits).view(batch_size, clips_per_sample, -1)
            if ensemble=='max':
                confidence = torch.max(confidence, dim=1)[0]
            elif ensemble=='sum':
                confidence = torch.sum(confidence, dim=1)
            elif ensemble=='mean':
                confidence = torch.mean(confidence, dim=1)
            else:
                raise ValueError(f'unknown ensemble method: {ensemble}')
            
            # making the same shape as logits
            target_tiled = target.unsqueeze(1).repeat(1, clips_per_sample, 1).view(batch_size*clips_per_sample, -1)
            loss = criterion(logits, target_tiled)
            pred_holder.append(confidence.detach())
            target_holder.append(target.detach())
        else:
            confidence = sigmoid(logits)
            # confidence = logits
            loss = criterion(logits, target)

        with torch.no_grad():
            # acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
            loss_meters.update(loss.item(), target.size(0))
            # top1_meters.update(acc1[0].item(), target.size(0))
            # top5_meters.update(acc5[0].item(), target.size(0))

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
            
    # calculating results during test
    if phase == 'test_dense':
        pred_holder = torch.vstack(pred_holder)
        target_holder = torch.vstack(target_holder)
        if args.distributed:
            pred_holder = synchronize_holder(pred_holder, args.gpu)
            target_holder = synchronize_holder(target_holder, args.gpu)
        else:
            pred_holder = [pred_holder]
            target_holder = [target_holder]

        pred_holder = torch.cat(pred_holder).cpu().numpy()
        target_holder = torch.cat(target_holder).cpu().numpy()   
        mAP_dict = mean_ap_metric(predicts=pred_holder, targets=target_holder)
        
    if args.distributed:
        progress.synchronize_meters_custom(args.gpu)
        progress.display(len(loader) * args.world_size)
        
    # if args.rank==0:
    #     logger.add_line(f'aud_phase: {phase} - epoch: {epoch} - mAP: {mAP}')
        
            
    torch.cuda.empty_cache()
    return mAP_dict
