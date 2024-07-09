# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 23:56:29 2021

@author: pritam
"""
import os
import torch
from engine import Feature_Bank_AV, set_grad
from datasets.augmentations import get_vid_aug, get_aud_aug
from datasets import get_dataset, dataloader, FetchSubset        
from tools import environment as environ
from models import get_backbone
import numpy as np
from collections import defaultdict
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import time
import pickle
from models import VideoViT, AudioViT, SVMWrapper
from collections import OrderedDict

def linear_svm(args, cfg, backbone_state_dict, logger, tb_writter, wandb_writter):
    
    global SEED
    SEED=args.seed
    
    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)
        
    # get model
    vid_model=SVMWrapper(backbone=VideoViT(**cfg['model']['vid_backbone']), 
                     feat_op=cfg['model']['fwd_kwargs']['feat_op'], use_amp=True)
    aud_model=SVMWrapper(backbone=AudioViT(**cfg['model']['aud_backbone']), 
                     feat_op=cfg['model']['fwd_kwargs']['feat_op'], use_amp=True)
        
    vid_state = OrderedDict()
    for key in backbone_state_dict['video']:
        val = backbone_state_dict['video'][key]
        if key.startswith('encoder.'):
            vid_state[key.replace('encoder.', '')] = val
        elif key.startswith('encoder_'):
            vid_state[key.replace('encoder_', '')] = val
        elif key == 'enc_pos_embed':
            vid_state['pos_embed'] = val
        elif key.startswith('cuboid_embed'):
            vid_state[key.replace('cuboid_embed', 'patch_embed')] = val
        else:
            vid_state[key] = val
    
    aud_state = OrderedDict()
    for key in backbone_state_dict['audio']:
        val = backbone_state_dict['audio'][key]
        if key.startswith('encoder.'):
            aud_state[key.replace('encoder.', '')] = val
        elif key.startswith('encoder_'):
            aud_state[key.replace('encoder_', '')] = val
        elif key == 'enc_pos_embed':
            aud_state['pos_embed'] = val
        else:
            aud_state[key] = val
                  
    ## load weights
    vid_model.backbone.load_state_dict(vid_state, strict=True)
    aud_model.backbone.load_state_dict(aud_state, strict=True)
    # set grad false
    set_grad(vid_model, requires_grad=False)
    set_grad(aud_model, requires_grad=False)
    # when extracting features it's important to set in eval mode
    vid_model.eval() 
    aud_model.eval()
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        vid_model = vid_model.cuda(args.gpu)
        aud_model = aud_model.cuda(args.gpu)
        # model = torch.nn.DataParallel(model)

    # dataset
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=get_vid_aug(name=cfg['dataset']['vid_transform'],
                                                                crop_size=cfg['dataset']['crop_size'],
                                                                num_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'],
                                                                mode=cfg['dataset']['train']['aug_mode'],
                                                                aug_kwargs=cfg['dataset']['train']['vid_aug_kwargs'])
                                ,
                                audio_transform=get_aud_aug(name=cfg['dataset']['aud_transform'],
                                                                audio_fps=cfg['dataset']['audio_fps'],
                                                                n_mels=cfg['dataset']['n_mels'],
                                                                duration=cfg['dataset']['audio_clip_duration'],
                                                                n_fft=cfg['dataset']['n_fft'],
                                                                hop_length=cfg['dataset']['hop_length'],
                                                                mode=cfg['dataset']['train']['aug_mode'],
                                                                aug_kwargs=cfg['dataset']['train']['aud_aug_kwargs'])
                                ,
                                split='train')

    val_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=get_vid_aug(name=cfg['dataset']['vid_transform'],
                                                                crop_size=cfg['dataset']['crop_size'],
                                                                num_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'],
                                                                mode=cfg['dataset']['test']['aug_mode'],
                                                                aug_kwargs=cfg['dataset']['test']['vid_aug_kwargs'])
                                ,
                                audio_transform=get_aud_aug(name=cfg['dataset']['aud_transform'],
                                                                audio_fps=cfg['dataset']['audio_fps'],
                                                                n_mels=cfg['dataset']['n_mels'],
                                                                duration=cfg['dataset']['audio_clip_duration'],
                                                                n_fft=cfg['dataset']['n_fft'],
                                                                hop_length=cfg['dataset']['hop_length'],
                                                                mode=cfg['dataset']['test']['aug_mode'],
                                                                aug_kwargs=cfg['dataset']['test']['aud_aug_kwargs'])
                                ,
                                split='test')        
            
    if args.debug:
        train_dataset = FetchSubset(train_dataset, cfg['dataset']['batch_size']*4)
        val_dataset = FetchSubset(val_dataset, cfg['dataset']['batch_size']*1)
        
    # adjusting as test is done in dense mode
    test_batch_size = max(cfg['dataset']['batch_size'] // cfg['dataset']['test']['clips_per_video'], 1)
    logger.add_line(f'test batch size is {test_batch_size}')
    logger.add_line(f'Training dataset size: {len(train_dataset)} - Validation dataset size: {len(val_dataset)}')
                    
    # dataloader
    train_loader = dataloader.make_dataloader(dataset=train_dataset, 
                                              batch_size=cfg['dataset']['batch_size'],
                                              use_shuffle=False,
                                              drop_last=False,
                                              num_workers=cfg['num_workers'],
                                              distributed=False)

    val_loader = dataloader.make_dataloader(dataset=val_dataset, 
                                          batch_size=test_batch_size,
                                          use_shuffle=False,
                                          drop_last=False,
                                          num_workers=cfg['num_workers'],
                                          distributed=False)

    feat_bank = Feature_Bank_AV(args.world_size, args.distributed, vnet=vid_model, anet=aud_model, 
                                logger=logger, mode='both', 
                                apply_l2=cfg['model']['apply_l2'], 
                                apply_bn=cfg['model']['apply_bn'], 
                                feat_dim=cfg['model']['feat_dim'])

    logger.add_line("computing features")
    end = time.time()
    train_vid_features, train_aud_features, train_labels, train_indices = feat_bank.fill_memory_bank(train_loader)
    logger.add_line(f'time spent for train feat extraction: {time.time() - end}')
    end = time.time()
    val_vid_features, val_aud_features, val_labels, val_indices = feat_bank.fill_memory_bank(val_loader)
    logger.add_line(f'time spent for val feat extraction: {time.time() - end}')

    train_vid_features, train_aud_features, train_labels, train_indices = train_vid_features.numpy(), train_aud_features.numpy(), train_labels.numpy(), train_indices.numpy()
    val_vid_features, val_aud_features, val_labels, val_indices = val_vid_features.numpy(), val_aud_features.numpy(), val_labels.numpy(), val_indices.numpy()
    
        
    # quick 
    train_features = np.empty((train_vid_features.shape[0], train_vid_features.shape[1]*2), dtype=train_vid_features.dtype)
    for k in range(train_vid_features.shape[0]):
        train_features[k] = np.concatenate((train_vid_features[k], train_aud_features[k]), axis=-1)

    val_features = np.empty((val_vid_features.shape[0], val_vid_features.shape[1], val_aud_features.shape[2]*2), dtype=val_vid_features.dtype)
    for k in range(val_vid_features.shape[0]):
        val_features[k] = np.concatenate((val_vid_features[k], val_aud_features[k]), axis=-1)

    ## save array
    # pickle.dump(train_features, open(os.path.join(args.log_dir, 'train_features.pkl'),'wb'))
    # pickle.dump(train_labels, open(os.path.join(args.log_dir, 'train_labels.pkl'),'wb'))
    # pickle.dump(train_indexs, open(os.path.join(args.log_dir, 'train_indexs.pkl'),'wb'))
    # pickle.dump(val_features, open(os.path.join(args.log_dir, 'val_features.pkl'),'wb'))
    # pickle.dump(val_labels, open(os.path.join(args.log_dir, 'val_labels.pkl'),'wb'))
    # pickle.dump(val_indexs, open(os.path.join(args.log_dir, 'val_indexs.pkl'),'wb'))
    

    best_top1=0.0
    logger.add_line("Running SVM...")
    logger.add_line(f"train_feat size: {train_features.shape}")     
    logger.add_line(f"val_feat size: {val_features.shape}")
    if isinstance(cfg['model']['svm']['cost'], list):
        for cost in cfg['model']['svm']['cost']:
            clip_top1, clip_top5 = _compute(cost, cfg, logger,
             train_features, train_labels, train_indices, 
             val_features, val_labels, val_indices,)
            
            if tb_writter is not None:
                tb_writter.add_scalar('Epoch/aud_svm_top1', clip_top1, cost)
                tb_writter.add_scalar('Epoch/aud_svm_top5', clip_top5, cost)

            if wandb_writter is not None:
                wandb_writter.log({'Epoch/aud_svm_top1': clip_top1, 'custom_step': cost})
                wandb_writter.log({'Epoch/aud_svm_top5': clip_top5, 'custom_step': cost})

            # show the best one
            if clip_top1 >= best_top1:
                best_top1 = clip_top1
                best_top5 = clip_top5 # top 1 matters 
    else:
        cost = cfg['model']['svm']['cost']
        best_top1, best_top5 = _compute(cost, cfg, logger,
             train_features, train_labels, train_indices, 
             val_features, val_labels, val_indices,)

    logger.add_line(f'Best Acc: top1: {best_top1} - top5: {best_top5}')
    # logger.add_line(f'total time spent for vid svm: {time.time() - end}')
    if tb_writter is not None:
        tb_writter.add_scalar('Epoch/audvid_svm_best1', best_top1, 1)
        tb_writter.add_scalar('Epoch/audvid_svm_best5', best_top5, 1)

    if wandb_writter is not None:
        wandb_writter.log({'Epoch/audvid_svm_best1': best_top1, 'custom_step': 1})
        wandb_writter.log({'Epoch/audvid_svm_best5': best_top5, 'custom_step': 1})
        
    torch.cuda.empty_cache()       
    return

def _compute(cost, cfg, logger,
             train_features, train_labels, train_indices, 
             val_features, val_labels, val_indices, 
             test_phase='test_dense'):
    
    # normalize
    if cfg['model']['svm']['scale_features']:   
        scaler = preprocessing.StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)   
        # val_features = scaler.transform(val_features)
    
    classifier = LinearSVC(C=cost, max_iter=cfg['model']['svm']['iter'], random_state=SEED)
    classifier.fit(train_features, train_labels.ravel())
    pred_train = classifier.decision_function(train_features)
    # for test dense, assuming this is default test case
    # reshape the data video --> cips
    if test_phase=='test_dense':
        total_samples, clips_per_sample = val_features.shape[0], val_features.shape[1]
        val_features = val_features.reshape(total_samples*clips_per_sample, -1)
    # scale if true
    if cfg['model']['svm']['scale_features']:
        val_features = scaler.transform(val_features)
    # predict
    pred_test = classifier.decision_function(val_features)
    if test_phase=='test_dense':
        pred_test = pred_test.reshape(total_samples, clips_per_sample, -1).mean(1)

    metrics = compute_accuracy_metrics(pred_train, train_labels[:, None], prefix='train_')
    metrics.update(compute_accuracy_metrics(pred_test, val_labels[:, None], prefix='test_'))
    logger.add_line(f"AudioVideo Linear SVM on {cfg['dataset']['name']} cost: {cost}")
    for metric in metrics:
        logger.add_line(f"{metric}: {metrics[metric]}") 
        
    return metrics['test_top1'], metrics['test_top5']
        
        
        
def compute_accuracy_metrics(pred, gt, prefix=''):
  order_pred = np.argsort(pred, axis=1)
  assert len(gt.shape) == len(order_pred.shape) == 2
  top1_pred = order_pred[:, -1:]
  top5_pred = order_pred[:, -5:]
  top1_acc = np.mean(top1_pred == gt)
  top5_acc = np.mean(np.max(top5_pred == gt, 1))
  return {prefix + 'top1': top1_acc*100,
          prefix + 'top5': top5_acc*100}