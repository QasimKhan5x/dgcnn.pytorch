#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_partseg.py
@Time: 2021/7/20 7:49 PM
"""


from __future__ import print_function

import argparse
import gc
import os
import sys

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from plyfile import PlyData, PlyElement
from sklearn import metrics
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data import ShapeNetPart_Augmented
from loss import cross_entropy
from models.model_partseg import Net
from util import IOStream

class_indexs = np.zeros((16,), dtype=int)
global visual_warning
visual_warning = True

class_choices = ['airplane', 'bag', 'cap', 'car', 'chair', 'earphone', 'guitar', 'knife', 'lamp', 'laptop', 'motorbike', 'mug', 'pistol', 'rocket', 'skateboard', 'table']
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]


def _init_():
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/'+args.exp_name):
        os.makedirs('outputs/'+args.exp_name)
    if not os.path.exists('outputs/'+args.exp_name+'/'+'models'):
        os.makedirs('outputs/'+args.exp_name+'/'+'models')
    if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'):
        os.makedirs('outputs/'+args.exp_name+'/'+'visualization')
    if not os.path.exists('outputs/'+args.exp_name+'/'+'checkpoints'):
        os.makedirs('outputs/'+args.exp_name+'/'+'checkpoints')
    if not os.path.exists('outputs/'+args.exp_name+'/'+'backups'):
        os.makedirs('outputs/'+args.exp_name+'/'+'backups')
    os.system('cp main_partseg.py outputs'+'/'+args.exp_name+'/'+'main_partseg.py.backup')
    os.system('cp loss.py outputs'+'/' +
              args.exp_name+'/'+'loss.py.backup')
    os.system('cp models/* outputs'+'/' +
              args.exp_name+'/'+'backups/')


def calculate_shape_IoU(pred_np, seg_np, label, class_choice, visual=False):
    if not visual:
        label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious


def visualization(visu, visu_format, data, pred, seg, label, partseg_colors, class_choice):
    global class_indexs
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        skip = False
        classname = class_choices[int(label[i])]
        class_index = class_indexs[int(label[i])]
        if visu[0] != 'all':
            if len(visu) != 1:
                if visu[0] != classname or visu[1] != str(class_index):
                    skip = True 
                else:
                    visual_warning = False
            elif visu[0] != classname:
                skip = True 
            else:
                visual_warning = False
        elif class_choice != None:
            skip = True
        else:
            visual_warning = False
        if skip:
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1
        else:  
            if not os.path.exists('outputs/'+args.exp_name+'/'+'visualization'+'/'+classname):
                os.makedirs('outputs/'+args.exp_name+'/'+'visualization'+'/'+classname)
            for j in range(0, data.shape[2]):
                RGB.append(partseg_colors[int(pred[i][j])])
                RGB_gt.append(partseg_colors[int(seg[i][j])])
            pred_np = []
            seg_np = []
            pred_np.append(pred[i].cpu().numpy())
            seg_np.append(seg[i].cpu().numpy())
            xyz_np = data[i].cpu().numpy()
            xyzRGB = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((xyz_np.transpose(1, 0), np.array(RGB_gt)), axis=1)
            IoU = calculate_shape_IoU(np.array(pred_np), np.array(seg_np), label[i].cpu().numpy(), class_choice, visual=True)
            IoU = str(round(IoU[0], 4))
            filepath = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_pred_'+IoU+'.'+visu_format
            filepath_gt = 'outputs/'+args.exp_name+'/'+'visualization'+'/'+classname+'/'+classname+'_'+str(class_index)+'_gt.'+visu_format
            if visu_format=='txt':
                np.savetxt(filepath, xyzRGB, fmt='%s', delimiter=' ') 
                np.savetxt(filepath_gt, xyzRGB_gt, fmt='%s', delimiter=' ') 
                print('TXT visualization file saved in', filepath)
                print('TXT visualization file saved in', filepath_gt)
            elif visu_format=='ply':
                xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3], xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
                xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3], xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath)
                vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                PlyData([vertex]).write(filepath_gt)
                print('PLY visualization file saved in', filepath)
                print('PLY visualization file saved in', filepath_gt)
            else:
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' % \
                (visu_format))
                exit()
            class_indexs[int(label[i])] = class_indexs[int(label[i])] + 1


def prepare_dl(dataset, drop_last, batch_size, pin_memory, num_workers=0):
    '''split the dataloader to each process in the process group'''
    sampler = DistributedSampler(dataset, drop_last=drop_last)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=pin_memory)
    return dataloader


def train(args, io):
    train_ds = ShapeNetPart_Augmented(
        partition="trainval", append_height=args.use_height)
    test_ds = ShapeNetPart_Augmented(
        partition="test", append_height=args.use_height)
    
    drop_last = len(train_ds) >= 100
    ngpus_per_node = torch.cuda.device_count()
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.test_batch_size = int(args.test_batch_size / ngpus_per_node)

    train_loader = prepare_dl(train_ds, drop_last=drop_last, batch_size=args.batch_size, pin_memory=True)
    test_loader = prepare_dl(test_ds, drop_last=False, batch_size=args.test_batch_size, pin_memory=False)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)

    # Try to load models
    seg_num_all = 50
    seg_start_index = 0
    if args.model == 'transformer':
        model = Net(args).cuda()
    else:
        raise Exception("Not implemented")
    
    if os.path.isfile(f'outputs/{args.exp_name}/checkpoints/ckpt.checkpoint'):
        model, opt, scheduler = load_checkpoint(
            path=f'outputs/{args.exp_name}/ckpt.checkpoint',
            args=args,
            train_dl_size=len(train_loader))
    else:
        # Convert BatchNorm to SyncBatchNorm.
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)        
        # wrap the model with DDP
        # device_ids tell DDP where is your model
        # output_device tells DDP where to output, in our case, it is rank
        # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank)
        # print("Let's use", torch.cuda.device_count(), "GPUs!")

        if args.use_sgd:
            opt = optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.decay)
        else:
            opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)

        if args.scheduler == 'cycle':
            scheduler = OneCycleLR(opt, max_lr=0.1, epochs=200, 
                                   steps_per_epoch=len(train_loader))
        else:
            # cosine scheduler
            scheduler = CosineAnnealingLR(opt, len(train_loader))

    criterion = cross_entropy
    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print("Number of parameters:", total_params)
    best_test_iou = 0
    # Mixed precision combines Floating Point (FP) 16 and FP 32 in different steps of the training.
    # FP16 training is also known as half-precision training, which comes with inferior performance.
    # Automatic mixed-precision is literally the best of both worlds:
    # reduced training time with comparable performance to FP32
    fp16_scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        ddp_train_loss = torch.zeros(2).to(torch.device(local_rank))
        # if we are using DistributedSampler, we have to tell it which epoch this is
        train_loader.sampler.set_epoch(epoch)  # type: ignore
        test_loader.sampler.set_epoch(epoch)  # type: ignore
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        train_label_seg = []
        # batch accumulation parameter
        for data, label, seg in tqdm(train_loader):
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = \
                data.cuda(local_rank, non_blocking=True), \
                label_one_hot.cuda(local_rank, non_blocking=True), \
                seg.cuda(local_rank, non_blocking=True)
            data = data.transpose(2, 1)
            batch_size = data.size()[0]
            opt.zero_grad()
            # forward
            with torch.cuda.amp.autocast():  # type: ignore
                seg_pred = model(data.contiguous(), label_one_hot.contiguous())
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                if torch.isnan(seg_pred).sum() > 0:
                    print("NaNs detected!!! Stopping training")
                    torch.save(seg_pred, f"preds_{local_rank}.pt")
                    torch.save(model.module.state_dict(), f"weights_{local_rank}.pt")
                    sys.exit(0)
                loss = criterion(seg_pred.view(-1, seg_num_all), seg.view(-1,1).squeeze())
                ddp_train_loss[0] += loss.item()
            fp16_scaler.scale(loss).backward()  # type: ignore
            fp16_scaler.step(opt)
            scheduler.step()
            fp16_scaler.update()
            pred = seg_pred.max(dim=2)[1]               # (batch_size, num_points)
            ddp_train_loss[1] += batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            train_true_cls.append(seg_np.reshape(-1))       # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))      # (batch_size * num_points)
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
            train_label_seg.append(label.reshape(-1))
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_label_seg = np.concatenate(train_label_seg)
        train_ious = calculate_shape_IoU(train_pred_seg, train_true_seg, train_label_seg, args.class_choice)
        dist.all_reduce(ddp_train_loss, op=dist.ReduceOp.SUM)
        if local_rank == 0:            
            outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,
                                                                                                      ddp_train_loss[0] / ddp_train_loss[1],
                                                                                                      train_acc,
                                                                                                      avg_per_class_acc,
                                                                                                      np.mean(train_ious))
            io.cprint(outstr)
        gc.collect()
        torch.cuda.empty_cache()
        ####################
        # Test
        ####################
        ddp_test_loss = torch.zeros(2).to(torch.device(local_rank))
        model.eval()
        test_true_cls = []
        test_pred_cls = []
        test_true_seg = []
        test_pred_seg = []
        test_label_seg = []
        for data, label, seg in test_loader:
            seg = seg - seg_start_index
            label_one_hot = np.zeros((label.shape[0], 16))
            for idx in range(label.shape[0]):
                label_one_hot[idx, label[idx]] = 1
            label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
            data, label_one_hot, seg = data.to(device), label_one_hot.to(
                device), seg.to(device)
            data = data.transpose(2, 1)
            batch_size = data.size()[0]
            with torch.no_grad():
                seg_pred = model(data, label_one_hot)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                loss = criterion(seg_pred.view(-1, seg_num_all),
                                seg.view(-1, 1).squeeze())
            ddp_test_loss[0] += loss.item()
            ddp_test_loss[1] += batch_size
            pred = seg_pred.max(dim=2)[1]
            seg_np = seg.cpu().numpy()
            pred_np = pred.detach().cpu().numpy()
            test_true_cls.append(seg_np.reshape(-1))
            test_pred_cls.append(pred_np.reshape(-1))
            test_true_seg.append(seg_np)
            test_pred_seg.append(pred_np)
            test_label_seg.append(label.reshape(-1))
        test_true_cls = np.concatenate(test_true_cls)
        test_pred_cls = np.concatenate(test_pred_cls)
        test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
        test_true_seg = np.concatenate(test_true_seg, axis=0)
        test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        test_label_seg = np.concatenate(test_label_seg)
        test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
        dist.all_reduce(ddp_test_loss, op=dist.ReduceOp.SUM)
        if local_rank == 0:
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
                                                                                                  ddp_test_loss[0] /
                                                                                                  ddp_test_loss[1],
                                                                                                  test_acc,
                                                                                                  avg_per_class_acc,
                                                                                                  np.mean(test_ious))
            io.cprint(outstr)
            if np.mean(test_ious) >= best_test_iou:
                best_test_iou = np.mean(test_ious)
                save_checkpoint(epoch, model, opt, scheduler, ddp_test_loss[0] /
                                ddp_test_loss[1], args.exp_name, best=True)
        gc.collect()
        torch.cuda.empty_cache()


def save_checkpoint(epoch, model, opt, scheduler, loss, exp_name, best=False):
    if best:
        save_dir = f'outputs/{exp_name}/models/'
        files = os.listdir(save_dir)
        for file in files:
            os.remove(os.path.join(save_dir, file))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, f'outputs/{exp_name}/models/transformer_{epoch}.checkpoint')
    else:
        save_dir = f'outputs/{exp_name}/checkpoints/'
        files = os.listdir(save_dir)
        for file in files:
            os.remove(os.path.join(save_dir, file))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, f'outputs/{exp_name}/checkpoints/ckpt_{epoch}.checkpoint')


def load_checkpoint(path, args, train_dl_size):
    checkpoint = torch.load(path)
    model = Net(args)
    opt = optim.SGD(model.parameters(), lr=args.lr*100,
                    momentum=args.momentum, weight_decay=1e-4)
    scheduler = OneCycleLR(opt, max_lr=args.lr*100,
                           epochs=200, steps_per_epoch=train_dl_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    return model, opt, scheduler


def test(args, io):
    test_ds = ShapeNetPart_Augmented(partition="test")
    ngpus_per_node = torch.cuda.device_count()
    args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
    test_loader = prepare_dl(test_ds, drop_last=False, 
                            batch_size=args.test_batch_size, pin_memory=False)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)
    
    #Try to load models
    seg_start_index = 0
    partseg_colors = test_loader.dataset.partseg_colors
    model = Net(args).to(device)

    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    test_true_cls = []
    test_pred_cls = []
    test_true_seg = []
    test_pred_seg = []
    test_label_seg = []
    for data, label, seg in test_loader:
        seg = seg - seg_start_index
        label_one_hot = np.zeros((label.shape[0], 16))
        for idx in range(label.shape[0]):
            label_one_hot[idx, label[idx]] = 1
        label_one_hot = torch.from_numpy(label_one_hot.astype(np.float32))
        data, label_one_hot, seg = data.to(device), label_one_hot.to(device), seg.to(device)
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        with torch.no_grad():
            seg_pred = model(data, label_one_hot)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        pred = seg_pred.max(dim=2)[1]
        seg_np = seg.cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        test_true_cls.append(seg_np.reshape(-1))
        test_pred_cls.append(pred_np.reshape(-1))
        test_true_seg.append(seg_np)
        test_pred_seg.append(pred_np)
        test_label_seg.append(label.reshape(-1))
        # visiualization
        visualization(args.visu, args.visu_format, data, pred, seg, label, partseg_colors, args.class_choice) 
    if visual_warning and args.visu != '':
        print('Visualization Failed: You can only choose a point cloud shape to visualize within the scope of the test class')
    test_true_cls = np.concatenate(test_true_cls)
    test_pred_cls = np.concatenate(test_pred_cls)
    test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
    test_true_seg = np.concatenate(test_true_seg, axis=0)
    test_pred_seg = np.concatenate(test_pred_seg, axis=0)
    test_label_seg = np.concatenate(test_label_seg)
    test_ious = calculate_shape_IoU(test_pred_seg, test_true_seg, test_label_seg, args.class_choice)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_acc,
                                                                             avg_per_class_acc,
                                                                             np.mean(test_ious))
    io.cprint(outstr)


def main(args):
    io = IOStream('outputs/' + args.exp_name + '/run.log')

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ['RANK'])

    if args.local_rank == 0:
        io.cprint(str(args))

    torch.cuda.set_device(args.local_rank)  # before your code runs
    torch.backends.cudnn.benchmark = True  # type: ignore
    torch.manual_seed(args.seed)
    io.cprint(
        'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    torch.cuda.manual_seed(args.seed)


    # setup the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # dist.init_process_group('nccl', init_method='env://', 
    #                         world_size=args.world_size,
    #                         rank=args.rank)
    dist.init_process_group('nccl')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
    dist.destroy_process_group()


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser(
        description='Point Cloud Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model', type=str, default='transformer', metavar='N',
                        choices=['dgcnn', 'transformer'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='shapenetpart', metavar='N',
                        choices=['shapenetpart'])
    parser.add_argument('--class_choice', type=str, default=None, metavar='N',
                        choices=['airplane', 'bag', 'cap', 'car', 'chair',
                                 'earphone', 'guitar', 'knife', 'lamp', 'laptop',
                                 'motor', 'mug', 'pistol', 'rocket', 'skateboard', 'table'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', default=False, action='store_true',
                        help='Use SGD')
    parser.add_argument('--decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cycle', metavar='N',
                        choices=['cos', 'cycle'],
                        help='Scheduler to use, [cos, cycle]')
    parser.add_argument('--no_cuda', default=False, action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--use_custom_attention', default=False, action='store_true',
                        help='use a custom attention mechanism for fusion')
    parser.add_argument('--use_height', default=False, action='store_true',
                        help='append relative height of each point as an extra feature')
    parser.add_argument('--ff_dims', type=int, default=512,
                        help='dimension of feed forward network inside transformer')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_heads', type=int, default=2,
                        help='number of attention heads')
    parser.add_argument('--n_blocks', type=int, default=2,
                        help='number of layers of encoder/decoder')
    parser.add_argument('--d_qkv', type=int, default=64,
                        help='dimension of q,k,v')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate'),
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--nclasses', type=int, default=50, 
                        help='number of classes to predict')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    args = parser.parse_args()

    import pickle
    with open("args.pkl", "wb") as f:
        pickle.dump(args, f)
    
    _init_()

    main(args)
