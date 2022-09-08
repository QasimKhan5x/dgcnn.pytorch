#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao, Pengliang Ji
@Contact: ta19@mails.tsinghua.edu.cn, jpl1723@buaa.edu.cn
@File: main_semseg.py
@Time: 2021/7/20 7:49 PM
"""

import argparse
import gc
import os
import random
import sys

import numpy as np
import sklearn.metrics as metrics
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from plyfile import PlyData, PlyElement
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import OneCycleLR  # type: ignore
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import s3dis_transforms as T
from loss import cross_entropy
from models.model_semseg import Net
from s3dis import S3DIS
from util import IOStream

global room_seg
room_seg = []
global room_pred
room_pred = []
global visual_warning
visual_warning = True


def _init_(exp_name):
    BASE_DIR = f'outputs/semseg/{exp_name}'
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
    if not os.path.exists(f'{BASE_DIR}/models'):
        os.makedirs(f'{BASE_DIR}/models')
    if not os.path.exists(f'{BASE_DIR}/visualization'):
        os.makedirs(f'{BASE_DIR}/visualization')
    if not os.path.exists(f'{BASE_DIR}/checkpoints'):
        os.makedirs(f'{BASE_DIR}/checkpoints')
    if not os.path.exists(f'{BASE_DIR}/backups'):
        os.makedirs(f'{BASE_DIR}/backups')
    os.system(f'cp main_semseg.py {BASE_DIR}/backups/main_semseg.py.backup')
    os.system(f'cp loss.py {BASE_DIR}/backups/loss.py.backup')
    os.system(f'cp data.py {BASE_DIR}/backups/data.py.backup')
    os.system(f'cp models/* {BASE_DIR}/backups/')


def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(
                pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(
                pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    if visual:
        for sem in range(13):
            if U_all[sem] == 0:
                I_all[sem] = 1
                U_all[sem] = 1
    return I_all / U_all


def visualization(visu, visu_format, test_choice, 
                  data, seg, pred,
                  visual_file_index, semseg_colors, exp_name):
    global room_seg, room_pred
    global visual_warning
    visu = visu.split('_')
    for i in range(0, data.shape[0]):
        RGB = []
        RGB_gt = []
        skip = False
        with open("data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
            files = f.readlines()
            test_area = files[visual_file_index][5]
            roomname = files[visual_file_index][7:-1]
            if visual_file_index + 1 < len(files):
                roomname_next = files[visual_file_index+1][7:-1]
            else:
                roomname_next = ''
        if visu[0] != 'all':
            if len(visu) == 2:
                if visu[0] != 'area' or visu[1] != test_area:
                    skip = True
                else:
                    visual_warning = False
            elif len(visu) == 4:
                if visu[0] != 'area' or visu[1] != test_area or visu[2] != roomname.split('_')[0] or visu[3] != roomname.split('_')[1]:
                    skip = True
                else:
                    visual_warning = False
            else:
                skip = True
        elif test_choice != 'all':
            skip = True
        else:
            visual_warning = False
        if skip:
            visual_file_index = visual_file_index + 1
        else:
            if not os.path.exists('outputs/semseg/'+exp_name+'/'+'visualization'+'/'+'area_'+test_area+'/'+roomname):
                os.makedirs('outputs/semseg/'+exp_name+'/' +
                            'visualization'+'/'+'area_'+test_area+'/'+roomname)

            data = np.loadtxt('data/indoor3d_sem_seg_hdf5_data_test/raw_data3d/Area_' +
                              test_area+'/'+roomname+'('+str(visual_file_index)+').txt')
            visual_file_index = visual_file_index + 1
            for j in range(0, data.shape[0]):
                RGB.append(semseg_colors[int(pred[i][j])])
                RGB_gt.append(semseg_colors[int(seg[i][j])])
            data = data[:, [1, 2, 0]]
            xyzRGB = np.concatenate((data, np.array(RGB)), axis=1)
            xyzRGB_gt = np.concatenate((data, np.array(RGB_gt)), axis=1)
            room_seg.append(seg[i].cpu().numpy())
            room_pred.append(pred[i].cpu().numpy())
            f = open('outputs/semseg/'+exp_name+'/'+'visualization'+'/' +
                     'area_'+test_area+'/'+roomname+'/'+roomname+'.txt', "a")
            f_gt = open('outputs/semseg/'+exp_name+'/'+'visualization'+'/' +
                        'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.txt', "a")
            np.savetxt(f, xyzRGB, fmt='%s', delimiter=' ')
            np.savetxt(f_gt, xyzRGB_gt, fmt='%s', delimiter=' ')

            if roomname != roomname_next:
                mIoU = np.mean(calculate_sem_IoU(
                    np.array(room_pred), np.array(room_seg), visual=True))
                mIoU = str(round(mIoU, 4))
                room_pred = []
                room_seg = []
                if visu_format == 'ply':
                    filepath = 'outputs/semseg/'+exp_name+'/'+'visualization'+'/' + \
                        'area_'+test_area+'/'+roomname+'/'+roomname+'_pred_'+mIoU+'.ply'
                    filepath_gt = 'outputs/semseg/'+exp_name+'/'+'visualization' + \
                        '/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.ply'
                    xyzRGB = np.loadtxt('outputs/semseg/'+exp_name+'/'+'visualization' +
                                        '/'+'area_'+test_area+'/'+roomname+'/'+roomname+'.txt')
                    xyzRGB_gt = np.loadtxt('outputs/semseg/'+exp_name+'/'+'visualization' +
                                           '/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.txt')
                    xyzRGB = [(xyzRGB[i, 0], xyzRGB[i, 1], xyzRGB[i, 2], xyzRGB[i, 3],
                               xyzRGB[i, 4], xyzRGB[i, 5]) for i in range(xyzRGB.shape[0])]
                    xyzRGB_gt = [(xyzRGB_gt[i, 0], xyzRGB_gt[i, 1], xyzRGB_gt[i, 2], xyzRGB_gt[i, 3],
                                  xyzRGB_gt[i, 4], xyzRGB_gt[i, 5]) for i in range(xyzRGB_gt.shape[0])]
                    vertex = PlyElement.describe(np.array(xyzRGB, dtype=[('x', 'f4'), ('y', 'f4'), (
                        'z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath)
                    print('PLY visualization file saved in', filepath)
                    vertex = PlyElement.describe(np.array(xyzRGB_gt, dtype=[(
                        'x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]), 'vertex')
                    PlyData([vertex]).write(filepath_gt)
                    print('PLY visualization file saved in', filepath_gt)
                    os.system('rm -rf '+'outputs/semseg/'+exp_name +
                              '/visualization/area_'+test_area+'/'+roomname+'/*.txt')
                else:
                    filename = 'outputs/semseg/'+exp_name+'/'+'visualization' + \
                        '/'+'area_'+test_area+'/'+roomname+'/'+roomname+'.txt'
                    filename_gt = 'outputs/semseg/'+exp_name+'/'+'visualization' + \
                        '/'+'area_'+test_area+'/'+roomname+'/'+roomname+'_gt.txt'
                    filename_mIoU = 'outputs/semseg/'+exp_name+'/'+'visualization'+'/' + \
                        'area_'+test_area+'/'+roomname+'/'+roomname+'_pred_'+mIoU+'.txt'
                    os.rename(filename, filename_mIoU)
                    print('TXT visualization file saved in', filename_mIoU)
                    print('TXT visualization file saved in', filename_gt)
            elif visu_format != 'ply' and visu_format != 'txt':
                print('ERROR!! Unknown visualization format: %s, please use txt or ply.' %
                      (visu_format))
                exit()


def prepare_dl(dataset, drop_last, batch_size, pin_memory, num_workers=0):
    '''split the dataloader to each process in the process group'''
    sampler = DistributedSampler(dataset, drop_last=drop_last)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers, pin_memory=pin_memory)
    return dataloader


def save_checkpoint(epoch, model, opt, scheduler, loss, exp_name, best=False):
    if best:
        save_dir = f'outputs/semseg/{exp_name}/models/'
        files = os.listdir(save_dir)
        for file in files:
            os.remove(os.path.join(save_dir, file))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, f'outputs/semseg/{exp_name}/models/transformer_{epoch}.checkpoint')
    else:
        save_dir = f'outputs/semseg/{exp_name}/checkpoints/'
        files = os.listdir(save_dir)
        for file in files:
            os.remove(os.path.join(save_dir, file))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
        }, f'outputs/semseg/{exp_name}/checkpoints/ckpt_{epoch}.checkpoint')


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


def train(args, io):
    train_ds = S3DIS(split='train', voxel_max=24000, presample=True, transform=T.Compose([
        # T.AppendHeight(), T.PointCloudFloorCentering(),
        # T.RandomScale(), T.RandomRotate(), T.RandomJitter(),
        T.ChromaticNormalize(), #T.ChromaticAutoContrast(),
        #T.RandomDropColor(),
         T.ToTensor()
    ]))
    test_ds = S3DIS(split='val', transform=T.Compose([
        T.PointCloudFloorCentering(),
        T.ChromaticNormalize(),
        T.ToTensor()])
    )

    ngpus_per_node = torch.cuda.device_count()
    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.test_batch_size = int(args.test_batch_size / ngpus_per_node)

    train_loader = prepare_dl(train_ds, drop_last=True,
                              batch_size=args.batch_size, pin_memory=True)
    test_loader = prepare_dl(test_ds, drop_last=False,
                             batch_size=args.test_batch_size, pin_memory=False)

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(local_rank)

    model = Net(args).cuda()

    if os.path.isfile(f'outputs/segmseg/{args.exp_name}/checkpoints/ckpt.checkpoint'):
        model, opt, scheduler = load_checkpoint(
            path=f'outputs/semseg/{args.exp_name}/ckpt.checkpoint',
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

        if args.use_sgd:
            opt = optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=1e-4)
        else:
            opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        if args.scheduler == 'cos':
            scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=1e-3)
        else:
            # 1-cycle scheduler
            scheduler = OneCycleLR(opt, max_lr=0.1, epochs=args.epochs,
                                   steps_per_epoch=len(train_loader))

    if local_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print("Number of parameters:", total_params)
    
    criterion = cross_entropy
    best_test_iou = 0
    fp16_scaler = torch.cuda.amp.GradScaler(enabled=True) # type: ignore
    for epoch in range(args.epochs):
        ####################
        # Train
        ####################
        ddp_train_loss = torch.zeros(2).to(torch.device(local_rank))
        # if we are using DistributedSampler, 
        # we have to tell it which epoch this is
        train_loader.sampler.set_epoch(epoch)  # type: ignore
        test_loader.sampler.set_epoch(epoch)  # type: ignore
        model.train()
        train_true_cls = []
        train_pred_cls = []
        train_true_seg = []
        train_pred_seg = []
        for item in tqdm(train_loader):
            pos, feat, seg = item['pos'], item['x'], item['y']
            pos, feat, seg = pos.cuda(local_rank, non_blocking=True), feat.cuda(
                local_rank, non_blocking=True), seg.cuda(local_rank, non_blocking=True)
            pos = pos.transpose(2, 1).contiguous()
            feat = feat.transpose(2, 1).contiguous()
            batch_size = pos.size()[0]
            opt.zero_grad()
            with torch.cuda.amp.autocast(): # type: ignore
                seg_pred = model(pos, feat)
                seg_pred = seg_pred.transpose(2, 1).contiguous()
                if torch.isnan(seg_pred).sum() > 0:
                    print("NaNs detected!!! Stopping training")
                    torch.save(seg_pred, f"preds_{local_rank}.pt")
                    torch.save(model.module.state_dict(),
                               f"weights_{local_rank}.pt")
                    sys.exit(0)
                loss = criterion(seg_pred.view(-1, args.nclasses),
                                 seg.view(-1, 1).squeeze())
                ddp_train_loss[0] += loss.item()
            fp16_scaler.scale(loss).backward()  # type: ignore
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10, norm_type=2)
            fp16_scaler.step(opt)
            fp16_scaler.update()
            scheduler.step()
            # (batch_size, num_points)
            pred = seg_pred.max(dim=2)[1]
            ddp_train_loss[0] += loss.item() * batch_size
            ddp_train_loss[1] += batch_size
            seg_np = seg.cpu().numpy()                  # (batch_size, num_points)
            pred_np = pred.detach().cpu().numpy()       # (batch_size, num_points)
            # (batch_size * num_points)
            train_true_cls.append(seg_np.reshape(-1))
            # (batch_size * num_points)
            train_pred_cls.append(pred_np.reshape(-1))
            train_true_seg.append(seg_np)
            train_pred_seg.append(pred_np)
        train_true_cls = np.concatenate(train_true_cls)
        train_pred_cls = np.concatenate(train_pred_cls)
        train_acc = metrics.accuracy_score(train_true_cls, train_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(
            train_true_cls, train_pred_cls)
        train_true_seg = np.concatenate(train_true_seg, axis=0)
        train_pred_seg = np.concatenate(train_pred_seg, axis=0)
        train_ious = calculate_sem_IoU(train_pred_seg, train_true_seg)
        if local_rank == 0:
            outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f, train iou: %.6f' % (epoch,
                                                                                                      ddp_train_loss[0] /
                                                                                                      ddp_train_loss[1],
                                                                                                      train_acc,
                                                                                                      avg_per_class_acc,
                                                                                                      np.mean(train_ious)
                                                                                                      )
            io.cprint(outstr)
        opt.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()
        ####################
        # Test
        ####################
        # ddp_test_loss = torch.zeros(2).to(torch.device(local_rank))
        # model.eval()
        # test_true_cls = []
        # test_pred_cls = []
        # test_true_seg = []
        # test_pred_seg = []
        # for item in test_loader:
        #     pos, feat, seg = item['pos'], item['x'], item['y']
        #     pos, feat, seg = pos.cuda(local_rank, non_blocking=True), feat.cuda(
        #         local_rank, non_blocking=True), seg.cuda(local_rank, non_blocking=True)
        #     pos = pos.transpose(2, 1).contiguous()
        #     feat = feat.transpose(2, 1).contiguous()
        #     batch_size = pos.size()[0]
        #     with torch.no_grad():
        #         seg_pred = model(pos, feat)
        #         seg_pred = seg_pred.transpose(2, 1).contiguous()
        #         loss = criterion(seg_pred.view(-1, 13), seg.view(-1, 1).squeeze())
        #     pred = seg_pred.max(dim=2)[1]
        #     ddp_test_loss[0] += loss.item() * batch_size
        #     ddp_test_loss[1] += batch_size
        #     seg_np = seg.cpu().numpy()
        #     pred_np = pred.detach().cpu().numpy()
        #     test_true_cls.append(seg_np.reshape(-1))
        #     test_pred_cls.append(pred_np.reshape(-1))
        #     test_true_seg.append(seg_np)
        #     test_pred_seg.append(pred_np)
        # test_true_cls = np.concatenate(test_true_cls)
        # test_pred_cls = np.concatenate(test_pred_cls)
        # test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
        # avg_per_class_acc = metrics.balanced_accuracy_score(
        #     test_true_cls, test_pred_cls)
        # test_true_seg = np.concatenate(test_true_seg, axis=0)
        # test_pred_seg = np.concatenate(test_pred_seg, axis=0)
        # test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
        # if local_rank == 0:
        #     outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (epoch,
        #                                                                                           ddp_test_loss[0] /
        #                                                                                           ddp_test_loss[1],
        #                                                                                           test_acc,
        #                                                                                           avg_per_class_acc,
        #                                                                                           np.mean(test_ious))
        #     io.cprint(outstr)
        #     if np.mean(test_ious) >= best_test_iou:
        #         best_test_iou = np.mean(test_ious)
        #         save_checkpoint(epoch, model, opt, scheduler, ddp_test_loss[0] /
        #                         ddp_test_loss[1], args.exp_name, best=True)
        # gc.collect()
        # torch.cuda.empty_cache()


def test(args, io):
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in range(1, 7):
        visual_file_index = 0
        test_area = str(test_area)
        if os.path.exists("data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt"):
            with open("data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
                for line in f:
                    if (line[5]) == test_area:
                        break
                    visual_file_index = visual_file_index + 1
        if (args.test_area == 'all') or (test_area == args.test_area):
            test_loader = DataLoader(S3DIS(partition='test', num_points=args.num_points, test_area=test_area),
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device(local_rank)

            #Try to load models
            semseg_colors = test_loader.dataset.semseg_colors
            model = Net(args).to(device)
            assert args.model_path != '', "No pretrained model path specified"
            model_path = f'outputs/semseg/models/{args.model_path}'
            model.load_state_dict(torch.load(model_path))
            model = model.eval()
            test_acc = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            for data, seg in test_loader:
                data, seg = data.to(device), seg.to(device)
                data = data.permute(0, 2, 1)
                seg_pred = model(data)
                seg_pred = seg_pred.permute(0, 2, 1).contiguous()
                pred = seg_pred.max(dim=2)[1]
                seg_np = seg.cpu().numpy()
                pred_np = pred.detach().cpu().numpy()
                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)
                # visiualization
                visualization(args.visu, args.visu_format, args.test_area,
                              data, seg, pred, visual_file_index, semseg_colors, exp_name)
                visual_file_index = visual_file_index + data.shape[0]
            if visual_warning and args.visu != '':
                print(
                    'Visualization Failed: You can only choose a room to visualize within the scope of the test area')
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(
                test_true_cls, test_pred_cls)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_ious = calculate_sem_IoU(test_pred_seg, test_true_seg)
            outstr = 'Test :: test area: %s, test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (test_area,
                                                                                                    test_acc,
                                                                                                    avg_per_class_acc,
                                                                                                    np.mean(test_ious))
            io.cprint(outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

    if args.test_area == 'all':
        all_true_cls = np.concatenate(all_true_cls)
        all_pred_cls = np.concatenate(all_pred_cls)
        all_acc = metrics.accuracy_score(all_true_cls, all_pred_cls)
        avg_per_class_acc = metrics.balanced_accuracy_score(
            all_true_cls, all_pred_cls)
        all_true_seg = np.concatenate(all_true_seg, axis=0)
        all_pred_seg = np.concatenate(all_pred_seg, axis=0)
        all_ious = calculate_sem_IoU(all_pred_seg, all_true_seg)
        outstr = 'Overall Test :: test acc: %.6f, test avg acc: %.6f, test iou: %.6f' % (all_acc,
                                                                                         avg_per_class_acc,
                                                                                         np.mean(all_ious))
        io.cprint(outstr)


def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def main(args):
    io = IOStream('outputs/semseg/' + args.exp_name + '/run.log')

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    args.rank = int(os.environ['RANK'])

    if args.local_rank == 0:
        io.cprint(str(args))

    torch.cuda.set_device(args.local_rank)  # before your code runs
    seed_everything(args.seed)
    io.cprint(
        'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    torch.cuda.manual_seed(args.seed)

    # setup the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
    dist.destroy_process_group()


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(
        description='Point Cloud Semantic Segmentation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['dgcnn', 'transformer'],
                        help='Model to use, [dgcnn]')
    parser.add_argument('--dataset', type=str, default='S3DIS', metavar='N',
                        choices=['S3DIS'])
    parser.add_argument('--test_area', type=str, default=None, metavar='N',
                        choices=['1', '2', '3', '4', '5', '6', 'all'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', default=False, action='store_true',
                        help='Use SGD')
    parser.add_argument('--decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--scheduler', type=str, default='cycle', metavar='N',
                        choices=['cos', 'cycle'],
                        help='Scheduler to use, [cos, step]')
    parser.add_argument('--no_cuda', default=False, action='store_true',
                        help='disables CUDA training')
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
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--k', type=int, default=32, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model root')
    parser.add_argument('--nclasses', type=int, default=13, 
                        help='number of classes to predict')
    parser.add_argument('--visu', type=str, default='',
                        help='visualize the model')
    parser.add_argument('--visu_format', type=str, default='ply',
                        help='file format of visualization')
    args = parser.parse_args()

    _init_(args.exp_name)
    main(args)
