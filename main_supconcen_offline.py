 # Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import models as torchvision_models

import utils
from losses import SupConLoss
from models import SimpleHead
from resnet_big import SupConResNet

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('SupConCen', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. resnet50 is recommended""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the head output.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Training/Optimization parameters
    parser.add_argument('--weight_decay', type=float, default=1e-5, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=1e-4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument("--lr", default=0.05, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument('--min_lr', type=float, default=1e-3, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--clip_grad', type=float, default=0., help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""") # def: 3.0 for dino
    parser.add_argument('--batch_size_per_gpu', default=192, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=201, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=-1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""") # def: 1
    parser.add_argument("--warmup_epochs", default=1, type=int,
        help="Number of epochs for the linear learning-rate warm up.") # def 10 for dino
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=111, type=int, help='Random seed.')# 111 222 333
    parser.add_argument('--num_workers', default=4, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--dataset_name", default='imagenet', type=str,
        choices=['imagenet', 'cifar10', 'cifar100'], help="Set dataset name.")
    # parser.add_argument("--loss_type", default='dino', type=str,
    #     choices=['dino', 'supcon'], help="Loss type.")
    parser.add_argument('--use_centroid', default=False, type=utils.bool_flag,\
        help="Use centroids.")
    parser.add_argument('--use_dual_net', default=False, type=utils.bool_flag,\
        help="Use student-teacher networks.")
    parser.add_argument('--use_fixed_centroids', default=False, type=utils.bool_flag,\
        help="Use pretrained fixed centroids.")
    parser.add_argument('--alpha', default=0.1, type=float,\
        help="Alpha for equation 1.")
    parser.add_argument('--eq_type', default='eq1', type=str,\
        choices=['eq1', 'eq2', 'eq3'], help="Equation type.")
    return parser


def train_supconcen(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    transform = utils.DataAugmentation(args.dataset_name)
    aug_count = transform.count

    if args.dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=args.data_path, transform=transform)
        num_lbls = 10
    elif args.dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=args.data_path, transform=transform)
        num_lbls = 100
    elif args.dataset_name == 'imagenet':
        dataset = datasets.ImageFolder(args.data_path, transform=transform)
        # num_lbls = 10

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    if args.arch in torchvision_models.__dict__.keys():
        # student = torchvision_models.__dict__[args.arch]()
        # teacher = torchvision_models.__dict__[args.arch]()
        # embed_dim = student.fc.weight.shape[1]
        student = SupConResNet(name=args.arch)
        embed_dim = student.embed_dim
        student = student.cuda()
        teacher = None

        if args.use_dual_net:
            teacher = SupConResNet(name=args.arch)
            teacher = teacher.cuda()
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    # student = utils.MultiCropWrapper(student, SimpleHead(
    #     embed_dim,
    #     args.out_dim
    # ))
    # teacher = utils.MultiCropWrapper(
    #     teacher, SimpleHead(embed_dim, args.out_dim),
    # )

    # move networks to gpu
    # synchronize batch norms (if any)

    teacher_without_ddp = None
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)

        if args.use_dual_net:
            teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

            # we need DDP wrapper to have synchro batch norms working...
            teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
            teacher_without_ddp = teacher.module
    else:
        if args.use_dual_net:
            # teacher_without_ddp and teacher are the same thing
            teacher_without_ddp = teacher


    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    
    if args.use_dual_net:
        # teacher and student start with the same weights
        teacher_without_ddp.load_state_dict(student.module.state_dict())
        # there is no backpropagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False
        print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    criterion = SupConLoss(temperature=0.07, contrast_mode=args.eq_type).cuda()
    
    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0., momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    
    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        dino_loss=criterion,
    )
    start_epoch = to_restore["epoch"]

    means = None
    if args.use_fixed_centroids:
        means = torch.load('mean.pt').cuda()

    start_time = time.time()
    print("Starting SupConCen training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of SupConCen ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, criterion,
            data_loader, aug_count, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, num_lbls, means, args.alpha, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict() if args.use_dual_net else None,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'criterion': criterion.state_dict(),
        }
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, criterion, data_loader, aug_count,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch, num_lbls, means, alpha, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, labels) in enumerate(metric_logger.log_every(data_loader, 10, header)):

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

            images_all = torch.cat(images[:], dim=0)
            if torch.cuda.is_available():
                images_all = images_all.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

            if args.use_dual_net:
                teacher_output = teacher(images_all)
                t_features = utils.collect_output(teacher_output, args.batch_size_per_gpu, aug_count)
            
            student_output = student(images_all)
            s_features = utils.collect_output(student_output, args.batch_size_per_gpu, aug_count)

            c_ft, c_lbl = None, None

            if args.use_centroid:
                if args.use_fixed_centroids:
                    c_lbl = torch.unique(labels).cuda()
                    c_ft = means[c_lbl,:]
                else:
                    # if args.use_dual_net:
                    #     aug_n, embed_dim = t_features.shape[1], t_features.shape[2]
                    #     aug_mean_ft = torch.mean(t_features, 1, dtype=torch.float).cuda()
                    # else:
                    aug_n, embed_dim = s_features.shape[1], s_features.shape[2]
                    # [bz, aug_n, embed_dim] t_features
                    # [bz, embed_dim] aug_mean_ft
                    aug_mean_ft = torch.mean(s_features, 1, dtype=torch.float).cuda()
                    mean = torch.zeros(num_lbls, embed_dim).cuda()
                    mean = mean.scatter_reduce(0, labels.contiguous().view(-1, 1).repeat(1, embed_dim).cuda(),\
                        aug_mean_ft, reduce="mean", include_self=False)
                    non_empty_mask = mean.abs().sum(dim=1).bool().cuda()
                    c_ft = mean[non_empty_mask,:]
                    c_lbl = torch.unique(labels)
                
                loss = criterion(s_features, labels, centroid_ft=c_ft, centroid_lbl=c_lbl, alpha=alpha)
            else:
                loss = criterion(student_output, teacher_output, epoch)
            
            loss_time = time.time()

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        loss.backward()
        if args.clip_grad:
            param_norms = utils.clip_gradients(student, args.clip_grad)
        utils.cancel_gradients_last_layer(epoch, student,
                                          args.freeze_last_layer)
        optimizer.step()

        # EMA update for the teacher
        if args.use_dual_net:
            with torch.no_grad():
                m = momentum_schedule[it]  # momentum parameter
                for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SupConCen', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_supconcen(args)
