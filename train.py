import argparse
import os
import time
import math
import yaml
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader

import models.vision_transformer as vits
from models.dino_head import DINOHead
from dino_datasets.transforms import DataAugmentationDINO
from losses.dino_loss import DINOLoss
from utils.schedulers import cosine_scheduler, get_params_groups, LRScheduler
from utils.metrics import AverageMeter

def build_model(cfg):
    # Backbone
    if cfg['model']['arch'] == 'vit_small':
        student = vits.vit_small(patch_size=cfg['model']['patch_size'])
        teacher = vits.vit_small(patch_size=cfg['model']['patch_size'])
    elif cfg['model']['arch'] == 'vit_tiny':
        student = vits.vit_tiny(patch_size=cfg['model']['patch_size'])
        teacher = vits.vit_tiny(patch_size=cfg['model']['patch_size'])
    else:
        raise NotImplementedError

    embed_dim = student.embed_dim

    # Multi-crop wrapper head
    student_head = DINOHead(
        embed_dim,
        cfg['model']['out_dim'],
        use_bn=cfg['model']['use_bn_in_head'],
        norm_last_layer=cfg['model']['norm_last_layer'] if 'norm_last_layer' in cfg['model'] else True,
    )
    teacher_head = DINOHead(
        embed_dim,
        cfg['model']['out_dim'],
        use_bn=cfg['model']['use_bn_in_head']
    )

    class MultiCropWrapper(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head
        def forward(self, x):
            # x is a list of tensors (crops)
            # We process crops of different resolutions separately to avoid tensor size mismatch
            idx_crops = torch.cumsum(torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)
            start_idx = 0
            for end_idx in idx_crops:
                _out = self.backbone(torch.cat(x[start_idx:end_idx]))
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx
            return self.head(output)

    student = MultiCropWrapper(student, student_head)
    teacher = MultiCropWrapper(teacher, teacher_head)

    # Move to GPU
    student = student.cuda()
    teacher = teacher.cuda()

    # Teacher identical to student at start, freeze teacher
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    return student, teacher

def train_one_epoch(student, teacher, dino_loss, data_loader, optimizer, lr_scheduler, epoch,
                    momentum_schedule, cfg, scaler):
    student.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    end = time.time()
    for it, (images, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        # update learning rate and weight decay
        global_it = len(data_loader) * epoch + it
        lr_scheduler.step(global_it)

        # Move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]

        # Teacher and Student forward passes
        with torch.cuda.amp.autocast(enabled=True):
            # Teacher only gets global views (first 2 crops)
            with torch.no_grad():
                teacher_output = teacher(images[:2])

            # Student gets all views
            student_output = student(images)

            # Compute loss
            loss = dino_loss(student_output, teacher_output, epoch)
            # Scale loss for gradient accumulation
            loss = loss / cfg['train']['gradient_accumulation_steps']

        losses.update(loss.item() * cfg['train']['gradient_accumulation_steps'], images[0].size(0))

        # Backward pass
        scaler.scale(loss).backward()

        # Optimizer step (gradient accumulation)
        if (it + 1) % cfg['train']['gradient_accumulation_steps'] == 0 or (it + 1) == len(data_loader):
            if cfg['train']['clip_grad']:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), cfg['train']['clip_grad'])

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # EMA update for teacher
            with torch.no_grad():
                m = momentum_schedule[global_it]  # momentum parameter
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        batch_time.update(time.time() - end)
        end = time.time()

        if it % 50 == 0:
            print(f"Epoch: [{epoch}][{it}/{len(data_loader)}] "
                  f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})")

    return losses.avg

def main(cfg):
    os.makedirs(cfg['train']['output_dir'], exist_ok=True)

    # Dataset and DataLoader
    transform = DataAugmentationDINO(
        global_crops_scale=cfg['dataset']['global_crops_scale'],
        local_crops_scale=cfg['dataset']['local_crops_scale'],
        local_crops_number=cfg['dataset']['local_crops_number']
    )

    # Warning if data_path doesn't exist
    if not os.path.exists(cfg['dataset']['data_path']):
        print(f"WARNING: Dataset path {cfg['dataset']['data_path']} not found. Please run prepare_subset.py first.")
        # Create a dummy dataset for dry-run if path is missing
        print("Creating a dummy ImageFolder for dry-run testing...")
        os.makedirs(os.path.join(cfg['dataset']['data_path'], 'dummy'), exist_ok=True)
        import numpy as np
        from PIL import Image
        for i in range(10):
            im = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
            im.save(os.path.join(cfg['dataset']['data_path'], 'dummy', f'{i}.jpg'))

    dataset = datasets.ImageFolder(cfg['dataset']['data_path'], transform=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg['train']['batch_size_per_gpu'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    # Model
    student, teacher = build_model(cfg)

    # Loss
    dino_loss = DINOLoss(
        out_dim=cfg['model']['out_dim'],
        ncrops=cfg['dataset']['local_crops_number'] + 2,
        warmup_teacher_temp=cfg['train']['warmup_teacher_temp'],
        teacher_temp=cfg['train']['teacher_temp'],
        warmup_teacher_temp_epochs=cfg['train']['warmup_teacher_temp_epochs'],
        nepochs=cfg['train']['epochs'],
    ).cuda()

    # Optimizer
    params_groups = get_params_groups(student)
    optimizer = torch.optim.AdamW(params_groups)

    # Schedulers
    lr_schedule = cosine_scheduler(
        cfg['train']['lr'] * (cfg['train']['batch_size_per_gpu'] * cfg['train']['gradient_accumulation_steps']) / 256.,
        cfg['train']['min_lr'],
        cfg['train']['epochs'], len(data_loader),
        warmup_epochs=cfg['train']['warmup_epochs'],
    )
    wd_schedule = cosine_scheduler(
        cfg['train']['weight_decay'],
        cfg['train']['weight_decay_end'],
        cfg['train']['epochs'], len(data_loader),
    )
    momentum_schedule = cosine_scheduler(
        cfg['train']['momentum_teacher'], 1,
        cfg['train']['epochs'], len(data_loader)
    )

    lr_scheduler = LRScheduler(optimizer, lr_schedule, wd_schedule)

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler()

    # Training Loop
    print("Starting DINO training...")
    for epoch in range(cfg['train']['epochs']):
        train_loss = train_one_epoch(
            student, teacher, dino_loss, data_loader, optimizer, lr_scheduler,
            epoch, momentum_schedule, cfg, scaler
        )

        # Save checkpoints
        if cfg['train']['save_ckp_freq'] and epoch % cfg['train']['save_ckp_freq'] == 0:
            torch.save({
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss': train_loss,
            }, os.path.join(cfg['train']['output_dir'], f'checkpoint{epoch:04}.pth'))

        print(f"End of epoch {epoch} | Loss {train_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/vit_small_200k.yaml', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    main(cfg)
