import argparse
import os
import time
from typing import Any
import yaml
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader

import models
import losses
import dino_datasets
from core.registry import TRANSFORMS
from core.builder import (
    build_model,
    build_loss,
    build_optimizer,
    build_schedulers,
)
from utils.metrics import AverageMeter


def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    dino_loss: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    epoch: int,
    momentum_schedule: list,
    cfg: dict,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
):
    student.train()
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses_meter = AverageMeter("Loss", ":.4e")

    end = time.time()
    for it, (images, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        # Update learning rate and weight decay
        global_it = len(data_loader) * epoch + it
        lr_scheduler.step(global_it)

        # Move images to device
        images = [im.to(device, non_blocking=True) for im in images]

        # Mixed precision context
        use_cuda = device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_cuda):
            # Teacher processes only the first 2 global views
            with torch.no_grad():
                teacher_output = teacher(images[:2])

            # Student processes all crops (2 global + N local)
            student_output = student(images)

            # Compute cross-entropy distillation loss
            loss = dino_loss(student_output, teacher_output, epoch)
            loss = loss / cfg["train"]["gradient_accumulation_steps"]

        losses_meter.update(
            loss.item() * cfg["train"]["gradient_accumulation_steps"],
            images[0].size(0),
        )

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation step
        if (it + 1) % cfg["train"]["gradient_accumulation_steps"] == 0 or (it + 1) == len(data_loader):
            if cfg["train"]["clip_grad"]:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(student.parameters(), cfg["train"]["clip_grad"])

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Teacher EMA update
            with torch.no_grad():
                m = momentum_schedule[global_it]
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        batch_time.update(time.time() - end)
        end = time.time()

        if it % 50 == 0:
            print(
                f"Epoch: [{epoch}][{it}/{len(data_loader)}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Loss {losses_meter.val:.4f} ({losses_meter.avg:.4f})"
            )

    return losses_meter.avg


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(cfg["train"]["output_dir"], exist_ok=True)

    # 1. Dataset & Transform via TRANSFORMS Registry
    transform = TRANSFORMS.build({
        "type": "DataAugmentationDINO",
        "global_crops_scale": cfg["dataset"]["global_crops_scale"],
        "local_crops_scale": cfg["dataset"]["local_crops_scale"],
        "local_crops_number": cfg["dataset"]["local_crops_number"],
    })

    if not os.path.exists(cfg["dataset"]["data_path"]):
        print(f"WARNING: Dataset path {cfg['dataset']['data_path']} not found.")
        print("Creating a dummy ImageFolder for dry-run testing...")
        os.makedirs(os.path.join(cfg["dataset"]["data_path"], "dummy"), exist_ok=True)
        import numpy as np
        from PIL import Image

        for i in range(64):
            im = Image.fromarray(np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8))
            im.save(os.path.join(cfg["dataset"]["data_path"], "dummy", f"{i}.jpg"))

    dataset = datasets.ImageFolder(cfg["dataset"]["data_path"], transform=transform)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size_per_gpu"],
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == "cuda" else False,
        drop_last=True,
    )

    # 2. Dynamic Model Construction via Builder & Registry
    student, teacher = build_model(cfg, device=device)

    # 3. Dynamic Loss Construction
    dino_loss = build_loss(cfg, device=device)

    # 4. Dynamic Optimizer & Schedulers
    optimizer = build_optimizer(student, cfg)
    lr_scheduler, momentum_schedule = build_schedulers(optimizer, cfg, len(data_loader))

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # 5. Training Loop
    print(f"Starting DINO training with {len(data_loader)} iterations per epoch...")
    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(
            student=student,
            teacher=teacher,
            dino_loss=dino_loss,
            data_loader=data_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            momentum_schedule=momentum_schedule,
            cfg=cfg,
            scaler=scaler,
            device=device,
        )

        # Save checkpoints
        if cfg["train"]["save_ckp_freq"] and epoch % cfg["train"]["save_ckp_freq"] == 0:
            ckp_path = os.path.join(cfg["train"]["output_dir"], f"checkpoint{epoch:04}.pth")
            torch.save(
                {
                    "student": student.state_dict(),
                    "teacher": teacher.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "loss": train_loss,
                },
                ckp_path,
            )
            print(f"Saved checkpoint to {ckp_path}")

        print(f"End of epoch {epoch} | Loss {train_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DINO self-supervised training")
    parser.add_argument("--config", default="configs/vit_tiny_yolo.yaml", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
