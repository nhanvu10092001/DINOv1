"""Common evaluation utilities, model loaders, and dataset builders for DINOv1."""

import os
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core.registry import BACKBONES
import models.vision_transformer as vits


def build_eval_model(
    arch: str,
    patch_size: int = 16,
    pretrained_weights: str = "",
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Builds and loads a Vision Transformer backbone for downstream evaluation.

    Args:
        arch: Registered backbone name (e.g. 'vit_small', 'vit_small_sparse_rope').
        patch_size: Patch size in pixels (default: 16).
        pretrained_weights: Path to checkpoint .pth file containing 'teacher' or 'student'.
        device: Target device.

    Returns:
        Evaluated nn.Module in eval() mode with gradients disabled.
    """
    if arch in BACKBONES:
        model = BACKBONES.build(dict(type=arch, patch_size=patch_size, num_classes=0))
    elif hasattr(vits, arch):
        model = getattr(vits, arch)(patch_size=patch_size, num_classes=0)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model = model.to(device)

    if pretrained_weights and os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        elif "student" in state_dict:
            state_dict = state_dict["student"]
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in state_dict.items()
            if "backbone." in k or not k.startswith("head")
        }
        msg = model.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded pretrained weights from {pretrained_weights} "
            f"(matched keys: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected)"
        )
    else:
        print(f"WARNING: No pretrained weights provided at {pretrained_weights}, using random initialization!")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_eval_data(
    data_path: str,
    dataset_name: str = "image_folder",
    batch_size: int = 128,
    num_workers: int = 4,
    for_linear_probing: bool = False,
) -> Tuple[DataLoader, DataLoader, int]:
    """Builds train and validation DataLoaders for evaluation (k-NN / Linear Probing).

    Args:
        data_path: Directory containing dataset files.
        dataset_name: 'cifar10', 'cifar100', or 'image_folder'.
        batch_size: Batch size for DataLoader.
        num_workers: Number of DataLoader worker threads.
        for_linear_probing: If True, uses standard linear probe train data augmentations
            (RandomResizedCrop + RandomHorizontalFlip). If False, uses center crop.

    Returns:
        Tuple of (train_loader, val_loader, num_classes).
    """
    if for_linear_probing:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    ds_lower = dataset_name.lower()
    if ds_lower == "cifar10":
        dataset_train = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_val)
        num_classes = 10
    elif ds_lower == "cifar100":
        dataset_train = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform_val)
        num_classes = 100
    else:
        train_path = os.path.join(data_path, "train")
        val_path = os.path.join(data_path, "val")
        if not os.path.exists(val_path):
            val_path = os.path.join(data_path, "validation")
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(f"Cannot find train/val directories under {data_path}")
        dataset_train = datasets.ImageFolder(train_path, transform=transform_train)
        dataset_val = datasets.ImageFolder(val_path, transform=transform_val)
        num_classes = len(dataset_train.classes)

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=for_linear_probing,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    return train_loader, val_loader, num_classes
