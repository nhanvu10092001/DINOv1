import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
import models.vision_transformer as vits
from core.registry import BACKBONES
from utils.metrics import AverageMeter

class LinearClassifier(nn.Module):
    def __init__(self, dim, num_classes=1000):
        super().__init__()
        self.linear = nn.Linear(dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)

def train(model, linear_classifier, optimizer, loader, epoch, device):
    linear_classifier.train()
    criterion = nn.CrossEntropyLoss()
    losses = AverageMeter('Loss', ':.4e')
    acc1 = AverageMeter('Acc@1', ':6.2f')
    acc5 = AverageMeter('Acc@5', ':6.2f')

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            output = model(images)

        output = linear_classifier(output)
        loss = criterion(output, target)

        # measure accuracy and record loss
        _, pred = output.topk(min(5, output.size(1)), 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in (1, min(5, output.size(1))):
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / images.size(0)))

        losses.update(loss.item(), images.size(0))
        acc1.update(res[0].item(), images.size(0))
        acc5.update(res[-1].item(), images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Train Epoch: {epoch} Loss: {losses.avg:.4f} Acc@1: {acc1.avg:.2f}% Acc@5: {acc5.avg:.2f}%')

@torch.no_grad()
def validate(model, linear_classifier, loader, epoch, device):
    linear_classifier.eval()
    criterion = nn.CrossEntropyLoss()
    losses = AverageMeter('Loss', ':.4e')
    acc1 = AverageMeter('Acc@1', ':6.2f')
    acc5 = AverageMeter('Acc@5', ':6.2f')

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        output = linear_classifier(output)
        loss = criterion(output, target)

        _, pred = output.topk(min(5, output.size(1)), 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in (1, min(5, output.size(1))):
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / images.size(0)))

        losses.update(loss.item(), images.size(0))
        acc1.update(res[0].item(), images.size(0))
        acc5.update(res[-1].item(), images.size(0))

    print(f'Test Epoch: {epoch} Loss: {losses.avg:.4f} Acc@1: {acc1.avg:.2f}% Acc@5: {acc5.avg:.2f}%')
    return acc1.avg

def build_eval_model(arch, patch_size=16, pretrained_weights="", device="cpu"):
    if arch in BACKBONES:
        model = BACKBONES.build(dict(type=arch, patch_size=patch_size, num_classes=0))
    elif hasattr(vits, arch):
        model = getattr(vits, arch)(patch_size=patch_size, num_classes=0)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model = model.to(device)

    if pretrained_weights and os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location='cpu', weights_only=False)
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        elif 'student' in state_dict:
            state_dict = state_dict['student']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if "backbone." in k or not k.startswith("head")}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_weights} (matched keys: {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected)")
    else:
        print(f"WARNING: No pretrained weights provided at {pretrained_weights}, using random initialization!")

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_data(data_path, dataset_name="image_folder", batch_size=256, num_workers=4):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if dataset_name.lower() == "cifar10":
        dataset_train = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_val)
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
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

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes

@torch.no_grad()
def extract_features(model, loader, device):
    features, labels = [], []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        out = model(images)
        features.append(out.cpu())
        labels.append(target)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', type=str)
    parser.add_argument('--dataset', default='image_folder', type=str, choices=['image_folder', 'cifar10', 'cifar100'])
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--arch', default='vit_small', type=str)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--cache_features', action='store_true', default=True, help='Cache frozen features for fast training')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, num_classes = load_data(args.data_path, args.dataset, args.batch_size, args.num_workers)
    print(f"Dataset '{args.dataset}' loaded with {num_classes} classes. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = build_eval_model(args.arch, args.patch_size, args.pretrained_weights, device)

    linear_classifier = LinearClassifier(model.embed_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(linear_classifier.parameters(), args.lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    criterion = nn.CrossEntropyLoss()

    if args.cache_features:
        print("Pre-extracting frozen features for train set...")
        train_features, train_labels = extract_features(model, train_loader, device)
        print("Pre-extracting frozen features for val set...")
        val_features, val_labels = extract_features(model, val_loader, device)

        # Create fast TensorDataLoaders
        train_dset = torch.utils.data.TensorDataset(train_features, train_labels)
        val_dset = torch.utils.data.TensorDataset(val_features, val_labels)
        fast_train_loader = DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)
        fast_val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)

        best_acc = 0.0
        for epoch in range(args.epochs):
            linear_classifier.train()
            losses = AverageMeter('Loss', ':.4e')
            acc1 = AverageMeter('Acc@1', ':6.2f')
            for feats, targets in fast_train_loader:
                feats, targets = feats.to(device), targets.to(device)
                outputs = linear_classifier(feats)
                loss = criterion(outputs, targets)
                _, pred = outputs.topk(1, 1, True, True)
                correct = pred.eq(targets.view(-1, 1)).float().sum()
                losses.update(loss.item(), feats.size(0))
                acc1.update((correct / feats.size(0) * 100.0).item(), feats.size(0))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Validation
            linear_classifier.eval()
            val_acc = AverageMeter('Acc@1', ':6.2f')
            with torch.no_grad():
                for feats, targets in fast_val_loader:
                    feats, targets = feats.to(device), targets.to(device)
                    outputs = linear_classifier(feats)
                    _, pred = outputs.topk(1, 1, True, True)
                    correct = pred.eq(targets.view(-1, 1)).float().sum()
                    val_acc.update((correct / feats.size(0) * 100.0).item(), feats.size(0))

            scheduler.step()
            if val_acc.avg > best_acc:
                best_acc = val_acc.avg
                os.makedirs("output", exist_ok=True)
                torch.save(linear_classifier.state_dict(), os.path.join("output", f"linear_best_{args.arch}.pth"))

            if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
                print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {losses.avg:.4f} Train Acc: {acc1.avg:.2f}% | Val Acc: {val_acc.avg:.2f}% (Best: {best_acc:.2f}%)")

        print(f"[{args.arch}] Linear Probe Best Top-1 Accuracy: {best_acc:.2f}%")
        return best_acc

    best_acc = 0.0
    for epoch in range(args.epochs):
        train(model, linear_classifier, optimizer, train_loader, epoch, device)
        acc1 = validate(model, linear_classifier, val_loader, epoch, device)
        scheduler.step()

        if acc1 > best_acc:
            best_acc = acc1
            os.makedirs("output", exist_ok=True)
            torch.save(linear_classifier.state_dict(), os.path.join("output", f"linear_best_{args.arch}.pth"))

    print(f"[{args.arch}] Best Top-1 Accuracy: {best_acc:.2f}%")
    return best_acc

if __name__ == '__main__':
    main()
