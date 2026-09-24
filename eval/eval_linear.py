import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.vision_transformer as vits
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
        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in (1, 5):
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / images.size(0)))

        losses.update(loss.item(), images.size(0))
        acc1.update(res[0].item(), images.size(0))
        acc5.update(res[1].item(), images.size(0))

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

        _, pred = output.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in (1, 5):
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / images.size(0)))

        losses.update(loss.item(), images.size(0))
        acc1.update(res[0].item(), images.size(0))
        acc5.update(res[1].item(), images.size(0))

    print(f'Test Epoch: {epoch} Loss: {losses.avg:.4f} Acc@1: {acc1.avg:.2f}% Acc@5: {acc5.avg:.2f}%')
    return acc1.avg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', type=str)
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--arch', default='vit_small', type=str)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform_train)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=transform_val)
    num_classes = len(dataset_train.classes)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.arch == 'vit_small':
        model = vits.vit_small(patch_size=args.patch_size, num_classes=0)
    elif args.arch == 'vit_tiny':
        model = vits.vit_tiny(patch_size=args.patch_size, num_classes=0)
    model = model.to(device)

    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location='cpu', weights_only=True)
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if "backbone." in k or not k.startswith("head")}
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    linear_classifier = LinearClassifier(model.embed_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(linear_classifier.parameters(), args.lr, momentum=0.9, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train(model, linear_classifier, optimizer, train_loader, epoch, device)
        acc1 = validate(model, linear_classifier, val_loader, epoch, device)
        scheduler.step()

        if acc1 > best_acc:
            best_acc = acc1
            os.makedirs("output", exist_ok=True)
            torch.save(linear_classifier.state_dict(), os.path.join("output", "linear_best.pth"))

    print(f"Best Top-1 Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
