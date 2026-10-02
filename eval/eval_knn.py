import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models
import models.vision_transformer as vits
from core.registry import BACKBONES

@torch.no_grad()
def extract_features(model, data_loader, device):
    features = None
    labels = None
    for images, target in tqdm(data_loader, desc="Extracting features"):
        images = images.to(device, non_blocking=True)
        out = model(images)
        out = torch.nn.functional.normalize(out, dim=1, p=2)

        if features is None:
            features = out.cpu()
            labels = target.cpu()
        else:
            features = torch.cat((features, out.cpu()), dim=0)
            labels = torch.cat((labels, target.cpu()), dim=0)

    return features, labels

@torch.no_grad()
def knn_classifier(train_features, train_labels, test_features, test_labels, k, T, num_classes):
    top1, top5, total = 0.0, 0.0, 0
    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 100
    imgs_per_chunk = max(1, num_test_images // num_chunks)
    retrieval_one_hot = torch.zeros(k, num_classes)
    for idx in range(0, num_test_images, imgs_per_chunk):
        features = test_features[idx : min((idx + imgs_per_chunk), num_test_images)]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = torch.exp(distances / T)
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        top5 = top5 + correct.narrow(1, 0, min(5, num_classes)).sum().item()
        total += targets.size(0)
    return top1 / total * 100, top5 / total * 100

def build_eval_model(arch, patch_size=16, pretrained_weights="", device="cpu"):
    if arch in BACKBONES:
        model = BACKBONES.build(dict(type=arch, patch_size=patch_size, num_classes=0))
    elif hasattr(vits, arch):
        model = getattr(vits, arch)(patch_size=patch_size, num_classes=0)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    model = model.to(device)

    # Load weights
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
    return model

def load_data(data_path, dataset_name="image_folder", batch_size=128, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if dataset_name.lower() == "cifar10":
        dataset_train = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
        dataset_val = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name.lower() == "cifar100":
        dataset_train = datasets.CIFAR100(root=data_path, train=True, download=True, transform=transform)
        dataset_val = datasets.CIFAR100(root=data_path, train=False, download=True, transform=transform)
        num_classes = 100
    else:
        train_path = os.path.join(data_path, "train")
        val_path = os.path.join(data_path, "val")
        if not os.path.exists(val_path):
            val_path = os.path.join(data_path, "validation")
        if not os.path.exists(train_path) or not os.path.exists(val_path):
            raise FileNotFoundError(f"Cannot find train/val directories under {data_path}")
        dataset_train = datasets.ImageFolder(train_path, transform=transform)
        dataset_val = datasets.ImageFolder(val_path, transform=transform)
        num_classes = len(dataset_train.classes)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', type=str)
    parser.add_argument('--dataset', default='image_folder', type=str, choices=['image_folder', 'cifar10', 'cifar100'])
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--arch', default='vit_small', type=str)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--k', default=20, type=int)
    parser.add_argument('--temperature', default=0.07, type=float)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, num_classes = load_data(args.data_path, args.dataset, args.batch_size, args.num_workers)
    print(f"Dataset '{args.dataset}' loaded with {num_classes} classes. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = build_eval_model(args.arch, args.patch_size, args.pretrained_weights, device)

    print("Extracting training features...")
    train_features, train_labels = extract_features(model, train_loader, device)

    print("Extracting validation features...")
    val_features, val_labels = extract_features(model, val_loader, device)

    print("Evaluating with k-NN...")
    top1, top5 = knn_classifier(train_features, train_labels, val_features, val_labels, args.k, args.temperature, num_classes)
    print(f"[{args.arch}] k-NN (k={args.k}) Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")
    return top1, top5

if __name__ == '__main__':
    main()
