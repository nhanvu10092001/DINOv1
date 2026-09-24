import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.vision_transformer as vits

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
    imgs_per_chunk = num_test_images // num_chunks
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
        top5 = top5 + correct.narrow(1, 0, 5).sum().item()
        total += targets.size(0)
    return top1 / total * 100, top5 / total * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data', type=str)
    parser.add_argument('--pretrained_weights', default='', type=str)
    parser.add_argument('--arch', default='vit_small', type=str)
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--k', default=20, type=int)
    parser.add_argument('--temperature', default=0.07, type=float)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=transform)
    num_classes = len(dataset_train.classes)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, num_workers=4, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # Model
    if args.arch == 'vit_small':
        model = vits.vit_small(patch_size=args.patch_size, num_classes=0)
    elif args.arch == 'vit_tiny':
        model = vits.vit_tiny(patch_size=args.patch_size, num_classes=0)
    model = model.to(device)

    # Load weights
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location='cpu', weights_only=True)
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if "backbone." in k or not k.startswith("head")}
        model.load_state_dict(state_dict, strict=False)

    model.eval()

    print("Extracting training features...")
    train_features, train_labels = extract_features(model, train_loader, device)

    print("Extracting validation features...")
    val_features, val_labels = extract_features(model, val_loader, device)

    print("Evaluating with k-NN...")
    top1, top5 = knn_classifier(train_features, train_labels, val_features, val_labels, args.k, args.temperature, num_classes)
    print(f"k-NN (k={args.k}) Top-1: {top1:.2f}%, Top-5: {top5:.2f}%")

if __name__ == '__main__':
    main()
