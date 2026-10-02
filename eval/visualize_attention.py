import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add parent dir to path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models.vision_transformer as vits

def get_args_parser():
    parser = argparse.ArgumentParser(description='Visualize self-attention maps')
    parser.add_argument('--image_path', default=None, type=str, help='Path to the image')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch size')
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_tiny_sparse_rope', 'vit_small_sparse_rope'])
    parser.add_argument('--pretrained_weights', default='', type=str, help='Path to pretrained weights')
    parser.add_argument('--output_dir', default='output', type=str, help='Path to save output images')
    return parser

def main():
    args = get_args_parser().parse_args()

    # Build model
    if args.arch == 'vit_small':
        model = vits.vit_small(patch_size=args.patch_size)
    elif args.arch == 'vit_tiny':
        model = vits.vit_tiny(patch_size=args.patch_size)
    elif args.arch == 'vit_small_sparse_rope':
        model = vits.vit_small_sparse_rope(patch_size=args.patch_size)
    elif args.arch == 'vit_tiny_sparse_rope':
        model = vits.vit_tiny_sparse_rope(patch_size=args.patch_size)
    else:
        raise ValueError(f"Unsupported arch: {args.arch}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load weights
    if os.path.isfile(args.pretrained_weights):
        state_dict = torch.load(args.pretrained_weights, map_location='cpu', weights_only=False)
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        elif 'student' in state_dict:
            state_dict = state_dict['student']
        # Remove wrapper prefixes if necessary
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if "backbone." in k or not k.startswith("head")}
        model.load_state_dict(state_dict, strict=False)
        print("Pretrained weights loaded.")
    else:
        print("WARNING: No pretrained weights provided, using random weights!")

    model.eval()

    # Read image
    if args.image_path is None or not os.path.isfile(args.image_path):
        # Generate dummy image for test
        print("Creating a dummy image for testing...")
        img = Image.fromarray(np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8))
        os.makedirs(args.output_dir, exist_ok=True)
        args.image_path = os.path.join(args.output_dir, 'dummy.jpg')
        img.save(args.image_path)

    with open(args.image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img_t = transform(img).unsqueeze(0).to(device)

    # Calculate attention
    w_featmap = img_t.shape[-2] // args.patch_size
    h_featmap = img_t.shape[-1] // args.patch_size

    with torch.no_grad():
        attn = model.get_last_selfattention(img_t) # (B, num_heads, N, N)

    # Get CLS token attention to all other tokens
    nh = attn.shape[1] # number of heads
    attn = attn[0, :, 0, 1:].reshape(nh, -1)

    # Reshape and upscale
    attn = attn.reshape(nh, w_featmap, h_featmap)
    attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # Plot
    os.makedirs(args.output_dir, exist_ok=True)
    img_resized = img.resize((224, 224))

    fig, axes = plt.subplots(1, nh + 1, figsize=(3 * (nh + 1), 3))
    axes[0].imshow(img_resized)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i in range(nh):
        axes[i+1].imshow(img_resized)
        axes[i+1].imshow(attn[i], cmap='cividis', alpha=0.8)
        axes[i+1].set_title(f'Head {i}')
        axes[i+1].axis('off')

    plt.tight_layout()
    output_path = os.path.join(args.output_dir, f'attn_map_{args.arch}.png')
    plt.savefig(output_path)
    print(f"Attention map saved to {output_path}")

if __name__ == '__main__':
    main()
