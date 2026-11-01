import os
import argparse
from tqdm import tqdm
from datasets import load_dataset

def download_and_prepare(dataset_name, output_dir, max_samples=0):
    print(f"Loading dataset '{dataset_name}' from Hugging Face Hub...", flush=True)
    if max_samples and max_samples > 0:
        ds_stream = load_dataset(dataset_name, split="train", streaming=True)
        split_dir = os.path.join(output_dir, "train")
        os.makedirs(split_dir, exist_ok=True)
        features = getattr(ds_stream, "features", None) or {}
        label_col = "label" if "label" in features else "labels"
        class_names = getattr(features.get(label_col, None), "names", None)
        print(f"Streaming {max_samples} real images from '{dataset_name}' (train) -> {split_dir}", flush=True)
        for i, item in enumerate(tqdm(ds_stream, total=max_samples)):
            if i >= max_samples:
                break
            img = item["image"]
            label_idx = item.get(label_col, item.get("label", 0))
            if class_names and 0 <= int(label_idx) < len(class_names):
                class_name = str(class_names[int(label_idx)]).replace("/", "_")
            else:
                class_name = f"class_{int(label_idx):04d}"
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_path = os.path.join(class_dir, f"img_{i:08d}.jpg")
            img.save(img_path, "JPEG", quality=95)
        print(f"\nDataset '{dataset_name}' ({max_samples} images) successfully prepared at {output_dir}", flush=True)
        return

    # Load dataset
    ds = load_dataset(dataset_name)

    # Process each split (e.g., 'train', 'validation')
    for split in ds.keys():
        print(f"\nProcessing split: {split} (Total: {len(ds[split])} images)")
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # We need to map labels to folder names.
        # Check if 'label' or 'labels' is in the features
        label_col = 'label' if 'label' in ds[split].features else 'labels'

        # Try to get class names if available
        features = ds[split].features[label_col]
        class_names = None
        if hasattr(features, 'names'):
            class_names = features.names

        for i, item in enumerate(tqdm(ds[split])):
            img = item['image']
            label_idx = item[label_col]

            # Format class name or index
            if class_names:
                class_name = str(class_names[label_idx]).replace('/', '_')
            else:
                class_name = f"class_{label_idx:04d}"

            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

            # Convert RGBA or L to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            img_path = os.path.join(class_dir, f"img_{i:08d}.jpg")
            img.save(img_path, "JPEG", quality=100)

    print(f"\nDataset '{dataset_name}' successfully prepared at {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare ImageNet subset from Hugging Face")
    parser.add_argument("--dataset", type=str, default="clane9/imagenet-100",
                        help="Hugging Face dataset ID (e.g., clane9/imagenet-100 or Maysee/tiny-imagenet)")
    parser.add_argument("--output_dir", type=str, default="data",
                        help="Output directory to save the ImageFolder structured dataset")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Optional max train samples to stream quickly (0 = full dataset)")
    args = parser.parse_args()

    download_and_prepare(args.dataset, args.output_dir, max_samples=args.max_samples)
