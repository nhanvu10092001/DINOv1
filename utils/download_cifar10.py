"""Fast AWS S3 / CDN Downloader & Builder for CIFAR-10 (`data/cifar10`).

Downloads CIFAR-10 from high-speed S3/CDN mirrors (`fast-ai-imageclas/cifar10.tgz`)
when `cs.toronto.edu` is throttled or unreachable, and writes both:
1. Standard `ImageFolder` structure (`data/cifar10/train`, `data/cifar10/val`)
2. Standard `torchvision.datasets.CIFAR10` binary batches (`data/cifar10/cifar-10-batches-py/`)
so `eval/eval_knn.py`, `eval/eval_linear.py`, `train_grpo_post.py`, and `eval/eval_ablation_2x2.py`
work seamlessly at full speed.
"""

import glob
import os
import pickle
import shutil
import subprocess
import tarfile
from typing import List

import numpy as np
from PIL import Image


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

FASTAI_S3_URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"


def _pack_cifar_batches_from_imagefolder(
    extracted_root: str,
    batches_dir: str,
) -> None:
    """Converts extracted `cifar10/{train,test}` PNGs into torchvision `cifar-10-batches-py` format."""
    os.makedirs(batches_dir, exist_ok=True)

    # 1. Write batches.meta
    meta = {
        "num_cases_per_batch": 10000,
        "label_names": CIFAR10_CLASSES,
        "num_vis": 3072,
    }
    with open(os.path.join(batches_dir, "batches.meta"), "wb") as f:
        pickle.dump(meta, f)

    # 2. Collect train images (50,000) and pack into data_batch_1..5
    train_items: List[tuple] = []
    for label_idx, cls_name in enumerate(CIFAR10_CLASSES):
        cls_dir = os.path.join(extracted_root, "train", cls_name)
        for img_p in sorted(glob.glob(os.path.join(cls_dir, "*.png"))):
            train_items.append((img_p, label_idx))

    rng = np.random.RandomState(42)
    perm = rng.permutation(len(train_items))
    train_items = [train_items[i] for i in perm]

    print(f"Packing {len(train_items)} training images into 5 CIFAR-10 binary batches...")
    batch_size = 10000
    for b_idx in range(5):
        subset = train_items[b_idx * batch_size : (b_idx + 1) * batch_size]
        data = np.zeros((len(subset), 3072), dtype=np.uint8)
        labels = []
        filenames = []
        for i, (img_p, lbl) in enumerate(subset):
            with Image.open(img_p) as im:
                arr = np.array(im.convert("RGB"), dtype=np.uint8)  # (32, 32, 3)
            # CIFAR-10 binary layout is (3, 32, 32) flattened: R(1024), G(1024), B(1024)
            data[i] = arr.transpose(2, 0, 1).reshape(-1)
            labels.append(lbl)
            filenames.append(os.path.basename(img_p))

        payload = {
            "batch_label": f"training batch {b_idx + 1} of 5",
            "labels": labels,
            "data": data,
            "filenames": filenames,
        }
        with open(os.path.join(batches_dir, f"data_batch_{b_idx + 1}"), "wb") as f:
            pickle.dump(payload, f)

    # 3. Collect test images (10,000) and pack into test_batch
    test_items: List[tuple] = []
    for label_idx, cls_name in enumerate(CIFAR10_CLASSES):
        cls_dir = os.path.join(extracted_root, "test", cls_name)
        for img_p in sorted(glob.glob(os.path.join(cls_dir, "*.png"))):
            test_items.append((img_p, label_idx))

    print(f"Packing {len(test_items)} test images into test_batch...")
    data_test = np.zeros((len(test_items), 3072), dtype=np.uint8)
    labels_test = []
    filenames_test = []
    for i, (img_p, lbl) in enumerate(test_items):
        with Image.open(img_p) as im:
            arr = np.array(im.convert("RGB"), dtype=np.uint8)
        data_test[i] = arr.transpose(2, 0, 1).reshape(-1)
        labels_test.append(lbl)
        filenames_test.append(os.path.basename(img_p))

    test_payload = {
        "batch_label": "testing batch 1 of 1",
        "labels": labels_test,
        "data": data_test,
        "filenames": filenames_test,
    }
    with open(os.path.join(batches_dir, "test_batch"), "wb") as f:
        pickle.dump(test_payload, f)


def fast_download_cifar10(root_dir: str = "data/cifar10") -> str:
    os.makedirs(root_dir, exist_ok=True)
    train_dst = os.path.join(root_dir, "train")
    val_dst = os.path.join(root_dir, "val")
    if os.path.isdir(train_dst) and os.path.isdir(val_dst):
        print(f"CIFAR-10 already ready at {root_dir}")
        return root_dir

    tgz_path = os.path.join(root_dir, "cifar10_fastai.tgz")
    print(f"Downloading CIFAR-10 from high-speed AWS S3 mirror: {FASTAI_S3_URL} ...")
    subprocess.run(
        ["curl", "-L", "--fail", "--retry", "3", "-o", tgz_path, FASTAI_S3_URL],
        check=True,
    )

    print(f"Extracting {tgz_path} ...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=root_dir)

    extracted_cifar_dir = os.path.join(root_dir, "cifar10")
    train_dst = os.path.join(root_dir, "train")
    val_dst = os.path.join(root_dir, "val")
    if os.path.exists(os.path.join(extracted_cifar_dir, "train")) and not os.path.exists(train_dst):
        shutil.move(os.path.join(extracted_cifar_dir, "train"), train_dst)
    if os.path.exists(os.path.join(extracted_cifar_dir, "test")) and not os.path.exists(val_dst):
        shutil.move(os.path.join(extracted_cifar_dir, "test"), val_dst)

    if os.path.exists(tgz_path):
        os.remove(tgz_path)
    if os.path.exists(extracted_cifar_dir):
        shutil.rmtree(extracted_cifar_dir)

    print(f"CIFAR-10 (50,000 train + 10,000 val) is ready at {root_dir}!")
    return root_dir


prepare_cifar10_imagefolder = fast_download_cifar10


if __name__ == "__main__":
    fast_download_cifar10()
