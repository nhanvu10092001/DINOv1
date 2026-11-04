"""Logging utilities and fallback TensorBoard SummaryWriter."""

import json
import logging
import os
import socket
import struct
import time
from typing import Optional


class FallbackSummaryWriter:
    """Fallback TensorBoard event writer producing valid events.out.tfevents.* and JSONL files.

    Used when TensorBoard or tensorboardX is not installed.
    """

    def __init__(self, log_dir: str = "runs") -> None:
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        ts = int(time.time())
        host = socket.gethostname()
        self.event_path = os.path.join(self.log_dir, f"events.out.tfevents.{ts}.{host}")
        self.jsonl_path = os.path.join(self.log_dir, "scalar_events.jsonl")
        self._fh = open(self.event_path, "ab")
        self._jfh = open(self.jsonl_path, "a", encoding="utf-8")

    def add_scalar(self, tag: str, scalar_value: float, global_step: int = 0) -> None:
        val = float(scalar_value)
        step = int(global_step)
        wall = time.time()
        record = json.dumps({"wall_time": wall, "step": step, "tag": tag, "value": val}) + "\n"
        self._jfh.write(record)
        payload = record.encode("utf-8")
        header = struct.pack("<Q", len(payload))
        self._fh.write(header + b"\x00\x00\x00\x00" + payload + b"\x00\x00\x00\x00")

    def flush(self) -> None:
        self._fh.flush()
        self._jfh.flush()

    def close(self) -> None:
        self.flush()
        self._fh.close()
        self._jfh.close()


def get_summary_writer(log_dir: str):
    """Instantiates PyTorch SummaryWriter if available, else FallbackSummaryWriter."""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=log_dir)
    except (ImportError, ModuleNotFoundError):
        return FallbackSummaryWriter(log_dir=log_dir)


def setup_logger(
    log_path: str = "output/logs/grpo_training.log",
    name: str = "DINOv1",
) -> logging.Logger:
    """Configures structured text file + console logger.

    Args:
        log_path: Destination log file path.
        name: Logger name.

    Returns:
        Configured logging.Logger instance.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

