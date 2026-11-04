"""Evaluation module for DINOv1."""

from eval.eval_utils import build_eval_model, load_eval_data
from eval.eval_knn import knn_classifier

__all__ = [
    "build_eval_model",
    "load_eval_data",
    "knn_classifier",
]
