# standard library
from pathlib import Path
import os, time, sys
from typing import *
# third party
import numpy as np
import torch
import cv2
import trimesh
from tqdm import tqdm


class Evaluator:
    def __init__(
        self,
        translation_threshold: float = np.inf,
        rotation_threshold: float = np.inf,
    ) -> None:
        self.data = []
        self.translation_threshold = translation_threshold
        self.rotation_threshold = rotation_threshold

    def update(
        self,
        trans_err: float,
        rot_err: float,
    ) -> int:
        self.data.append((trans_err, rot_err))
        return len(self.data)
    
    def avg_translation_error(self) -> float:
        return sum([T for T, _ in self.data]) / len(self.data)

    def avg_rotation_error(self) -> float:
        return sum([R for _, R in self.data]) / len(self.data)
    
    def avg_recall(self) -> float:
        return sum(
            [1 for T, R in self.data if 
                T < self.translation_threshold and R < self.rotation_threshold
            ]
        ) / len(self.data)


    def tostr(self) -> str:
        return f"AR: {self.avg_recall():.2f}, MTE: {self.avg_translation_error():.3f}, MRE: {self.avg_rotation_error():.2f}"
    
    def tocsv(self) -> str:
        return f"{self.avg_recall()},{self.avg_translation_error()},{self.avg_rotation_error()}"

    def save_raw(self, path: str) -> None:
        data = np.asarray(self.data, dtype=np.float32)
        np.save(path, data)


    @staticmethod
    def evaluate_loss(
        gt_pose: np.ndarray,
        pred_pose: np.ndarray,
    ) -> Tuple[float, float]:
        gt_trans = gt_pose[:3, 3]
        pred_trans = pred_pose[:3, 3]
        loss_trans_l2 = np.linalg.norm(x=(pred_trans - gt_trans), ord=2)

        # rotation error calculation
        gt_rotMat = gt_pose[:3, :3]
        pred_rot = pred_pose[:3, :3]
        acos_val = (np.trace(np.dot(gt_rotMat, pred_rot.T)) - 1) / 2
        if acos_val > 1: acos_val = 1
        if acos_val < -1: acos_val = -1
        loss_rot = np.arccos(acos_val) * 180 / np.pi

        return loss_trans_l2, loss_rot