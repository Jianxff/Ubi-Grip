# standard library
from pathlib import Path
import os, time
from typing import *
import numpy as np
import pickle
# torch
import torch
from torch.utils.data import Dataset
# opencv
import cv2

def convert_bbox(hand_bbox: np.ndarray) -> np.ndarray:
    cx, cy = (hand_bbox[0] + hand_bbox[2]) // 2, (hand_bbox[1] + hand_bbox[3]) // 2
    hbox_w, hbox_h = hand_bbox[2] - hand_bbox[0], hand_bbox[3] - hand_bbox[1]
    sz = max(hbox_w, hbox_h) * 1.5
    return np.array([cx, cy, sz, sz]).astype(np.int32)


class BaseSeq(Dataset):
    def __init__(
        self, 
        name: str,
        root: str,
        imsize: Tuple[int, int] = None,
        obj_name: str = None,
        obj_model_path: str = None,
        obj_meta_path: str = None,
        obj_unit_in_meter: float = 1.0,
        obj_from_opengl: bool = False,
        intrinsic: Optional[np.ndarray] = None, 
    ):
        super().__init__()
        self.root = Path(root)
        self.name = name

        self.imsize = imsize
        self.obj_name = obj_name
        self.obj_model_path = obj_model_path
        self.obj_meta_path = obj_meta_path
        self.intrinsic = intrinsic

        self.obj_unit_in_meter = obj_unit_in_meter
        self.obj_from_opengl = obj_from_opengl

    def get_object_pose(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_handbbox(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_rgb_image(self, idx):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def preload_data(self) -> None:
        """Preload data if necessary. This method should be overridden by subclasses."""
        pass

    def __len__(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def __getitem__(self, idx):
        rgb = self.get_rgb_image(idx)
        obj_pose = self.get_object_pose(idx)
        hand_pose = self.get_handbbox(idx)
        metadata = {
            'obj_pose': obj_pose,
            'hand_pose': hand_pose
        }

        return rgb, metadata