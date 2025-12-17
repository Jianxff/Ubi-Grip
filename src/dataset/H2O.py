# standard library
from pathlib import Path
from typing import *
import numpy as np
# opencv
import cv2

from .base import BaseSeq, convert_bbox

# opencv to/from opengl
SE3_OPENCV_TO_OPENGL = np.array([
    [1, -1, -1, 1],
    [-1, 1, 1, -1],
    [-1, 1, 1, -1],
    [1, 1, 1, 1]
]).astype(np.float32)

SO3_OPENCV_TO_OPENGL = np.array([
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, 1]
]).astype(np.float32)

ROTMAT_OPENCV_TO_OPENGL = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
]).astype(np.float32)


# se3_OPENCV_TO_OPENGL = np.array([
#     [1, 0, 0, 0],
#     [0, -1, 0, 0],
#     [0, 0, -1, 0],
#     [0, 0, 0, 1]
# ]).astype(np.float32)


"""
H2O data organization:
    label_split
        action_test.txt
        ...
    object
        book
            book.obj
            book.mtl
            book.png
        ...
    subject1_ego
        h1
            0/cam4
                action_label
                    000000.txt
                    ...
                cam_pose
                    000000.txt
                    ...
                depth
                    000000.png
                    ...
                hand_pose
                    000000.txt
                    ...
                hand_pose_mano
                    000000.txt
                    ...
                obj_pose
                    000000.txt
                    ...
                obj_pose_rt
                    000000.txt
                    ...
                rgb
                    000000.png
                    ...
                rgb256
                    000000.jpg
                    ... 
                verb_label
                    000000.txt
                    ...
                cam_intrinsics.txt
            ...
        ...
"""

OBJ_LABEL_MAP = [
    "background",
    "book",
    "espresso",
    "lotion",
    "spray",
    "milk",
    "cocoa",
    "chips",
    "cappuccino"
]


class H2O_Seq(BaseSeq):
    def __init__(
        self,
        name: str,
        root: Union[str, Path],
        obj_model_dir: Union[str, Path],
        obj_meta_dir: Union[str, Path] = None,
    ) -> None:
        root = Path(root) / 'cam4'
        super().__init__(
            name=name, 
            root=root,
            obj_from_opengl=True,
            obj_unit_in_meter=1.0
        )

        self.rgb_dir = root / 'rgb'
        self.pose_dir = root / 'obj_pose_rt'
        self.hand_dir = root / 'hand_pose'

        # load rgb
        self.rgb = sorted(list(self.rgb_dir.glob('*.png')))
        self.pose = sorted(list(self.pose_dir.glob('*.txt')))
        self.hand = sorted(list(self.hand_dir.glob('*.txt')))
        assert len(self.rgb) == len(self.pose), 'Data length mismatch'

        # read intrinsic
        intr = np.loadtxt(root / 'cam_intrinsics.txt')
        fx, fy, cx, cy, w, h = intr[0], intr[1], intr[2], intr[3], intr[4], intr[5]
        # get object
        pose0 = np.loadtxt(self.pose[0])
        obj_name = OBJ_LABEL_MAP[int(pose0[0])]

        # preload data
        self.preloaded = False
        

        # Setting
        self.intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
        self.imsize = (int(h), int(w))
        self.obj_name = obj_name
        self.obj_model_path = str(Path(obj_model_dir) / obj_name / f"{obj_name}.obj")
        self.obj_meta_path = str(Path(obj_meta_dir) / f"{obj_name}.meta")

    
    def __len__(self) -> int:
        return len(self.rgb)
    

    def preload_data(self) -> None:
        self.preloaded = True
        self.rgb = [cv2.imread(str(img), cv2.IMREAD_COLOR)[..., ::-1] for img in self.rgb]
        self.pose = [np.loadtxt(pose) for pose in self.pose]
        self.hand = [np.loadtxt(hand) for hand in self.hand]


    def get_rgb_image(self, idx):
        return self.rgb[idx] if self.preloaded else \
            cv2.imread(str(self.rgb[idx]), cv2.IMREAD_COLOR)[..., ::-1]  # BGR to RGB
    

    def get_object_pose(self, idx):
        data = self.pose[idx] if self.preloaded else np.loadtxt(self.pose[idx])
        
        data = data[1:].reshape(4, 4)
        trans = data[:3, 3]
        rot = data[:3, :3]
        # rot = rot * SO3_OPENCV_TO_OPENGL
        rot = rot @ ROTMAT_OPENCV_TO_OPENGL   # OpenCV to OpenGL coordinate
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot
        pose[:3, 3] = trans
        return pose
    

    def get_handbbox(self, idx):
        data = self.hand[idx] if self.preloaded else np.loadtxt(self.hand[idx])
        
        ldata = data[:64]
        rdata = data[64:]

        #####################################################################
        #              !! ONLY USE RIGHT Hand FOR EVALUATION                #
        #####################################################################
        data = rdata
        if data[0] == 0:
            return None
        
        handJoints3D = data[1:].reshape(-1, 3)
        camMat = self.intrinsic
        handJoints2D = np.matmul(camMat, handJoints3D.T).T
        handJoints2D = handJoints2D[:, :2] / handJoints2D[:, 2:]
        handJoints2D = handJoints2D.astype(np.int32)
        x1, y1 = np.min(handJoints2D, axis=0)
        x2, y2 = np.max(handJoints2D, axis=0)
        return convert_bbox([x1, y1, x2, y2])


    def unload(self):
        if self.preloaded:
            del self.rgb
            del self.pose
            del self.hand

    def __del__(self):
        for img in self.rgb:
            if isinstance(img, np.ndarray):
                del img
        for pose in self.pose:
            if isinstance(pose, np.ndarray):
                del pose
        for hand in self.hand:
            if isinstance(hand, np.ndarray):
                del hand
        self.rgb = []
        self.pose = []
        self.hand = []