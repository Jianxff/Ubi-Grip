# standard library
from pathlib import Path
import os, time
from typing import *
import numpy as np
import pickle
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




"""
HO3D data organization:
    calibration  
    evaluation   
        AP10
            depth  
            meta  
            rgb  
        ...
    evaluation.txt           
    manual_annotations  
    train
        ABF10
            depth  
            meta  
            rgb  
        ...
    train.txt
"""

""" 
YCB_Video models:
    models
        002_master_chef_can
            002_master_chef_can.xml
            points.xyz
            texture_map.png
            textured_simple.obj
            textured_simple.obj.mtl
            texutred.mtl
            textured.obj
        ...
"""

class HO3D_Seq(BaseSeq):
    def __init__(
        self,
        name: str,
        root: str,
        obj_model_dir: str,
        obj_meta_dir: str,
        seq_type: str,
    ) -> None:
        root = Path(root)
        super().__init__(
            name=name, 
            root=root,
            obj_from_opengl=True,
            obj_unit_in_meter=1.0
        )

        self.seq_type = seq_type
        # directory
        self.rgb_dir = root / 'rgb'
        self.meta_dir = root / 'meta'
        self.depth_dir = root / 'depth'
        # read data
        self.rgb = sorted(list(self.rgb_dir.glob('*.jpg')))
        self.meta = sorted(list(self.meta_dir.glob('*.pkl')))
        self.depth = sorted(list(self.depth_dir.glob('*.png')))
        # assert
        assert len(self.rgb) == len(self.meta) == len(self.depth), 'Data length mismatch'

        # basic data
        meta = self.get_meta(self.meta[0])
        obj_name = meta['objName']
        obj_label = meta['objLabel']
        # image size
        img = cv2.imread(str(self.rgb[0]))
        

        # Setting
        self.imsize = img.shape[:2]
        self.intrinsic = meta['camMat']
        self.obj_name = obj_name
        self.obj_model_path = str(Path(obj_model_dir) / obj_name / 'textured_simple.obj')
        self.obj_meta_path = str(Path(obj_meta_dir) / f'{obj_name}.meta')

    def read_metadata(self, path: Union[str, Path]) -> Dict:
        """
        meta:
            'objName': Object name
            'objRot': Object rotation matrix
            'objTrans': Object translation
            'objCorners3DRest': Object 3D corner locations
            'handPose': Hand pose (train)
            'handTrans': Hand translation (train)
            'handBeta': Hand beta (train)
            'handJoints3D': Hand 3D joint locations (evaluation)
            'handBoundingBox': Hand bounding box (evaluation)
            'camMat': Camera matrix
        """
        with open(path, 'rb') as f:
            try:
                meta = pickle.load(f, encoding='latin1')
            except:
                meta = pickle.load(f)
        return meta
    
    
    def __len__(self) -> int:
        return len(self.rgb)
    
    def get_rgb_image(self, idx):
        return cv2.imread(str(self.rgb[idx]))[..., ::-1]

    def get_handbbox(self, idx):
        meta = self.read_metadata(self.meta[idx])
        if self.seq_type == 'evaluation':
            return meta['handBoundingBox']
        handJoints3D = meta['handJoints3D']
        if handJoints3D is None:
            return None
        camMat = meta['camMat']
        handJoints2D = np.matmul(camMat, handJoints3D.T).T
        handJoints2D = handJoints2D[:, :2] / handJoints2D[:, 2:]
        handJoints2D = handJoints2D.astype(np.int32)
        x1, y1 = np.min(handJoints2D, axis=0)
        x2, y2 = np.max(handJoints2D, axis=0)
        return convert_bbox([x1, y1, x2, y2])
    
    def get_object_pose(self, idx):
        meta = self.read_metadata(self.meta[idx])
        # opengl
        trans = np.array(meta['objTrans']) 
        rot = np.array(meta['objRot'])
        rotMat = cv2.Rodrigues(rot)[0]
        # to opencv
        trans = trans * np.array([1, -1, -1]).astype(np.float32)
        rotMat = rotMat * SO3_OPENCV_TO_OPENGL
        
        pose = np.eye(4)
        pose[:3, 3] = trans
        pose[:3, :3] = rotMat
        return pose.astype(np.float32)



# class HO3D_Dataset:
#     def __init__(
#         self,
#         data_root: Union[str, Path],
#     ) -> None:
#         self.root = Path(data_root)
#         # evaluation
#         self.evaluations = self.parse_seqs('evaluation')
#         # train
#         self.trainings = self.parse_seqs('train')

    # def sequence(
    #     self,
    #     type: str, 
    #     seq: str
    # ) -> HO3D_Seq:
    #     if type == 'evaluation':
    #         return HO3D_Seq(self.root / 'evaluation' / seq, 'evaluation')
    #     elif type == 'train':
    #         return HO3D_Seq(self.root / 'train' / seq, 'train')
    #     else:
    #         raise ValueError('Unknown dataset type')

    # def parse_seqs(
    #     self,
    #     base_dir: Union[str, Path]
    # ) -> List[str]:
    #     dirs = [x for x in (self.root / base_dir).iterdir()]
    #     seqs = []
    #     for d in dirs:
    #         if d.is_dir(): seqs.append(d.name)
    #     return sorted(seqs)