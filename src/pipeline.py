# standard library
from typing import List, Dict, Any, Union, Tuple, Optional
from pathlib import Path
import os, time, sys
# third-party
import torch
import numpy as np
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R
# local
import pysrt3d
from .renderer import Renderer
from .hamer import HaMeR
from .mediapipe import HandDetector

# opencv to/from opengl
SE3_OPENCV_TO_OPENGL = np.array([
    [1, -1, -1, 1],
    [-1, 1, 1, -1],
    [-1, 1, 1, -1],
    [1, 1, 1, 1]
]).astype(np.float32)


class HandFilter:
    def __init__(
        self,
        hamer: HaMeR,
        imwidth: int,
        imheight: int,
    ) -> None:
        # engine initialization
        self.hamer = hamer
        self.renderer = Renderer(width=imwidth, height=imheight, focal=5000)
        # variables
        self.hand_mask = np.zeros((imheight, imwidth), np.uint8)
        self.result = None
        # record last pose
        self.last_rotation = None
        self.delta_rotation = np.eye(3)
        self.last_trans = None
        self.delta_trans = np.zeros(0)

    def update(
        self, image_rgb: np.ndarray, detections: Tuple[np.ndarray, bool]
    ) -> None:
        if self.result is not None:
            self.last_rotation = self.result['orient'].astype(np.float32)
            self.last_trans = self.result['bias'].astype(np.float32)
        # update hand reconstruction
        st = time.time()
        result = self.hamer.predict(image_rgb=image_rgb, detections=[detections])[0]
        time_pred = (time.time() - st) * 1000
        # print(f"Hand prediction time: {time_pred:.2f} ms")

        # print(result['bias'])
        st = time.time()
        hand_mesh = self.hamer.create_trimesh(result=result, is_right=detections[1])
        # render hand mesh
        self.renderer.add(name='hand', mesh=hand_mesh, color=self.renderer.COLOR_BLUE, replace=True)
        # update hand mask
        hand_mask = self.renderer.render_mask()
        self.hand_mask = cv2.dilate(hand_mask, np.ones((3, 3), np.uint8), iterations=1)
        time_render = (time.time() - st) * 1000
        # print(f"Hand mask rendering time: {time_render:.2f} ms")
        # update rotation
        rotation = result['orient'].astype(np.float32) # OpenCV coordinate
        trans = result['bias'].astype(np.float32)
        if self.last_rotation is not None:
            self.delta_rotation = rotation @ np.linalg.inv(self.last_rotation)
            self.delta_trans = trans - self.last_trans
        self.result = result
        return time_pred, time_render

    def render_hand(self) -> np.ndarray:
        return self.renderer.render_color()

    def reset(self) -> None:
        self.result = None
        self.delta_rotation = np.eye(3)
        self.delta_trans = np.zeros(0)
        self.last_rotation = None
        self.last_trans = None

    def __del__(self):
        del self.renderer


class ObjectTracker:
    def __init__(
        self,
        imwidth: int,
        imheight: int,
        name: str,
        model_path: Union[str, Path],
        meta_path: Union[str, Path],
        intrinsic: np.ndarray,
        model_unit_in_meter: float = 1.0,
        model_from_opengl: bool = False,
        use_renderer: Optional[bool] = False,
        kl_threshold: float = 1.0,
        tracking_thres_init: float = 0.0,
        tracking_thres_track: float = 0.0,
        overwrite_render_color: bool = False,
        debug: Optional[bool] = False
    ) -> None:
        self.obj_model = pysrt3d.Model(
            name=name, 
            model_path=str(model_path), 
            meta_path=str(meta_path),
            from_opengl=model_from_opengl, 
            unit_in_meter=model_unit_in_meter, 
            threshold_init = tracking_thres_init,   # no limit while initialization
            threshold_track = tracking_thres_track,  # no limit while tracking
            kl_threshold = kl_threshold,
            debug_visualize=False
        )
        self.obj_tracker = pysrt3d.Tracker(
            imwidth=imwidth, imheight=imheight, K=intrinsic
        )
        self.obj_tracker.add_model(self.obj_model)
        self.obj_tracker.setup()
        # variables
        self.last_pose = np.eye(4)
        self.pose = np.eye(4)
        self.pose_gl = np.eye(4)
        self.valid_percentage = 0.0
        self.use_renderer = use_renderer
        if use_renderer:
            self.renderer = Renderer(width=imwidth, height=imheight, intrinsic=intrinsic)
            self.renderer.add(
                name='obj', 
                mesh=trimesh.load_mesh(model_path), 
                color=Renderer.COLOR_GREEN if overwrite_render_color else None,
            )


    def update(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> None:
        # update variables
        self.last_pose = self.obj_model.pose.copy() # OpenCV coordinate
        # update object tracking
        st = time.time()
        self.obj_tracker.update(image=image, mask=mask)
        # print(f"Object tracking time: {(time.time() - st) * 1000:.2f} ms")

        self.valid_percentage = self.obj_model.valid_line_prop
        self.pose = self.obj_model.pose.copy() # OpenCV coordinate
        self.pose_gl = self.pose * SE3_OPENCV_TO_OPENGL
        # self.pose_gl = self.obj_model.pose_gl.copy() # OpenGL coordinate
        # update renderer
        if self.use_renderer:
            self.renderer.set('obj', pose=self.pose_gl)
    
    def reset(
        self, pose: np.ndarray
    ) -> None:
        self.obj_model.reset_pose(pose)
        self.last_pose = pose
        self.pose = pose
        self.pose_gl = pose * SE3_OPENCV_TO_OPENGL
        # update renderer
        if self.use_renderer:
            self.renderer.set('obj', pose=self.pose_gl)
    
    def render_mask(self) -> np.ndarray:
        if not self.use_renderer:
            raise ValueError('Renderer is not initialized')
        return self.renderer.render_mask()

    def render_image(self, image_rgb: np.ndarray) -> np.ndarray:
        if not self.use_renderer:
            raise ValueError('Renderer is not initialized')
        return self.renderer.render_image(image_rgb=image_rgb)

    def __del__(self):
        if self.use_renderer:
            del self.renderer
        del self.obj_tracker
        del self.obj_model


class HandObjectTrackingPipeline:
    def __init__(
        self,
        hamer_ckpt_path: Union[str, Path],
        mano_dir: Union[str, Path],
        device: Union[str, torch.device] = 'cuda',
    ) -> None:
        self.hamer = HaMeR(
            ckpt_path=hamer_ckpt_path,
            mano_dir=mano_dir,
            device=device
        )
        self.hand_detector = HandDetector(gpu_delegate=False)

    def init(
        self,
        imwidth: int,
        imheight: int,
        intrinsic: np.ndarray,
        obj_model_path: Union[str, Path],
        obj_meta_path: Union[str, Path],
    ) -> None:
        self.hand_filter = HandFilter(
            hamer=self.hamer,
            imwidth=imwidth,
            imheight=imheight
        )

        self.object_tracker = ObjectTracker(
            imwidth=imwidth,
            imheight=imheight,
            name='object',
            model_path=obj_model_path,
            meta_path=obj_meta_path,
            intrinsic=intrinsic,
            use_renderer=False,
            debug=False
        )

        self.im_w = imwidth
        self.im_h = imheight


    def reset_object_pose(
        self,
        pose: np.ndarray
    ) -> None:
        self.object_tracker.reset(pose=pose)


    def update(
        self,
        frame: np.ndarray,
    ) -> None:
        # detect 2D hand
        detect_result = self.hand_detector.predict(image_rgb=frame)
        hand_det = self.hand_detector.get_detect_bbox(
            detect_result, image_HW=(self.im_h, self.im_w), force_side='Left')

        # update hand 3d prediction
        hand_mask = None
        if len(hand_det) > 0:
            self.hand_filter.update(image_rgb=frame, detections=hand_det[0])
            hand_mask = self.hand_filter.hand_mask

        # update object tracking
        self.object_tracker.update(image=frame, mask=hand_mask)

        # reset object pose if no hand detected
        if self.objtracker.valid_percentage < 0.5:
            last_rotation = self.objtracker.last_pose[:3, :3].copy()
            last_trans = self.objtracker.last_pose[:3, 3].copy()
            cur_trans = self.objtracker.pose[:3, 3].copy()
            opt_rotation = self.hand_filter.delta_rotation @ last_rotation
            pred_pose = np.eye(4)
            pred_pose[:3, :3] = opt_rotation
            pred_pose[:3, 3] = (cur_trans + last_trans) / 2
            self.objtracker.reset(pred_pose)
        
        return self.hand_filter.result, self.object_tracker.pose



