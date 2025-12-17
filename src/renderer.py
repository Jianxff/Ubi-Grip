### standard library
import os
from typing import *
from pathlib import Path
### third party
import numpy as np
import pyrender
import trimesh
# import open3d as o3d
import cv2

os.environ['PYOPENGL_PLATFORM'] = 'egl'

class Renderer:
    COLOR_BLUE = (0.65, 0.74, 0.86)
    COLOR_GREEN = (0.65, 0.86, 0.74)
    COLOR_WHITE = (1.0, 1.0, 1.0)
    
    def __init__(
        self,
        width: int,
        height: int, 
        focal: float = None,
        intrinsic: Optional[np.ndarray] = None,
        znear: Optional[float] = 0.01,
        zfar: Optional[float] = 100
    ) -> None:
        # init offscreen render pipe
        self.render_pipe = pyrender.OffscreenRenderer(
            viewport_width=width,
            viewport_height=height,
            point_size=1.0
        )
        # init scene
        self.scene = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        # nodes
        self.nodes = {}
        # init camera
        if intrinsic is None:
            assert focal is not None, 'focal length must be provided if intrinsic is None'
            fx, fy, cx, cy = focal, focal, width / 2., height / 2.
        elif len(intrinsic) == 4:
            fx, fy, cx, cy = intrinsic
        elif isinstance(intrinsic, np.ndarray):
            fx, fy = intrinsic[0, 0], intrinsic[1, 1]
            cx, cy = intrinsic[0, 2], intrinsic[1, 2]
        camera_pose = np.eye(4)
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=znear, zfar=zfar)
        # add it to pyRender scene
        self.cam_node = pyrender.Node(camera=camera, matrix=camera_pose)
        self.scene.add_node(self.cam_node)
        # self.scene.add(camera, pose=camera_pose)
        # add light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
        self.scene.add(light)
        self.nodes.clear()

    def __del__(self) -> None:
        for node in self.nodes.values():
            self.scene.remove_node(node)
            del node
        self.scene.clear()
        self.nodes.clear()
        self.render_pipe.delete()


    def add(
        self, 
        name: str, 
        mesh: trimesh.Trimesh, 
        color: Optional[Tuple[float]] = None,
        replace: Optional[bool] = False
    ) -> pyrender.Node:
        # replace old node
        if replace:
            self.remove(name)
        # create pyrender mesh
        if color is not None:
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode='OPAQUE',
                baseColorFactor=color,
            )
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        else:
            mesh = pyrender.Mesh.from_trimesh(mesh)

        # create node
        mesh_node = pyrender.Node(mesh=mesh, name=name, matrix=np.eye(4))
        # add node to scene
        self.scene.add_node(mesh_node)
        self.nodes[name] = mesh_node

        return mesh_node
    
    def set(
        self, 
        name: str, 
        pose: Optional[np.ndarray] = None,
        visible: Optional[bool] = None
    ) -> None:
        node = self.nodes.get(name, None)
        if node is not None:
            if pose is not None:
                self.scene.set_pose(node, pose)
            if visible is not None:
                node.mesh.is_visible = visible
    
    def get(self, name: str) -> pyrender.Node:
        return self.nodes.get(name, None)
    
    def remove(self, name) -> None:
        node = self.nodes.get(name, None)
        if node is not None:
            self.scene.remove_node(node)
            del self.nodes[name]

    def render_mask(
        self
    ) -> np.ndarray:
        color, _ = self.render_pipe.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        mask = (color[..., 3:] > 0).astype(np.uint8) * 255
        mask = np.squeeze(mask)
        return mask
    
    def render_sideview(
        self,
        pose_t: List[float] = [0.3, 0, -0.3],
        pose_r: List[List[float]] = [[0, 0, 1],[0, 1, 0],[-1, 0, 0]]
    ) -> np.ndarray:
        side_pose = np.eye(4)
        side_pose[:3, 3] = np.array(pose_t)
        # look to the negative x-axis
        side_pose[:3, :3] = np.array(pose_r)
        self.scene.set_pose(self.cam_node, side_pose)
        color, _ = self.render_pipe.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        self.scene.set_pose(self.cam_node, np.eye(4))
        color = cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
        return color

    def render_image(
        self,
        image_rgb: np.ndarray
    ) -> np.ndarray:
        color, _ = self.render_pipe.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        # convert to float type
        color = color.astype(np.float32) / 255.0
        image = image_rgb.astype(np.float32) / 255.0
        # convert to RGBA
        image = np.concatenate([
            image, np.ones_like(image[..., :1])
        ], axis=2)
        # overlay render image and original image
        image_overlay = image[:,:,:3] * (1 - color[:,:,3:]) + color[:,:,:3] * color[:,:,3:]
        return (image_overlay * 255).astype(np.uint8)
    
    def render_color(
        self,
        color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        self.scene.bg_color = color
        color, _ = self.render_pipe.render(self.scene, flags=pyrender.RenderFlags.RGBA)
        self.scene.bg_color=[0.0, 0.0, 0.0, 0.0]
        self.scene.set_pose(self.cam_node, np.eye(4))
        return cv2.cvtColor(color, cv2.COLOR_RGBA2RGB)
    