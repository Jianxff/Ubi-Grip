### standard library
from typing import *
from pathlib import Path
### third party
import numpy as np
import torch
from PIL import Image
import cv2
import time
import trimesh
### hamer
from .models import load_hamer, download_models

def crop_image_numpy(
    image: np.ndarray,
    center: Tuple[int, int],
    dsize: Tuple[int, int],
    padding: Optional[str] = 'edge',
    **kwargs,
) -> np.ndarray:
    """
    Crop image with center and size.
    Args:
        image: input image (H, W, C) or (H, W)
        center: (x, y) center of the crop
        dsize: (width, height) size of the crop
        padding: padding size
    Returns:
        cropped image
    """
    gray_image = (len(image.shape) == 2)
    if gray_image:  image = image[:, :, None]
    
    h, w = image.shape[:2]
    u, v = center
    dw, dh = dsize
    # bbox
    u0, u1 = u - dw // 2, u + dw // 2
    v0, v1 = v - dh // 2, v + dh // 2
    bbox = [max(v0, 0), min(v1, h), max(u0, 0), min(u1, w)]
    image = image[bbox[0]:bbox[1], bbox[2]:bbox[3], :]
    # padding
    pad_v = (abs(v0) - bbox[0], abs(v1) - bbox[1])
    pad_u = (abs(u0) - bbox[2], abs(u1) - bbox[3])
    image = np.pad(image, 
        [pad_v, pad_u, (0,0)], 
        mode=padding, **kwargs
    )

    if gray_image:  image = image[:, :, 0]

    return image


LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

MANO_FACES_NEW = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239], [239, 122, 279],
                        [122, 118, 279], [279, 118, 215], [118, 117, 215], [215, 117, 214],
                        [117, 119, 214], [214, 119, 121], [119, 120, 121], [121, 120, 78],
                        [120, 108, 78], [78, 108, 79]])


class HaMeR:
    model = None            # hamer model

    def __init__(
        self, 
        ckpt_path: Union[str, Path],
        mano_dir: Union[str, Path],
        device: Union[str, torch.device] = 'cuda',
    ) -> None:
        self.device = torch.device(device)
        # load hamer model
        # ckpt_path = PROJECT_CACHE_DIR / 'hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt'
        # mano_dir = PROJECT_MANO_DATA_DIR
        # if not ckpt_path.exists():  download_models(PROJECT_CACHE_DIR / 'hamer')
        
        # model init
        model, model_cfg = load_hamer(str(ckpt_path), str(mano_dir))
        model.half()
        self.model = model.to(self.device)
        self.model.eval()
        self.model_cfg = model_cfg
        self.FOCAL = 5000

        # hamer config
        self.img_size = model_cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(model_cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(model_cfg.MODEL.IMAGE_STD)
        self.device = device

        # mano faces
        self.mano_faces = np.concatenate([model.mano.faces, MANO_FACES_NEW], axis=0)
        self.mano_faces_left = self.mano_faces[:, [0, 2, 1]]

    
    def patch_data(
        self, 
        image_rgb: np.ndarray, 
        bbox: Tuple[int, int, int, int],
        is_right: bool
    ) -> Tuple[Dict, Dict]:
        H, W = image_rgb.shape[:2]

        ### patch bounding box
        # image_patch, _ = generate_image_patch_cv2(
        #     img=image_rgb, c_x = bbox[0], c_y = bbox[1], bb_width=bbox[2], bb_height=bbox[3],
        #     patch_width=self.img_size, patch_height=self.img_size,
        #     do_flip=(not is_right), scale=1.0, rot=0,
        #     border_mode=cv2.BORDER_CONSTANT
        # )
        # scale = self.img_size / max(H, W)

        ### crop and resize
        # crop to bounding box
        image_patch = crop_image_numpy(image=image_rgb,
            center=(bbox[0], bbox[1]), dsize=(bbox[2], bbox[3]),
            padding='constant',
        )

        # resize to model input size
        scale = self.img_size / max(image_patch.shape[:2])
        image_patch = cv2.resize(image_patch, (0, 0), fx=scale, fy=scale)
        if not is_right:
            image_patch = cv2.flip(image_patch, 1)

        ### convert image strucutre
        image_patch = np.transpose(image_patch, (2, 0, 1)) # HWC to CHW
        image_patch = image_patch.astype(np.float32)
        # apply normalization
        for c in range(3):
            image_patch[c, :, :] = (image_patch[c, :, :] - self.mean[c]) / self.std[c]
        # convert to tensor
        image_torch = torch.from_numpy(image_patch).unsqueeze(0).half()

        # make extra data
        # scale = self.img_size / max([H, W])
        extra = {
            'is_right': is_right,
            'bbox': bbox,       # xywh
            'img_size': (H, W), # HW
            'scale': self.img_size / max(H, W), # scale factor
        }

        ### transform focal length
        # scaled_focal = focal_length * scale

        return {'img' : image_torch.to(self.device), 'focal': self.FOCAL}, extra
    

    def recover_camera(
        self,
        pred_cam: torch.Tensor,
        bbox: np.ndarray,
        img_size: Tuple[int, int],
        scale: float,
        focal_length: float = 5000
    ) -> np.ndarray:
        cx, cy, b, _ = bbox
        h, w = img_size
        w_2, h_2 = w / 2., h / 2.
        # focal transform
        focal = self.FOCAL / scale
        # weak perspective transform
        bs = b * pred_cam[0] + 1e-9
        tz = 2 * focal_length / bs
        tx = (2 * (cx - w_2) / bs) + pred_cam[1]
        ty = (2 * (cy - h_2) / bs) + pred_cam[2]

        # cam_full = torch.tensor([tx, ty, tz * (focal_length / focal)], device=pred_cam.device, dtype=pred_cam.dtype)
        cam_full = torch.tensor([tx, ty, tz], device=pred_cam.device, dtype=pred_cam.dtype)
        return cam_full

    def extract_mano(
        self,
        prediction: Dict,
        extra_info: Dict,
        focal_length: Optional[float] = 5000
    ) -> Dict:
        is_right = extra_info['is_right']
        # get vertices and pred_cam
        kpts3d = prediction['pred_keypoints_3d'][0].detach().cpu().numpy()
        # kpts2d = prediction['pred_keypoints_2d'][0].detach().cpu().numpy()
        orient = prediction['pred_mano_params']['global_orient'][0].detach().cpu().numpy()
        verts = prediction['pred_vertices'][0].detach().cpu().numpy()

        camera = prediction['pred_cam'][0].detach()
        if not is_right:
            verts[:, 0] *= -1
            camera[1] *= -1
            orient[:, 0] *= -1
        
        bias = self.recover_camera(
            camera,
            extra_info['bbox'],
            extra_info['img_size'],
            extra_info['scale'],
            focal_length
        ).cpu().numpy()

        return {
            'verts': verts + bias,
            'kps3d': kpts3d + bias,
            'bias': bias,
            'orient': orient,
            'is_right': is_right
        }
    

    @torch.no_grad()
    def predict(
        self,
        image_rgb: Union[np.ndarray, str, Path],
        detections: Optional[List[Tuple[Tuple, bool]]],
        focal_length: Optional[float] = 5000,
    ) -> List[Dict]:
        if isinstance(image_rgb, (str, Path)):
            image_rgb = np.array(Image.open(image_rgb))

        if not isinstance(detections, list):
            detections = [detections]
        if detections is None or len(detections) == 0:
            return []

        results = []
        for bbox, is_right in detections:
            # patch data
            data, extra_info = self.patch_data(image_rgb, bbox, is_right)
            # inference
            out = self.model(data)
            """
            Data structure of model output:
                pred_cam
                pred_mano_params
                    hand_pose
                    global_orient
                    betas
                pred_cam_t
                focal_length
                pred_keypoints_3d
                pred_vertices
                pred_keypoints_2d
            """
            
            # ## debug ##
            # debug_res = self.render_result(data['img'][0], out)
            # cv2.imwrite('debug.jpg', debug_res)
            # ## 
            
            # verts
            res = self.extract_mano(out, extra_info, focal_length)
            res['bbox'] = bbox
            results.append(res)

        return results

    
    @staticmethod
    def convert_verts_to_opengl(verts: np.ndarray) -> np.ndarray:
        verts_opengl = verts.copy()
        verts_opengl[:, 1] *= -1
        verts_opengl[:, 2] *= -1
        return verts_opengl

    def create_trimesh(
        self,
        result: Dict = None,
        is_right: Optional[bool] = True,
        color: Optional[Tuple] = LIGHT_BLUE,
    ) -> trimesh.Trimesh:
        assert result is not None or verts is not None, 'Either result or verts must be provided'
        
        verts = result['verts'] * np.array([1, -1, -1]).astype(np.float32)

        verts_color = np.array([(*color, 1)] * verts.shape[0])
        if is_right:
            mesh = trimesh.Trimesh(vertices=verts, faces=self.mano_faces.copy(), vertex_colors=verts_color)
        else:
            mesh = trimesh.Trimesh(vertices=verts, faces=self.mano_faces_left.copy(), vertex_colors=verts_color)
        return mesh
    

    @torch.no_grad()
    def predict_params(
        self,
        image_rgb: Union[np.ndarray, str, Path],
        detections: Optional[List[Tuple[Tuple, bool]]],
        focal_length: Optional[float] = 5000,
    ) -> List[Dict]:
        if isinstance(image_rgb, (str, Path)):
            image_rgb = np.array(Image.open(image_rgb))
        if not isinstance(detections, list):
            detections = [detections]
        if detections is None or len(detections) == 0:
            return []
        results = []
        for bbox, is_right in detections:
            # patch data
            data, extra_info = self.patch_data(image_rgb, bbox, is_right)
            # inference
            out = self.model.forward_parameter(data)
            """
            Data structure of model params output:
                pred_cam
                pred_mano_params
                    hand_pose
                    global_orient
                    betas
            """
            pred_cam = out['pred_cam'][0]
            cam_full = self.recover_camera(
                pred_cam,
                extra_info['bbox'],
                extra_info['img_size'],
                extra_info['scale'],
                focal_length
            )
            params = out['pred_mano_params']
            results.append({
                'pred_cam_full': cam_full,
                'pred_mano_params': params,
                'is_right': is_right
            })
        return results

    
    # def create_geometry(
    #     self,
    #     verts: np.ndarray,
    #     is_right: Optional[bool] = True,
    #     color: Optional[Tuple] = LIGHT_BLUE,
    # ) -> open3d.geometry.TriangleMesh:
    #     triangles = self.mano_faces.copy() if is_right else self.mano_faces_left.copy()
    #     mesh = open3d.geometry.TriangleMesh(
    #         vertices=open3d.utility.Vector3dVector(verts), 
    #         triangles=open3d.utility.Vector3iVector(triangles)
    #     )
    #     mesh.compute_vertex_normals()
    #     mesh.paint_uniform_color(color)
    #     return mesh
        

    # def render_result(
    #     self,
    #     image_rgb: torch.Tensor,
    #     out
    # ) -> np.ndarray:
    #     from .hamer.utils.renderer import Renderer

    #     renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
    #     regression_img = renderer(
    #         out['pred_vertices'][0].detach().cpu().numpy(),
    #         out['pred_cam_t'][0].detach().cpu().numpy(),
    #         out['focal_length'][0][0],
    #         image_rgb,
    #         mesh_base_color=LIGHT_BLUE,
    #         scene_bg_color=(1, 1, 1),
    #     )

    #     return 255 * regression_img[:, :, ::-1]
        
        


