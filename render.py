# standard library
from pathlib import Path
from typing import List, Dict, Optional
import os
# third party
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
import time
import trimesh
# local
from src.dataset import BaseSeq, HO3D_Seq, H2O_Seq
from src.renderer import Renderer
from src.utils import image_sharpening


EVAL_OUTPUT_DIR = Path('output2')
SE3_OPENCV_TO_OPENGL = np.array([
    [1, -1, -1, 1],
    [-1, 1, 1, -1],
    [-1, 1, 1, -1],
    [1, 1, 1, 1]
]).astype(np.float32)

def main(
    seq_list: List[BaseSeq],
    output_root: str = './output',
    reset_on_fail: Optional[bool] = True,
    Terr_threshold: Optional[float] = 0.05,
    Rerr_threshold: Optional[float] = 5.0, # degree
    make_video: Optional[bool] = False,
) -> None:
    OUTPUT_DIR = Path(output_root)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGFILE = OUTPUT_DIR / 'result.csv'

    # sequence dataset
    for seq in seq_list:
        if LOGFILE.exists():
            with open(LOGFILE, "r") as f:
                lines = f.readlines()
            finished_seq = [line.split(',')[0] for line in lines]
            if seq.name in finished_seq:
                print(f"Sequence {seq.name} already evaluated, skipping...")
                del seq
                continue

        seq.preload_data()  # preload data if necessary

        H, W = seq.imsize

        # pipeline
        tracker = MegaPose(
            image_size=seq.imsize,
            intrinsic=seq.intrinsic,
            obj_path=seq.obj_model_path,
            mesh_unit='m'
        )

        init_pose = seq.get_object_pose(0)
        first_frame = True

        seq_evaluator = Evaluator(Terr_threshold, Rerr_threshold)

        if make_video:
            out_video = cv2.VideoWriter(
                f"{OUTPUT_DIR}/{seq.name}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                30, (W, H)
            )

            renderer = Renderer(width=W, height=H, intrinsic=seq.intrinsic)
            renderer.add(
                name='obj',
                mesh=trimesh.load_mesh(seq.obj_model_path),
                color=None
            )
    
        # ============================================================================================================
        """
        frame iteration
            i. hand reconstruction
            ii. hand mask rendering
            iii. object tracking with hand mask
            iv. object pose optimization using hand pose
            v. evaluation error calculation
        """

        pbar = tqdm(seq)


        for rgb, meta in pbar:
            TCO_GT = meta['obj_pose']

            rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
            rgb = image_sharpening(rgb, amount=2)

            TCO, conf = tracker.refine(rgb, TCO_GT)

            if make_video:
                TCO_GL = TCO * SE3_OPENCV_TO_OPENGL
                renderer.set('obj', pose=TCO_GL)
                frame = renderer.render_image(image_rgb=rgb)
                out_video.write(frame)

            Terr, Rerr = seq_evaluator.evaluate_loss(TCO_GT, TCO)
            pbar.set_description(f"[{seq.name}] Conf: {conf * 100:.1f} Err: {Terr*100:.1f}cm/{Rerr:.1f}deg")
            seq_evaluator.update(Terr, Rerr)

            if reset_on_fail and (Terr > Terr_threshold or Rerr > Rerr_threshold):
                init_pose = TCO_GT
            else:
                init_pose = TCO

        if make_video:
            out_video.release()

        with open(OUTPUT_DIR / f"result.csv", "a+") as f:
            f.write(f"{seq.name},{seq.obj_name},{seq_evaluator.avg_recall()}\n")

        del tracker
        del seq


def eval_h2o(root_dir: str, subject: str):
    os.makedirs(EVAL_OUTPUT_DIR / 'h2o/obj_meta', exist_ok=True)

    root_dir = Path(root_dir)
    model_dir = root_dir / 'object'
    subject_dir = root_dir / subject

    seq_list = []

    for seq_group in subject_dir.iterdir():
        if not seq_group.is_dir(): 
            continue
        # [h1, h2, k1, k2, o1, o2]
        for seq in seq_group.iterdir():    
            if not seq.is_dir():
                continue 
            # [1, 2 , 3 ...]
            seq_list.append(
                H2O_Seq(
                    root=seq,
                    name=f"{subject}_{seq_group.stem}_{seq.stem}",
                    obj_model_dir=model_dir,
                    obj_meta_dir=EVAL_OUTPUT_DIR / 'h2o/obj_meta',
                )
            )
    
    main(
        seq_list=seq_list,
        output_root=EVAL_OUTPUT_DIR / 'h2o' / subject,
        reset_on_fail=True,
        Terr_threshold=0.05,
        Rerr_threshold=10.0,  # degree
        make_video=False
    )


if __name__ == '__main__':
    # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5

    eval_h2o(root_dir='/data/datasets/H2O', subject='subject3_ego')
        