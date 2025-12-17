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
# local
from src.pipeline import HandFilter, ObjectTracker, HaMeR, HandDetector
from src.dataset import BaseSeq, HO3D_Seq, H2O_Seq
from src.evaluate import Evaluator
from src.utils import image_sharpening

EVAL_OUTPUT_DIR = Path('output')

def main(
    seq_list: List[BaseSeq],
    output_root: str,
    use_hand_filter: Optional[bool] = False,
    use_hand_rotation: Optional[bool] = False,
    reset_on_fail: Optional[bool] = True,
    Terr_threshold: Optional[float] = 0.05,
    Rerr_threshold: Optional[float] = 5.0, # degree
    make_video: Optional[bool] = False,
    save_rawdata: Optional[bool] = True,
) -> None:
    OUTPUT_DIR = Path(output_root)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGFILE = OUTPUT_DIR / 'result.csv'

    OPT = ''

    if use_hand_filter:
        handdetector = HandDetector(gpu_delegate=False)
        hamer = HaMeR(ckpt_path='data/ckpt/hamer.ckpt', mano_dir='data/mano', device='cuda')
        OPT = 'F' + ('R' if use_hand_rotation else '')

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

        """init object tracking"""
        objtracker = ObjectTracker(
            name='object',
            imwidth=W, 
            imheight=H, 
            intrinsic=seq.intrinsic,
            model_path=seq.obj_model_path,
            meta_path=seq.obj_meta_path,
            model_unit_in_meter=seq.obj_unit_in_meter,
            model_from_opengl=seq.obj_from_opengl,
            use_renderer=make_video
        )

        init_pose = seq.get_object_pose(0)
        objtracker.reset(init_pose)
        first_frame = True

        """
        renderer initialization for hand mask rendering and object visualization
        """
        if use_hand_filter:
            handfilter = HandFilter(hamer=hamer, imwidth=W, imheight=H)
        
        if make_video:
            out_video = cv2.VideoWriter(
                f"{OUTPUT_DIR}/{seq.name}.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                30, (W, H)
            )
        
        seq_evaluator = Evaluator(Terr_threshold, Rerr_threshold)

        # ============================================================================================================
        """
        frame iteration
            i. hand reconstruction
            ii. hand mask rendering
            iii. object tracking with hand mask
            iv. object pose optimization using hand pose
            v. evaluation error calculation
        """
        opt_step = 0
        total_opt_step = 0

        pbar = tqdm(seq)
        times_det = []
        times_hamer = []
        times_render = []
        times_obj = []

        for rgb, meta in pbar:
            obj_pose_GT = meta['obj_pose']
            hand_bbox = meta['hand_bbox']
            # image preprocessing
            rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
            rgb = image_sharpening(rgb, amount=2)

            # DEBUG
            st = time.time()
            # rgb_half = rgb.resize((H//2, W//2)
            rgb_half = np.resize(rgb, (H//2, W//2))
            detect_result = handdetector.predict(image_rgb=rgb_half)
            hand_det = handdetector.get_detect_bbox(detect_result, image_HW=(H//2, W//2), force_side='Left')
            # time_det = (time.time() - st) * 1000
            # times_det.append(time_det)
            # print(f"hand detection time: {time_det:.2f} ms")

            hand_mask = None
            # hand pose estimation
            if use_hand_filter:
                if hand_bbox is not None:
                    time_pred, time_render = handfilter.update(rgb.copy(), (hand_bbox, True))
                    times_hamer.append(time_pred)
                    times_render.append(time_render)
                    hand_mask = handfilter.hand_mask
                else:
                    handfilter.reset()
            
            if not first_frame:
                st = time.time()
                objtracker.update(image=rgb, mask=hand_mask)
                time_obj = (time.time() - st) * 1000
                times_obj.append(time_obj)


            # optimization using hand pose
            if opt_step > 10 or objtracker.valid_percentage >= 0.5 or first_frame: 
                opt_step = 1
            elif objtracker.valid_percentage < 0.5 and use_hand_filter and use_hand_rotation:
                last_rotation = objtracker.last_pose[:3, :3].copy()
                last_trans = objtracker.last_pose[:3, 3].copy()
                cur_trans = objtracker.pose[:3, 3].copy()

                opt_rotation = handfilter.delta_rotation @ last_rotation
                pred_pose = np.eye(4)
                pred_pose[:3, :3] = opt_rotation
                pred_pose[:3, 3] = (cur_trans + last_trans) / 2
                objtracker.reset(pred_pose)
                opt_step += 1
                total_opt_step += 1
            
            first_frame = False

            """
            evaluation error calculation
                i. translation error
                ii. rotation error
            """
            if obj_pose_GT is not None:
                Terr, Rerr = seq_evaluator.evaluate_loss(obj_pose_GT, objtracker.pose)
                seq_evaluator.update(Terr, Rerr)

                if reset_on_fail and (Terr > Terr_threshold or Rerr > Rerr_threshold):
                    objtracker.reset(obj_pose_GT)
            
            pbar.set_description(f'seq: {seq.name}, {seq_evaluator.tostr()}')

            if make_video:
                track_im = objtracker.render_image(image_rgb=rgb)[:, :, ::-1]
                # draw hand edge
                if use_hand_filter:
                    hand_edge = cv2.Canny(handfilter.hand_mask, 0, 255)
                    track_im[hand_edge > 0] = [0, 255, 0]
                # write video
                out_video.write(track_im)
                # cv2.imshow("temp", track_im)
                # cv2.waitKey(1)
                # cv2.imwrite("temp.png", track_im)
            
            # # time statistics
            # if len(times_det) > 0:
            #     print(f"hand detection: {np.mean(times_det):.2f} ms, "
            #             f"hand prediction: {np.mean(times_hamer):.2f} ms, "
            #             f"hand mask rendering: {np.mean(times_render):.2f} ms, "
            #             f"object tracking: {np.mean(times_obj):.2f} ms")

        # ============================================================================================================

        if make_video:
            out_video.release()

        """
        average sequence evaluation result
        """
        with open(OUTPUT_DIR / f"result.csv", "a+") as f:
            f.write(f"{seq.name},{seq.obj_name},{OPT},{seq_evaluator.avg_recall()},{total_opt_step},{total_opt_step / len(seq_evaluator.data)}\n")

        if save_rawdata:
            seq_evaluator.save_raw(path=OUTPUT_DIR / f'{seq.name}_loss.npy')

        del objtracker
        if use_hand_filter:
            del handfilter
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
        use_hand_filter=True,
        use_hand_rotation=False,
        reset_on_fail=True,
        Terr_threshold=0.05,
        Rerr_threshold=10.0,  # degree
        make_video=False,
        save_rawdata=False,
    )


def eval_ho3d(root_dir: str, model_dir: str, seq_type: str = 'evaluation'):
    os.makedirs(EVAL_OUTPUT_DIR / 'ho3d/obj_meta', exist_ok=True)

    root_dir = Path(root_dir)
    seq_list = []
    for d in (root_dir / seq_type).iterdir():
        if not d.is_dir():
            continue
        seq_list.append(
            HO3D_Seq(
                name=f"{seq_type}_{d.stem}",
                root=str(d),
                obj_model_dir=model_dir,
                obj_meta_dir=EVAL_OUTPUT_DIR / 'ho3d/obj_meta',
                seq_type=seq_type
            )
        )

    main(
        seq_list=seq_list,
        output_root=EVAL_OUTPUT_DIR / 'h2o',
        use_hand_filter=True,
        use_hand_rotation=False,
        reset_on_fail=True,
        Terr_threshold=0.05,
        Rerr_threshold=10.0,  # degree
        make_video=False,
    )


if __name__ == '__main__':
    # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtiff.so.5

    eval_h2o(root_dir='/data/datasets/H2O', subject='subject3_ego')
    # eval_ho3d(root_dir='/data/datasets/HO3D_v3', model_dir='/data/datasets/HO3D_v3/models', seq_type='evalutaion')

    