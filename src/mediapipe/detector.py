# standard library
from typing import *
from pathlib import Path
# third party
import numpy as np
from PIL import Image
import time
# mediapipe
import mediapipe as mp


class HandDetector:
    def __init__(
        self,
        gpu_delegate: bool = False,
    ) -> None:
        conf_arg = {
            'model_asset_path': Path(__file__).absolute().parent / 'hand_landmarker.task'
        }
        if gpu_delegate:
            conf_arg['delegate'] = mp.tasks.BaseOptions.Delegate.GPU
        # make base options
        base_option = mp.tasks.BaseOptions(**conf_arg)
        # init landmarker
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=base_option,
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
            )
        )


    def predict(
        self,
        image_rgb: Union[np.ndarray, str, Path],
        timestamp_ms: Optional[int] = None
    ) -> NamedTuple:
        if isinstance(image_rgb, (str, Path)):
            image_rgb = np.array(Image.open(image_rgb))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        # reset timestamp
        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # detect hand landmarks
        results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        return results
    

    @staticmethod
    def get_detect_bbox(
        results: NamedTuple,
        image_HW: Tuple[int, int],
        extend_scale: float = 1,
        force_side: Optional[str] = None
    ) -> List[Tuple[Tuple[int], bool]]:
        bbox = []
        right = []
        if results.hand_landmarks:
            for hand in results.handedness:
                is_right = hand[0].category_name == 'Right'
                if force_side is not None:
                    is_right = (force_side == 'Right')
                right.append(is_right)

            image_hight, image_width = image_HW

            for hand_landmarks in results.hand_landmarks:
                # collect all x and y
                lm_x = [lm.x for lm in hand_landmarks]
                lm_y = [lm.y for lm in hand_landmarks]
                # calculate center point
                x = np.mean(lm_x) * image_width
                y = np.mean(lm_y) * image_hight
                # calculate width and height
                x_min, x_max = min(lm_x), max(lm_x)
                y_min, y_max = min(lm_y), max(lm_y)
                w = (x_max - x_min) * image_width * extend_scale
                h = (y_max - y_min) * image_hight * extend_scale
                sz = max(w, h)
                # append to bbox
                bbox.append(np.array([x, y, sz, sz]).astype(np.int32))

        # return list of (bbox, is_right)
        return list(zip(bbox, right)) 