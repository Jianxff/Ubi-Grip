import numpy as np
import cv2

def convert_bbox(hand_bbox: np.ndarray) -> np.ndarray:
    cx, cy = (hand_bbox[0] + hand_bbox[2]) // 2, (hand_bbox[1] + hand_bbox[3]) // 2
    hbox_w, hbox_h = hand_bbox[2] - hand_bbox[0], hand_bbox[3] - hand_bbox[1]
    sz = max(hbox_w, hbox_h) * 1.5
    return np.array([cx, cy, sz, sz]).astype(np.int32)

def image_sharpening(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.abs(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened