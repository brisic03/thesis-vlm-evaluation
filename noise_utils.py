import cv2
import numpy as np
import random
from PIL import Image
from io import BytesIO

def apply_gaussian_blur(frame: Image.Image, kernel_size: int) -> Image.Image:
    """Apply Gaussian blur. kernel_size must be odd (3, 5, or 7)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    arr = np.array(frame)
    blurred = cv2.GaussianBlur(arr, (kernel_size, kernel_size), 0)
    return Image.fromarray(blurred)

def apply_jpeg_compression(frame: Image.Image, quality: int) -> Image.Image:
    """Simulate JPEG compression artifacts. quality between 20-60."""
    buffer = BytesIO()
    frame.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return Image.open(buffer).copy()

def apply_random_occlusion(frame: Image.Image, occlusion_ratio: float) -> Image.Image:
    """Black out a random rectangular region. occlusion_ratio between 0.1-0.3."""
    arr = np.array(frame).copy()
    h, w = arr.shape[:2]
    region_area = int(h * w * occlusion_ratio)
    region_h = int(h * (occlusion_ratio ** 0.5))
    region_w = region_area // region_h
    x = random.randint(0, max(0, w - region_w))
    y = random.randint(0, max(0, h - region_h))
    arr[y:y+region_h, x:x+region_w] = 0
    return Image.fromarray(arr)

def drop_frames(frames: list, drop_ratio: float) -> list:
    """Randomly drop a proportion of frames. drop_ratio between 0.2-0.4."""
    n_keep = max(1, int(len(frames) * (1 - drop_ratio)))
    kept = sorted(random.sample(range(len(frames)), n_keep))
    return [frames[i] for i in kept]

def vary_sampling_rate(v_path: str, n_frames: int, rate_multiplier: float) -> list:
    """
    Sample frames at a different rate by adjusting step size.
    rate_multiplier > 1 = sparser sampling, < 1 = denser sampling.
    """
    import cv2
    vid = cv2.VideoCapture(v_path)
    total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        return []
    step = int((total // n_frames) * rate_multiplier)
    step = max(1, step)
    frames = []
    for i in range(n_frames):
        pos = min(i * step, total - 1)
        vid.set(cv2.CAP_PROP_POS_FRAMES, pos)
        success, frame = vid.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    vid.release()
    return frames

GAUSSIAN_BLUR_LEVELS   = [3, 5, 7]
JPEG_QUALITY_LEVELS    = [60, 40, 20]
OCCLUSION_LEVELS       = [0.10, 0.20, 0.30]
FRAME_DROP_LEVELS      = [0.20, 0.30, 0.40]
SAMPLING_RATE_LEVELS   = [1.5, 2.0, 3.0]

def apply_visual_noise(frames: list, noise_type: str, level) -> list:
    """
    Apply visual noise to a list of frames.
    noise_type: 'blur', 'jpeg', 'occlusion'
    """
    if noise_type == 'blur':
        return [apply_gaussian_blur(f, level) for f in frames]
    elif noise_type == 'jpeg':
        return [apply_jpeg_compression(f, level) for f in frames]
    elif noise_type == 'occlusion':
        return [apply_random_occlusion(f, level) for f in frames]
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
