import cv2
import io
from typing import List
import numpy as np
from loguru import logger


def extract_frames_from_video(video_bytes: bytes):
    videoCapture = cv2.VideoCapture(io.BytesIO(video_bytes))
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        # todo 转25fps
        logger.warning('The input video is not 25 fps, it would be better to trans it to 25 fps!')
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_ndarrays: List[np.ndarray] = []
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        frame_ndarrays.append(frame)
    frame_ndarrays: np.ndarray = np.stack(frame_ndarrays)
    return frame_ndarrays
