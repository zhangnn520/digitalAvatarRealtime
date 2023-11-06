import os.path
import cv2
from typing import List
import numpy as np
from loguru import logger
import uuid


def extract_frames_from_video_bytes(video_bytes: bytes, return_list: bool = False):
    """
    获取视频二进制数据的ndarray表达形式

    return_list: 是否以列表形式返回结果
    """
    temp_file_path = os.path.join("/dev/shm", str(uuid.uuid4()) + ".mp4")
    with open(temp_file_path, "wb") as f:
        f.write(video_bytes)

    videoCapture = cv2.VideoCapture(temp_file_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if int(fps) != 25:
        # todo 转25fps
        logger.warning('The input video is not 25 fps, it would be better to trans it to 25 fps!')
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_ndarrays: List[np.ndarray] = []
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        frame_ndarrays.append(frame)
    videoCapture.release()
    os.remove(temp_file_path)

    if not return_list:
        frame_ndarrays: np.ndarray = np.stack(frame_ndarrays)
    return frame_ndarrays
