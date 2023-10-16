import cv2
from numpy import ndarray


def ndarray2frame(full_frame: ndarray):
    """ndarray转frame"""
    # 将帧转换为JPEG格式。
    ret, buffer = cv2.imencode(".jpg", full_frame)
    if ret:
        return buffer
