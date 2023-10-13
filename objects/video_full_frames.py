import os
from typing import List
import face_alignment
import torch
from numpy import ndarray
import numpy as np
from config import Settings
import cv2
import magic
import subprocess
import platform


class VideoFrames:
    """保存视频帧数据的数据结构"""

    def __init__(self, video_path: str):
        self.fps = 0
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.full_frames: List = []

    def gen_frames(self):
        """将视频文件转为帧数据和人脸关键点"""
        # 文件名，数字人的名字
        file_name_without_ext = os.path.splitext(os.path.basename(self.video_path))[0]

        settings = Settings()
        video_stream = cv2.VideoCapture(self.video_path)
        self.fps = video_stream.get(cv2.CAP_PROP_FPS)
        if self.fps != float(settings.fps) or 'video/mp4' not in magic.Magic(mime=True).from_file(
                self.video_path):  # 修改帧率 和 类型
            video_stream.release()
            tmp_file = os.path.join(os.getcwd(), "temp", file_name_without_ext + f"_{settings.fps}fps.mp4")
            # 转帧率和类型
            command = f'ffmpeg -i {self.video_path} -r {settings.fps} {tmp_file} -y'
            subprocess.call(command, shell=platform.system() != 'Windows')
            # 读取到内存
            try:
                video_stream = cv2.VideoCapture(tmp_file)
            finally:
                os.remove(tmp_file)
            self.fps = video_stream.get(cv2.CAP_PROP_FPS)
            assert self.fps == settings.fps
        # 以上，视频的类型和帧率得到确保
        # 获取逐帧的ndarray表示
        full_frames: List[ndarray] = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            full_frames.append(frame)
        # 人脸68关键点
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_HALF_D, device=device, face_detector='blazeface')
        full_frames: ndarray = np.stack(full_frames)
        # 批图片人脸遮罩
        batch_landmarks = fa.get_landmarks_from_batch(torch.Tensor(full_frames.transpose(0, 3, 1, 2)))
        # 封装进VideoFullFrame对象
        self.full_frames = [VideoFullFrame(full_frame, landmarks) for full_frame, landmarks in
                            zip(full_frames, batch_landmarks)]


class VideoFullFrame:
    """保存视频单帧数据的数据结构"""

    def __init__(self, full_frame: ndarray, landmarks: ndarray):
        # 原帧
        self._full_frame: ndarray = full_frame
        # 人脸68特征点
        self._landmarks: ndarray = landmarks

    @property
    def full_frame(self):
        return self._full_frame.copy()

    @property
    def landmarks(self):
        return self._landmarks
