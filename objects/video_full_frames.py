import os
from typing import List
import torch
from numpy import ndarray
import numpy as np
from torch import Tensor
from configuration import Settings
import cv2
import magic
import subprocess
import platform
import random
from DINet.data_processing import compute_crop_radius


class VideoFrames:
    """保存视频帧数据的数据结构"""

    def __init__(self, video_path: str):
        self.fps = 0
        self.video_path = video_path
        self.video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.full_frames: List[VideoFullFrame] = []
        self.ref_img_tensor: Tensor | None = None

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
        full_frames: ndarray = np.stack(full_frames)
        # 批图片人脸遮罩
        from preprocess import get_fa
        fa = get_fa()
        batch_landmarks = fa.get_landmarks_from_batch(torch.Tensor(full_frames.transpose(0, 3, 1, 2)))
        batch_landmarks = [landmarks[:68, :] for landmarks in batch_landmarks]
        assert all(landmarks.shape == (68, 2) for landmarks in batch_landmarks)
        # 封装进VideoFullFrame对象
        self.full_frames = [VideoFullFrame(full_frame, landmarks) for full_frame, landmarks in
                            zip(full_frames, batch_landmarks)]
        self.pick5ref_images()

    def pick5ref_images(self):
        '''selecting five reference images'''
        ref_img_list = []
        settings = Settings()
        resize_w = int(settings.mouth_region_size + settings.mouth_region_size // 4)
        resize_h = int((settings.mouth_region_size // 2) * 3 + settings.mouth_region_size // 8)
        ref_index_list = random.sample(range(5, len(self.full_frames)), 5)
        for ref_index in ref_index_list:
            crop_flag, crop_radius = compute_crop_radius(
                self.full_frames[0].full_frame.shape[:2][::-1],
                np.stack(
                    [video_full_frame.landmarks for video_full_frame in self.full_frames[ref_index - 5:ref_index]]),
            )
            if not crop_flag:
                raise ValueError('Our method can not handle videos with large change of facial size!!')
            crop_radius_1_4 = crop_radius // 4
            ref_img = self.full_frames[ref_index - 3].full_frame[:, :, ::-1]
            ref_landmark = self.full_frames[ref_index - 3].landmarks
            ref_img_crop = ref_img[
                           ref_landmark[29, 1] - crop_radius:
                           ref_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                           ref_landmark[33, 0] - crop_radius - crop_radius_1_4:
                           ref_landmark[33, 0] + crop_radius + crop_radius_1_4,
                           :]  # 裁剪鼻嘴部
            ref_img_crop = cv2.resize(ref_img_crop, (resize_w, resize_h))
            ref_img_crop = ref_img_crop / 255.0  # 颜色比例
            ref_img_list.append(ref_img_crop)
        ref_video_frame = np.concatenate(ref_img_list, 2)
        self.ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()  # 预加载


class VideoFullFrame:
    """保存视频单帧数据的数据结构"""

    def __init__(self, full_frame: ndarray, landmarks: ndarray):
        # 原帧
        self._full_frame: ndarray = full_frame
        # 人脸68特征点
        self._landmarks: ndarray = landmarks.astype(int)

    @property
    def full_frame(self):
        return self._full_frame.copy()

    @property
    def landmarks(self):
        return self._landmarks
