import os
import glob
from typing import Dict
import face_alignment
import torch

from objects import VideoFrames

video_full_frames: Dict[str, VideoFrames] = {}

fa = None


def get_fa():
    """获取FaceAlignment实例"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global fa
    if fa is None:
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_HALF_D, device=device, face_detector='blazeface')
    return fa


def preload_videos():
    """
    预加载视频文件，存为内存数据供推理用
    """
    # 指定视频文件的扩展名
    video_extensions = ["*.mp4", "*.avi", "*.mkv", "*.flv", "*.mov", "*.wmv"]
    # 指定需要搜索的文件夹
    folder = os.path.join(os.getcwd(), 'faces')
    # 用于存储找到的所有视频文件的路径
    videos = []
    for video_extension in video_extensions:
        # os.path.join用于合并路径
        # glob.glob返回所有匹配的文件路径列表
        videos.extend(glob.glob(os.path.join(folder, video_extension)))

    # 迭代找到的视频文件路径，转成video_frames
    for video in videos:
        vff = VideoFrames(video)
        vff.gen_frames()
        video_full_frames[os.path.splitext(os.path.basename(video))[0]] = vff
