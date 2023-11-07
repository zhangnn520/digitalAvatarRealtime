import os
import glob
from typing import Dict
import face_alignment
import torch
from DINet.models.DINet import DINet
from objects import VideoFrames
from collections import OrderedDict
from DINet.utils.deep_speech import DeepSpeech
from loguru import logger
from wav2lip_288x288.inference import load_model as _load_wav2lip_model
from concurrent.futures import ProcessPoolExecutor
from configuration.development_config import Settings

video_full_frames: Dict[str, VideoFrames] = {}
# 人脸检测
_fa = None
# 推理模型
_DINet_model = None
# deepspeech 模型
_DSModel = None
# Wav2Lip推理模型
_Wav2Lip_model = None
# 进程池执行器
_pool_executor: ProcessPoolExecutor = None


def get_DINet_model():
    """获取DINet推理模型"""
    global _DINet_model
    if _DINet_model is None:
        logger.info(f"load DINet model")
        _DINet_model = DINet(3, 15, 29).cuda()
        pretrained_clip_DINet_path = "./DINet/asserts/clip_training_DINet_256mouth.pth"
        if not os.path.exists(pretrained_clip_DINet_path):
            raise FileNotFoundError(
                'wrong path of pretrained model weight: {}。Reference "https://github.com/monk-after-90s/DINet" to download.'.format(
                    pretrained_clip_DINet_path))
        state_dict = torch.load(pretrained_clip_DINet_path)['state_dict']['net_g']
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module.
            new_state_dict[name] = v
        _DINet_model.load_state_dict(new_state_dict)
        _DINet_model.eval()
    return _DINet_model


def get_Wav2Lip_model():
    """获取Wav2Lip288推理模型"""
    global _Wav2Lip_model
    if _Wav2Lip_model is None:
        logger.info("load Wav2Lip 288×288 model...")
        checkpoint_path = "./wav2lip_288x288/checkpoints/checkpoint_prod.pth"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"File {checkpoint_path} doesn't exist. Refer to https://github.com/monk-after-90s/wav2lip_288x288.git.")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _Wav2Lip_model = _load_wav2lip_model(checkpoint_path, device)
    return _Wav2Lip_model


def get_DSModel():
    """获取deepspeech 模型"""
    global _DSModel
    if _DSModel is None:
        logger.info(f"load deepspeech model")
        deepspeech_model_path = "./DINet/asserts/output_graph.pb"
        if not os.path.exists(deepspeech_model_path):
            raise FileNotFoundError(
                'pls download pretrained model of deepspeech.Refer to "https://github.com/monk-after-90s/DINet" to download.')
        _DSModel = DeepSpeech(deepspeech_model_path)
    return _DSModel


def get_fa():
    """获取FaceAlignment实例"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    global _fa
    if _fa is None:
        logger.info(f"load face_alignment model")
        _fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_HALF_D, device=device, face_detector='blazeface')
    return _fa


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


def load_model():
    """加载模型到GPU"""
    # DINet预训练模型
    get_DINet_model()
    # deepspeech模型
    get_DSModel()
    # face-alignment
    get_fa()
    # Wav2Lip288预训练模型
    get_Wav2Lip_model()


def get_pool_executor():
    global _pool_executor
    if _pool_executor is None:
        logger.info(f"instantiate ProcessPoolExecutor")
        _pool_executor = ProcessPoolExecutor(max_workers=Settings().max_workers)
    return _pool_executor


def ensure_pool_executor_closed():
    """关闭进程池执行器"""
    if _pool_executor is not None:
        _pool_executor.shutdown()
