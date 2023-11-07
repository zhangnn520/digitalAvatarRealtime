import asyncio
import traceback
import uuid
from typing import List, Dict
from configuration.development_config import Settings
import shutil
import numpy as np
from numpy import ndarray
import cv2
import random
from loguru import logger
from tqdm import tqdm
import tempfile
from moviepy.editor import *
import torch
from utilities.audio_bytes2np_array import voice_bytes2array
from wav2lip_288x288 import audio
from wav2lip_288x288.inference import datagen
from preprocess import get_fa, get_Wav2Lip_model
from routers.global_process_pool_executor import get_pool_executor
from utilities.extract_frames_from_video import extract_frames_from_video_bytes
from collections import namedtuple


def _get_frames_landmarks_pad(frames_ndarray: ndarray, video_landmark_data: ndarray, res_frame_length: int):
    """
    align frame with driving audio
    首位加点pad
    """
    video_frames_data_cycle = np.concatenate([frames_ndarray, np.flip(frames_ndarray, 0)], 0)
    video_landmark_data_cycle = np.concatenate([video_landmark_data, np.flip(video_landmark_data, 0)], 0)
    video_frames_data_cycle_length = len(video_frames_data_cycle)
    if video_frames_data_cycle_length >= res_frame_length:
        res_video_frames_data = video_frames_data_cycle[:res_frame_length, :, :, :]
        res_video_landmark_data = video_landmark_data_cycle[:res_frame_length, :, :]
    else:
        divisor = res_frame_length // video_frames_data_cycle_length
        remainder = res_frame_length % video_frames_data_cycle_length
        res_video_frames_data = np.concatenate(
            [video_frames_data_cycle] * divisor + [video_frames_data_cycle[:remainder]], 0)
        res_video_landmark_data = np.concatenate(
            [video_landmark_data_cycle] * divisor + [video_landmark_data_cycle[:remainder, :, :]], 0)
    res_video_frames_data_pad: ndarray = np.pad(res_video_frames_data, ((2, 2), (0, 0), (0, 0), (0, 0)), mode='edge')
    res_video_landmark_data_pad = np.pad(res_video_landmark_data, ((2, 2), (0, 0), (0, 0)), mode='edge')
    return res_video_frames_data_pad, res_video_landmark_data_pad


def _pick5frames(res_video_frames_data_pad: ndarray,
                 res_video_landmark_data_pad: ndarray,
                 resize_w: int,
                 resize_h: int):
    ref_index_list = random.sample(range(5, res_video_frames_data_pad.shape[0] - 2), 5)
    ref_img_list = []
    video_size = res_video_frames_data_pad.shape[1:3][::-1]
    for ref_index in ref_index_list:
        crop_flag, crop_radius = compute_crop_radius(video_size,
                                                     res_video_landmark_data_pad[ref_index - 5:ref_index, :, :])
        if not crop_flag:
            raise ValueError('our method can not handle videos with large change of facial size!!')
        crop_radius_1_4 = crop_radius // 4
        ref_img = res_video_frames_data_pad[ref_index - 3, :, :, ::-1]
        ref_landmark = res_video_landmark_data_pad[ref_index - 3, :, :]
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
    return ref_video_frame, video_size


def inf2frames(ref_video_frame,
               video_size,
               pad_length,
               res_video_landmark_data_pad,
               res_video_frames_data_pad,
               resize_w,
               resize_h,
               mouth_region_size,
               ds_feature_padding,
               model) -> List[ndarray]:
    ref_img_tensor = torch.from_numpy(ref_video_frame).permute(2, 0, 1).unsqueeze(0).float().cuda()
    frames = []
    for clip_end_index in tqdm(range(5, pad_length, 1)):
        crop_flag, crop_radius = compute_crop_radius(
            video_size,
            res_video_landmark_data_pad[clip_end_index - 5:clip_end_index, :, :],
            random_scale=1.05)  # 5个图片一包，窗口移动处理
        if not crop_flag:
            raise ValueError('our method can not handle videos with large change of facial size!!')
        crop_radius_1_4 = crop_radius // 4
        frame_data = res_video_frames_data_pad[clip_end_index - 3, :, :, ::-1]  # 包里面5个图片的中间那个
        frame_landmark = res_video_landmark_data_pad[clip_end_index - 3, :, :]
        crop_frame_data = frame_data[
                          frame_landmark[29, 1] - crop_radius:frame_landmark[29, 1] + crop_radius * 2 + crop_radius_1_4,
                          frame_landmark[33, 0] - crop_radius - crop_radius_1_4:frame_landmark[
                                                                                    33, 0] + crop_radius + crop_radius_1_4,
                          :]  # 裁剪面部
        crop_frame_h, crop_frame_w = crop_frame_data.shape[0], crop_frame_data.shape[1]
        crop_frame_data = cv2.resize(crop_frame_data, (resize_w, resize_h))  # [32:224, 32:224, :]
        crop_frame_data = crop_frame_data / 255.0
        # todo 平均亮度校正
        crop_frame_data[mouth_region_size // 2:mouth_region_size // 2 + mouth_region_size,
        mouth_region_size // 8:mouth_region_size // 8 + mouth_region_size, :] = 0
        crop_frame_tensor = torch.from_numpy(crop_frame_data).float().cuda().permute(2, 0, 1).unsqueeze(0)
        deepspeech_tensor = torch.from_numpy(
            ds_feature_padding[clip_end_index - 5:clip_end_index, :]).permute(1, 0).unsqueeze(0).float().cuda()
        with torch.no_grad():
            pre_frame = model(crop_frame_tensor, ref_img_tensor, deepspeech_tensor)
            pre_frame = pre_frame.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255  # 面部
        pre_frame_resize = cv2.resize(pre_frame, (crop_frame_w, crop_frame_h))  # 恢复原面部高宽
        frame_data[
        frame_landmark[29, 1] - crop_radius:
        frame_landmark[29, 1] + crop_radius * 2,
        frame_landmark[33, 0] - crop_radius - crop_radius_1_4:
        frame_landmark[33, 0] + crop_radius + crop_radius_1_4,
        :] = pre_frame_resize[:crop_radius * 3, :, :]  # 将推理的面部写回原帧
        frames.append(frame_data)
    return frames


def audio_bytes2mel_chunks(audio_bytes: bytes, fps: float):
    """
    由wav音频二进制数据生成音频mel频谱块
    """
    # 音频文件
    mel = audio.melspectrogram(voice_bytes2array(audio_bytes))

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    mel_step_size = 16
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1
    return mel_chunks


def full_frames_mel_chunks2video_file(model, full_frames, mel_chunks, audio_bytes):
    """由full_frames和mel_chunks推理视频"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = Settings().wav2lip_batch_size
    gen = datagen(full_frames, mel_chunks)
    # tmp_result_path = f"/dev/shm/{uuid.uuid4()}.avi"
    # 推理出的脸部图片和待回帖的图片以及位置
    stickMaterial = namedtuple("stickMaterial", ["org_img", "coordinate", "pred_img"])
    pred_materials: List[stickMaterial] = []
    for i, (img_batch, mel_batch, frames, coords) in enumerate(
            tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            # out = cv2.VideoWriter(tmp_result_path,
            #                       cv2.VideoWriter_fourcc(*'DIVX'),
            #                       Settings().fps,
            #                       (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            pred_materials.append(stickMaterial(org_img=f.copy(), coordinate=c, pred_img=p))
            # f[y1:y2, x1:x2] = p

    #         out.write(f)
    #
    # out.release()
    #
    # audio_file_path = f"/dev/shm/{uuid.uuid4()}.wav"
    # with open(audio_file_path, "wb") as f:
    #     f.write(audio_bytes)
    #
    # command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_file_path, tmp_result_path, args.outfile)
    # subprocess.call(command, shell=platform.system() != 'Windows')


async def inf_video(filename, audio_bytes, video_bytes, inf_video_tasks, vid):
    # 转音频数据为mel频谱
    audio_bytes2mel_chunks_fut = asyncio.get_running_loop().run_in_executor(get_pool_executor(),
                                                                            audio_bytes2mel_chunks,
                                                                            audio_bytes,
                                                                            Settings().fps)
    full_frames: ndarray = await asyncio.get_running_loop().run_in_executor(
        get_pool_executor(), extract_frames_from_video_bytes, video_bytes)  # 视频处理 得到视频的帧ndarray表示
    mel_chunks = await audio_bytes2mel_chunks_fut
    full_frames = full_frames[:len(mel_chunks)]
    asyncio.get_running_loop().run_in_executor(get_pool_executor(),
                                               full_frames_mel_chunks2video_file,
                                               get_Wav2Lip_model(),
                                               full_frames,
                                               mel_chunks,
                                               audio_bytes)


def face_join2video_file(pred_frames, pred_batch_landmarks, org_frames_ndarr, audio_bytes, res_video_path):
    joined_frames = []
    for p, points_68, frame_ndarray in zip(pred_frames, pred_batch_landmarks, org_frames_ndarr):
        if points_68.shape[0] >= 17:
            face_points = points_68[:17]
            if points_68.shape[0] >= 25:
                face_points = np.append(face_points, [points_68[24], points_68[19]], axis=0)
            face_points = np.stack(face_points).astype(np.int32)
            # 1. 创建一个长方形遮罩
            mask = np.zeros(p.shape[:2], dtype=np.uint8)
            # 2. 使用fillPoly绘制人脸遮罩
            cv2.fillPoly(mask, [face_points], (255, 255, 255))
            # 反向遮罩
            reverse_mask = cv2.bitwise_not(mask)
            # 3. 使用遮罩提取人脸
            face_image = cv2.bitwise_and(p, p, mask=mask)
            # 提取人脸周围
            face_surrounding = cv2.bitwise_and(frame_ndarray, frame_ndarray, mask=reverse_mask)
            # 推理出的人脸贴回原帧
            joined_frame = cv2.add(face_image, face_surrounding[:, :, ::-1])
            joined_frames.append(joined_frame)

    # 添加声音
    # 创建一个 VideoClip 对象
    video_clip = ImageSequenceClip(joined_frames, fps=25)
    # 将音频数据保存到临时文件
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_audio_file.write(audio_bytes)
    temp_audio_file.close()
    # 创建一个 AudioFileClip 对象
    audio_clip = AudioFileClip(temp_audio_file.name)
    # 将音频添加到视频
    final_clip = video_clip.set_audio(audio_clip)
    # 保存最终的视频文件
    final_clip.write_videofile(res_video_path, codec="libx264", audio_codec="aac")
    # 删除临时音频文件
    os.unlink(temp_audio_file.name)


async def delay_clear(delay_sec: float, vid, inf_video_tasks):
    # 延迟清理视频
    sleep_task = asyncio.create_task(asyncio.sleep(delay_sec))
    try:
        await sleep_task
    finally:
        if vid in inf_video_tasks.keys(): inf_video_tasks.pop(vid)
        res_video_dir = f"/dev/shm/{vid}"
        if os.path.exists(res_video_dir):
            try:
                shutil.rmtree(res_video_dir)
            except:
                ...
        logger.info(f"Video id:{vid} cleared.")


def inf_video_from_ndarray2frames(frames_ndarray,
                                  DINet_model,
                                  ds_feature,
                                  batch_landmarks):
    try:
        res_frame_length = ds_feature.shape[0]
        ds_feature_padding = np.pad(ds_feature, ((2, 2), (0, 0)), mode='edge')
        # 人脸68关键点
        batch_landmarks = [landmarks[:68, :] for landmarks in batch_landmarks]
        video_landmark_data: np.ndarray = np.stack(batch_landmarks).astype(int)
        ############################################## align frame with driving audio ##############################################从视频无限回环中截取以对齐音频
        res_video_frames_data_pad, res_video_landmark_data_pad = _get_frames_landmarks_pad(frames_ndarray,
                                                                                           video_landmark_data,
                                                                                           res_frame_length)
        assert ds_feature_padding.shape[0] == res_video_frames_data_pad.shape[0] == res_video_landmark_data_pad.shape[0]
        pad_length = ds_feature_padding.shape[0]
        ############################################## randomly select 5 reference images ##############################################
        mouth_region_size = Settings().mouth_region_size
        resize_w = int(mouth_region_size + mouth_region_size // 4)
        resize_h = int((mouth_region_size // 2) * 3 + mouth_region_size // 8)
        ref_video_frame, video_size = _pick5frames(res_video_frames_data_pad, res_video_landmark_data_pad, resize_w,
                                                   resize_h)
        ############################################## inference frame by frame ##############################################
        return res_video_frames_data_pad.copy(), inf2frames(ref_video_frame, video_size, pad_length,
                                                            res_video_landmark_data_pad,
                                                            res_video_frames_data_pad, resize_w, resize_h,
                                                            mouth_region_size,
                                                            ds_feature_padding,
                                                            DINet_model)
    except:
        traceback.print_exc()
