import asyncio
import os.path
import shutil
import traceback
import torch
from loguru import logger
from preprocess import get_Wav2Lip_model, get_fa, get_pool_executor
import aiofiles
import uuid
from functools import partial

from wav2lip_288x288.inference import main


async def save_video_bytes_2shm_file(video_bytes: bytes):
    """
    保存视频二进制数据为内存中的文件
    """
    face = f"/dev/shm/{uuid.uuid4()}.mp4"
    async with aiofiles.open(face, mode='wb') as f:
        await f.write(video_bytes)
    return face


async def save_audio_bytes_2shm_file(audio_bytes: bytes):
    """
    保存音频二进制数据为内存中的文件
    """
    audio = f"/dev/shm/{uuid.uuid4()}.wav"
    async with aiofiles.open(audio, mode='wb') as f:
        await f.write(audio_bytes)
    return audio


async def inf_video(filename, audio_bytes, video_bytes, inf_video_tasks, vid):
    """
    filename: 文件名（不包括拓展名）
    """
    try:
        # 获得
        face, audio = await asyncio.gather(save_video_bytes_2shm_file(video_bytes),
                                           save_audio_bytes_2shm_file(audio_bytes))
        try:
            # 合成的视频结果所在文件夹
            res_video_dir = f"./result_videos/{vid}"
            if os.path.exists(res_video_dir):
                try:
                    shutil.rmtree(res_video_dir)
                except:
                    ...
            os.makedirs(res_video_dir, exist_ok=True)

            await asyncio.get_running_loop().run_in_executor(
                get_pool_executor(),
                partial(main, face, audio, get_Wav2Lip_model(),
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        outfile=os.path.join(res_video_dir, f"{filename}_result.mp4")))

            asyncio.create_task(delay_clear(300, vid, inf_video_tasks))
        finally:
            if os.path.exists(face):
                os.remove(face)
            if os.path.exists(audio):
                os.remove(audio)
    except:
        traceback.print_exc()


async def delay_clear(delay_sec: float, vid, inf_video_tasks):
    # 延迟清理视频
    sleep_task = asyncio.create_task(asyncio.sleep(delay_sec))
    try:
        await sleep_task
    finally:
        if vid in inf_video_tasks.keys(): inf_video_tasks.pop(vid)
        res_video_dir = f"./result_videos/{vid}"
        if os.path.exists(res_video_dir):
            try:
                shutil.rmtree(res_video_dir)
            except:
                ...
        logger.info(f"Video id:{vid} cleared.")
