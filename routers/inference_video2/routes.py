from fastapi import APIRouter
from fastapi import UploadFile, File
import asyncio
from asyncio import Task
import uuid
from typing import Dict
import os
from routers.helper2 import inf_video
from utilities.is_bytes_wav import is_wav
from fastapi import HTTPException
import aiofiles

router = APIRouter()

inf_video_tasks: Dict[str, Task] = {}


@router.post("/uploadAv", summary="接收音频和视频", tags=["音视频"])
async def uploadAv(audio: UploadFile = File(...), video: UploadFile = File(...)):
    """
    推理视频
    """
    audio_bytes, video_bytes = await asyncio.gather(audio.read(), video.read())  # 读取音视频数据
    if not is_wav(audio_bytes):
        # 转为wav格式
        tmp_file_name = str(uuid.uuid4())
        tmp_audio_path = os.path.join("/dev/shm", tmp_file_name)
        async with aiofiles.open(tmp_audio_path, mode='wb') as f:
            await f.write(audio_bytes)

        cmd = f"ffmpeg -i {tmp_audio_path} {tmp_audio_path}.wav"
        proc = await asyncio.create_subprocess_shell(cmd,
                                                     stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        if stderr is not None:
            raise HTTPException(status_code=500, detail=f"执行cmd:'{cmd}'报错。\nstdout={stdout}，\nstderr={stderr}")
        # 一切正常
        os.remove(tmp_audio_path)
        async with aiofiles.open(f"{tmp_audio_path}.wav", mode='r') as f:
            audio_bytes = await f.read()
        os.remove(f"{tmp_audio_path}.wav")
    filename, extension = os.path.splitext(video.filename)
    if extension != ".mp4":
        raise HTTPException(status_code=415, detail=f"需要视频格式为mp4")
    vid = str(uuid.uuid4())  # 视频id
    inf_video_tasks[vid] = asyncio.create_task(inf_video(filename, audio_bytes, video_bytes, inf_video_tasks, vid))
    return {"videoId": vid}
