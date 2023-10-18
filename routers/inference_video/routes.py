from fastapi import APIRouter
from fastapi import UploadFile, File
import asyncio
from asyncio import Task
import uuid
from typing import Dict
from routers.helpers import inf_video, delay_rm_video
import os

router = APIRouter()

inf_video_tasks: Dict[str, Task] = {}


@router.post("/uploadAv", summary="接收音频和视频", tags=["音视频"])
async def uploadAv(audio: UploadFile = File(...), video: UploadFile = File(...)):
    """
    推理视频
    """
    audio_bytes, video_bytes = await asyncio.gather(audio.read(), video.read())  # 读取音视频数据
    filename, extension = os.path.splitext(video.filename)
    vid = str(uuid.uuid4())  # 视频id
    inf_video_tasks[vid] = asyncio.create_task(inf_video(vid, filename, video_bytes, audio_bytes))
    inf_video_tasks[vid].add_done_callback(lambda task: delay_rm_video(300, inf_video_tasks, vid))
    return {"videoId": vid}
