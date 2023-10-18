from fastapi import APIRouter
from fastapi import UploadFile, File
import asyncio
from asyncio import Task
import uuid
from typing import Dict
import os
from starlette.responses import FileResponse
import fnmatch
from fastapi import HTTPException
from routers.helpers import inf_video

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
    inf_video_tasks[vid] = asyncio.create_task(inf_video(filename, audio_bytes, video_bytes, inf_video_tasks, vid))
    return {"videoId": vid}


@router.get("/downloadVideo")
def downloadVideo(vid: str):
    if vid not in inf_video_tasks.keys():
        raise HTTPException(status_code=404, detail="不存在的Video ID")
    target_dir = f"temp/{vid}"
    pattern = '*facial_dubbing_add_audio.mp4'
    matches = []
    for root, dirnames, filenames in os.walk(target_dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    if not matches:
        if inf_video_tasks[vid].done():
            raise HTTPException(status_code=404, detail="视频文件不存在，因为文件过期")
        else:
            raise HTTPException(status_code=404, detail="视频文件不存在，因为文件正在合成中")

    return FileResponse(matches[0], filename=os.path.basename(matches[0]))
