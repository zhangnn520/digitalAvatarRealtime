import fnmatch
from starlette.responses import FileResponse
from fastapi import APIRouter
from fastapi import UploadFile, File
import asyncio
from asyncio import Task
import uuid
from typing import Dict
import os
from routers.helper2 import inf_video
from utilities.is_bytes_wav import is_wav
from fastapi import HTTPException, Query
import aiofiles

router = APIRouter()

inf_video_tasks: Dict[str, Task] = {}


@router.post("/uploadAv", summary=
"""
接收音频和视频，用于数字人视频合成。返回的vid用于获取合成结果。
Python Demo:

import requests


def uploadAv():
    '''上传音视频'''
    url = "http://127.0.0.1:6006/inferenceVideoV2/uploadAv"

    audio = "1697513088193.wav"
    video = "yangshi.mp4"

    files = {'video': open(video, 'rb'), 'audio': open(audio, 'rb')}
    result = requests.post(url=url, files=files)
    print(result.text)


if __name__ == '__main__':
    uploadAv()


""", tags=["音视频"])
async def uploadAv(audio: UploadFile = File(..., title="Audio", description="让数字人说话的音频"),
                   video: UploadFile = File(..., title="Video", description="合成数字人的基础视频")):
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


@router.get("/downloadVideo",
            summary="在浏览器中下载合成的视频，比如直接粘贴URL‘http://127.0.0.1:6006/inferenceVideoV2/downloadVideo?vid=b299784a-854c-4dee-92be-1e9e7755be52’到地址栏然后Enter，正常会触发下载。使用代码下载我就不演示了，原理一样。")
def downloadVideo(vid: str = Query(title="video id", description="The ID for video downloading")):
    """
    下载vid对应的合成视频
    """
    if vid not in inf_video_tasks.keys():
        raise HTTPException(status_code=404, detail="不存在的Video ID")
    if not inf_video_tasks[vid].done():
        raise HTTPException(status_code=404, detail="视频文件不存在，因为文件正在合成中")
    # 触发可能的报错
    inf_video_tasks[vid].result()

    target_dir = f"result_videos/{vid}"
    pattern = '*.mp4'
    matches = []
    for root, dirnames, filenames in os.walk(target_dir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    if not matches:
        raise HTTPException(status_code=404, detail="视频文件不存在，因为文件过期")

    return FileResponse(matches[0], filename=os.path.basename(matches[0]))
