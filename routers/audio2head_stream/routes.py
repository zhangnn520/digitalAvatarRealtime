from fastapi import APIRouter
from typing import Dict
from objects.audio_video_stream import AVStream
from routers.helpers1 import scheduler
from loguru import logger

router = APIRouter()
# 流队列 字典
streams: Dict[str, AVStream] = {}


@router.get("/establish_stream")
async def establish_stream(digital_man: str):
    """
    建立一个持续的流

    :return:
    """
    av_stream = AVStream(digital_man, scheduler)
    streams[av_stream.stream_id] = av_stream
    streams[av_stream.stream_id].update_del_time_from(streams)
    logger.info(f"len(streams)={len(streams)}")
    logger.debug(f"len(scheduler.get_jobs())={len(scheduler.get_jobs())}")
    return av_stream.stream_id
