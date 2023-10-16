from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler


class AVStream:
    """音视频流"""

    def __init__(self, digital_man: str, scheduler: AsyncIOScheduler):
        self.digital_man: str = digital_man
        self._scheduler: Optional[AsyncIOScheduler] = scheduler
