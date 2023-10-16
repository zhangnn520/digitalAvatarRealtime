import asyncio
from typing import AsyncGenerator
from objects import QUEUE_OVER


def async_generator2q(gen: AsyncGenerator, q: asyncio.Queue = None):
    """将一个异步生成器转化为异步队列，队列数据以QUEUE_OVER表示生成器的结束"""
    if q is None:
        q = asyncio.Queue()
    asyncio.create_task(_generate(gen, q))
    return q


async def _generate(gen: AsyncGenerator, q: asyncio.Queue):
    async for data in gen:
        q.put_nowait(data)
    q.put_nowait(QUEUE_OVER)
