import asyncio
from asyncio import Queue

from objects import QUEUE_OVER
from typing import Optional, Callable
from concurrent.futures import CancelledError
import traceback


async def async_q2async_generator(q: Queue,
                                  timeout: Optional[int] = None,
                                  cancelled_callback: Callable = None):
    """将一个异步队列转化为异步生成器，要求队列数据以QUEUE_OVER表示生成器的结束"""
    while True:
        try:
            if timeout is None:
                data = await q.get()
            else:
                data = await asyncio.wait_for(q.get(), timeout=timeout)
            q.task_done()
        except asyncio.TimeoutError:
            break
        except CancelledError:
            traceback.print_exc()
            if cancelled_callback is not None:
                # 普通函数
                if not asyncio.iscoroutinefunction(cancelled_callback):
                    cancelled_callback()
                else:  # 异步函数
                    await cancelled_callback()

            raise
        else:
            # 结束标志
            if data == QUEUE_OVER:
                break
            yield data
