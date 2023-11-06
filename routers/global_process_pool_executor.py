from concurrent.futures import ProcessPoolExecutor
from configuration.development_config import Settings

_pool_executor: ProcessPoolExecutor = None


def get_pool_executor():
    global _pool_executor
    if _pool_executor is None:
        _pool_executor = ProcessPoolExecutor(max_workers=Settings().max_workers)
    return _pool_executor


def ensure_pool_executor_closed():
    """关闭进程池执行器"""
    if _pool_executor is not None:
        _pool_executor.shutdown()
