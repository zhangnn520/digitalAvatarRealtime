from concurrent.futures import ProcessPoolExecutor
from configuration.development_config import Settings

_pool_executor: ProcessPoolExecutor = None


def get_pool_executor():
    global _pool_executor
    if _pool_executor is None:
        _pool_executor = ProcessPoolExecutor(max_workers=Settings().max_workers)
    return _pool_executor
