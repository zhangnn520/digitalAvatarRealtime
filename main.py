import os

os.environ['USE_SIMPLE_THREADED_LEVEL3'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from loguru import logger
import git

if not os.path.exists("./DINet"):
    logger.info("Download DINet...")
    git.Repo.clone_from("https://github.com/monk-after-90s/DINet.git", "./DINet")
import sys

sys.path.append(os.path.abspath("./DINet"))

import multiprocessing
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from configuration import Settings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    from preprocess import preload_videos
    # 预加载视频
    preload_videos()


@app.on_event("shutdown")
async def shutdown_event():
    ...


if __name__ == "__main__":
    logger.info(f"multiprocessing.set_start_method:fork")
    multiprocessing.set_start_method("fork", True)
    # 这个项目只能启动Python脚本而不能启动uvicorn
    develop_mode = os.getenv("PYTHONUNBUFFERED") == "1"

    settings = Settings()
    uvicorn.run("main:app",
                host=settings.host,
                port=settings.port,
                log_level="debug" if develop_mode else None,
                reload=True if develop_mode else False
                )
