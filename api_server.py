import os

os.environ['USE_SIMPLE_THREADED_LEVEL3'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from loguru import logger
import multiprocessing
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import Settings

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
    ...


@app.on_event("shutdown")
async def shutdown_event():
    ...


if __name__ == "__main__":
    logger.info(f"multiprocessing.set_start_method:fork")
    multiprocessing.set_start_method("fork", True)
    # 这个项目只能启动Python脚本而不能启动uvicorn
    develop_mode = os.getenv("PYTHONUNBUFFERED") == "1"

    settings = Settings()
    uvicorn.run("api_server:app",
                host=settings.host,
                port=settings.port,
                log_level="debug" if develop_mode else None,
                reload=True if develop_mode else False
                )
