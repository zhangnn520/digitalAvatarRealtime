from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = '0.0.0.0'
    port: int = 6006

    fps: float = 25  # 帧率
