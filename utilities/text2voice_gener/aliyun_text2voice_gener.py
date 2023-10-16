import asyncio
import json
import requests
from aiohttp import ClientSession
from utilities.wav_bytes_2channel import wav_bytes_2channel

host = 'nls-gateway.aliyuncs.com'
url = 'https://' + host + '/stream/v1/tts'

sem = None

common_payload = {
    "format": "mp3",
    "sample_rate": 32000
}


async def text2voice(token: str, appkey: str, text: str, voice=None):
    """说出文本"""
    payload = {
        "appkey": appkey,
        "text": text,
        "token": token
    }
    # 说话人
    if voice:
        payload['voice'] = voice

    payload.update(common_payload)

    global sem
    if sem is None:
        sem = asyncio.Semaphore(2)  # fixme

    async with sem:
        async with ClientSession(headers={"Content-Type": "application/json"}) as session:
            async with session.post(url, data=json.dumps(payload)) as resp:
                resp.raise_for_status()
                if 'audio/mpeg' == resp.content_type:
                    return await resp.read()


async def text2voice_gener(token: str, appkey: str, text: str):
    """说出文本的语音生成器，不需要实现语音流 todo 延迟高的话 封装类单例只用一个session 再不行阿里云请求也流起来，更精细的流"""
    payload = {
        "appkey": appkey,
        "text": text,
        "token": token
    }
    payload.update(common_payload)

    async with ClientSession(headers={"Content-Type": "application/json"}) as session:
        async with session.post(url, data=json.dumps(payload)) as resp:
            resp.raise_for_status()
            if 'audio/mpeg' == resp.content_type:
                v_data = await resp.read()

                v_data_2chan = await asyncio.get_running_loop().run_in_executor(None, wav_bytes_2channel, v_data)
                yield v_data_2chan
                # 用来测试音频数据是不是流起来了
                await asyncio.sleep(5)
                print("text2voice_gener over")


def sync_text2voice_gener(token: str, appkey: str, text: str):
    payload = {
        "appkey": appkey,
        "text": text,
        "token": token,
    }
    payload.update(common_payload)

    response = requests.post(url, stream=True, headers={"Content-Type": "application/json"}, json=payload)
    for chunk in response.iter_content(chunk_size=1024):
        yield chunk


# async def text2voice_gener(TOKEN: str, APPKEY: str, TEXT: str):
#     """
#     说出文本的语音流生成器
#
#     :return:
#     """
#     q = asyncio.Queue()
#     asyncio.get_running_loop().run_in_executor(None, read_text, q, TOKEN, APPKEY, TEXT)
#     while True:
#         try:
#             yield await asyncio.wait_for(q.get(), 5)
#         except asyncio.TimeoutError:
#             print("audio over")
#             return
#
#


if __name__ == '__main__':
    ...
