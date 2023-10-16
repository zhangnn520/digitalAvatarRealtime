import os
from loguru import logger
from aiohttp import ClientSession
from collections import namedtuple
import time

# 10分寿命的token
Token = namedtuple("Token", ["create_ts", "token_s"])
token: Token = None


async def get_available_token():
    """获取可用的token"""
    global token
    if token is None or time.time() - token.create_ts > 9 * 60:  # 更新的时机
        async with ClientSession(headers={"Ocp-Apim-Subscription-Key": os.environ['RESOURCE_KEY']}) as session:
            async with session.post("https://eastasia.api.cognitive.microsoft.com/sts/v1.0/issuetoken") as resp:
                token = Token(token_s=await resp.text(), create_ts=time.time())
                logger.info(f"New azure resource token：{token}")
    return token.token_s


async def azure_text2speech(text: str,
                            ssml: str = """
                                            <speak version='1.0' xml:lang='zh-CN'>
                                                <voice name='zh-CN-YunzeNeural'>
                                                    {}
                                                </voice>
                                            </speak>
                                        """
                            ):
    """
    文本转语音

    ssml: ssml标记语言模板
    """
    global token
    async with ClientSession(headers={
        "Authorization": f"Bearer {await get_available_token()}",
        "Content-Type": "application/ssml+xml",
        "X-Microsoft-OutputFormat": "riff-24khz-16bit-mono-pcm"}) as session:#todo 可以流的格式
        async with session.post("https://southeastasia.tts.speech.microsoft.com/cognitiveservices/v1",
                                data=ssml.format(text)) as resp:
            return await resp.read()
