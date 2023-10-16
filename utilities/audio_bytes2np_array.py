from typing import Union, Iterable, Generator
import soundfile as sf
from librosa.util import valid_audio
import numpy as np
import io
from librosa import resample


def voice_bytes2array(voice_byte: bytes):
    """将音频二进制数据转为numpy的ndarray"""
    data, samplerate = sf.read(io.BytesIO(voice_byte))
    data = data.T

    valid_audio(data, mono=False)

    if data.ndim > 1:
        data = np.mean(data, axis=0)

    data = resample(y=data, orig_sr=samplerate, target_sr=16000, res_type="kaiser_best")
    return data


def voice_iter2array(iter: Union[Iterable, Generator]):
    """将可以迭代音频二进制数据的迭代器或生成器转为numpy的ndarray"""
    data, samplerate = sf.read(io.BytesIO(b''.join(iter)))
    data = data.T

    valid_audio(data, mono=False)

    if data.ndim > 1:
        data = np.mean(data, axis=0)

    data = resample(data, samplerate, 16000, res_type="kaiser_best")
    return data
