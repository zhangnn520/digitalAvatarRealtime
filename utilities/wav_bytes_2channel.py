import io
import wave


def wav_bytes_2channel(single_channel_data: bytes):
    """
    将单声道wav二进制改为双声道二进制，采样率32000；
    未测试如果传入多声道音频二进制的结果
    """
    # 将单声道数据分割为16位样本
    samples = [single_channel_data[i:i + 2] for i in range(0, len(single_channel_data), 2)]

    # 将每个样本复制一次以创建双声道数据
    double_channel_data = b''.join(samples[:20] + [sample + sample for sample in samples[20:]])

    # 创建一个用于写入WAV的字节流
    output_wav = io.BytesIO()

    # 设置WAV参数
    num_channels = 2
    sampwidth = 2  # 16位样本宽度
    framerate = 32000  # 设置帧率为32000
    num_frames = len(double_channel_data) // (num_channels * sampwidth)

    # 创建WAV文件
    with wave.open(output_wav, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.setnframes(num_frames)
        wav_file.writeframes(double_channel_data)

    return output_wav.getvalue()
