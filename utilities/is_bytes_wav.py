def is_wav(data: bytes):
    """
    判断一个二进制数据是否是wav格式的音频二进制数据
    """
    # Ensure there are enough bytes to check
    if len(data) < 16:
        return False

    # Check for the 'RIFF' identifier
    if data[0:4] != b'RIFF':
        return False

    # Check for the 'WAVE' identifier
    if data[8:12] != b'WAVE':
        return False

    # Check for the 'fmt ' subchunk identifier
    if data[12:16] != b'fmt ':
        return False

    return True

# Example usage:
# with open('audio_file.wav', 'rb') as f:
#     data = f.read()
#     print(is_wav(data))  # Output: True or False
