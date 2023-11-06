import requests


def uploadAv():
    """上传音视频"""
    url = "http://127.0.0.1:6006/inferenceVideoV2/uploadAv"

    audio = "1697513088193.wav"
    video = "yangshi.mp4"

    files = {'video': open(video, 'rb'), 'audio': open(audio, 'rb')}
    result = requests.post(url=url, files=files)
    print(result.text)


if __name__ == '__main__':
    uploadAv()
