import requests
from urllib.parse import urlencode
from moviepy.editor import VideoFileClip


disk_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'


def __download_video(video_link: str, video_path: str):
    request_url = disk_url + urlencode(dict(public_key=video_link))
    response = requests.get(request_url)
    download_url = response.json()["href"]

    loaded_file = requests.get(download_url).content
    with open(video_path, "wb") as video_file:
        video_file.write(loaded_file)


def __video_to_audio(mp4file: str, mp3file: str):
    video = VideoFileClip(mp4file)
    audio = video.audio
    audio.write_audiofile(mp3file)
    audio.close()
    video.close()


def get_audio(video_link: str,
              audio_saving_path: str = "audio.mp3",
              video_saving_path: str = "video.mp4"):
    __download_video(video_link, video_saving_path)
    __video_to_audio(video_saving_path, audio_saving_path)

