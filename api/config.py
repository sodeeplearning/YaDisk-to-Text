import os

use_gpu = False
audio_file_name = "audio.mp3"
video_file_name = "video.mp4"
delete_after = False
threshold = 50

os.environ.setdefault("USE_GPU", str(use_gpu))
