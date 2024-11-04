from transformers import pipeline
import os

from api.utils import get_audio


class SpeechRecognition:
    """Class made for Speech-To-Text task."""
    def __init__(self, use_gpu: bool = False, model_name: str = "openai/whisper-large-v3"):
        """Constructor of a SpeechRecognition class.

        :param use_gpu: 'True' if you need to use GPU.
        :param model_name: STT model's name.
        """
        self.device = "cuda" if use_gpu else "cpu"
        self.model = pipeline(
            task="automatic-speech-recognition",
            model=model_name,
            device=self.device
        )

    def __call__(self, audio_file: str, *args, **kwargs) -> str:
        """Get brief from audio file

        :param audio_file: Audio file path.
        :return: Text of audio file.
        """
        return self.model(audio_file)["text"]


class TextSummarization:
    """Class made for text summarization task."""

    def __init__(
        self,
        model_name: str = "UrukHan/t5-russian-summarization",
        use_gpu: bool = False,
        threshold: int = 15
    ):
        """Init func of TextSummarization class.

        :param model_name: Name of the Summarization model.
        :param use_gpu: 'True' if you need to use GPU.
        """
        self.device = "cuda" if use_gpu else "cpu"
        self.model = pipeline(model=model_name, device=self.device)
        self.threshold = threshold

    def __call__(self, text: str, *args, **kwargs) -> str:
        """Get summarization of the text.

        :param text: Text to get summarization.
        :return: Brief of text.
        """
        if isinstance(text, str):
            if len(text) > self.threshold:
                output = self.model(text)
                return output[0]["generated_text"]
            return text


class VideoToText:
    """Class made for Video-To-Text task."""
    def __init__(self,
                 use_gpu: bool = False,
                 audio_file: str = "audio.mp3",
                 video_saving_path: str = "video.mp4",
                 threshold: int = 50,
                 delete_after_sum: bool = False):
        """Constructor of VideoToText class.

        :param use_gpu: 'True' if you need to use GPU.
        :param audio_file: Path to an audio file.
        :param video_saving_path: Path for saving loaded video.
        :param threshold: Limit for not-summarized text.
        """
        self.delete_after_sum = delete_after_sum
        self.audio_file = audio_file
        self.video_saving_path = video_saving_path

        self.STT = SpeechRecognition(use_gpu=use_gpu)
        self.Summarizer = TextSummarization(use_gpu=use_gpu, threshold=threshold)

    def __call__(self, video_link: str, *args, **kwargs):
        """Get summary of audio text.

        :param video_link: Link to a video.
        :return: Summarized text extracted from video file.
        """
        get_audio(video_link=video_link,
                  audio_saving_path=self.audio_file,
                  video_saving_path=self.video_saving_path)
        extracted_text = self.STT(self.audio_file)

        if self.delete_after_sum:
            os.remove(self.audio_file)
            os.remove(self.video_saving_path)

        return self.Summarizer(extracted_text)
