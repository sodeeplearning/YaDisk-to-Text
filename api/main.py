from fastapi import FastAPI
import uvicorn

from api.utils.ml import VideoToText
import config
import models


app = FastAPI()

vtt_model = VideoToText(
    use_gpu=config.use_gpu,
    audio_file=config.audio_file_name,
    video_saving_path=config.video_file_name,
    delete_after_sum=config.delete_after,
    threshold=config.threshold
)


@app.get("/")
def start():
    return "Welcome to Video-Text application!"


@app.post("/api/sumvideo")
def summarize_video(body: models.TextModel):
    """Get summarization from a video link."""
    return models.TextModel(text=vtt_model(body.text))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)