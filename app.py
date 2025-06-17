import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import os

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def get_transcript(url):
    try:
        video_id = url.split("v=")[-1]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([t["text"] for t in transcript])
        return full_text
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_video(url):
    transcript = get_transcript(url)
    if transcript.startswith("Error:"):
        return transcript
    summary = summarizer(transcript[:1024], max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

iface = gr.Interface(
    fn=summarize_video,
    inputs=gr.Textbox(label="YouTube URL"),
    outputs=gr.Textbox(label="Summary"),
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
