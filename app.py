import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import os

# Load summarization pipeline using a small, efficient public model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Extract transcript from YouTube
def get_transcript(url):
    try:
        video_id = url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([segment["text"] for segment in transcript])
    except Exception as e:
        return f"Error: {str(e)}"

# Summarization function
def summarize_video(url):
    transcript = get_transcript(url)
    if transcript.startswith("Error:"):
        return transcript
    shortened_text = transcript[:1024]  # Max token length for this model
    summary = summarizer(shortened_text, max_length=130, min_length=30, do_sample=False)
    return summary[0]["summary_text"]

# Gradio UI
iface = gr.Interface(
    fn=summarize_video,
    inputs=gr.Textbox(label="YouTube Video URL"),
    outputs=gr.Textbox(label="Video Summary"),
    title="📺 YouTube Summarizer App",
    description="Paste a YouTube link and get a concise summary using a Transformer model."
)

# Launch with port binding (for Render/Netlify backend-compatible platforms)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
