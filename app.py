import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

model_id = "google/flan-t5-small"  # ✅ public model

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


# Function to extract transcript from YouTube video
def get_transcript(youtube_url):
    try:
        if "v=" in youtube_url:
            video_id = youtube_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[-1].split("?")[0]
        else:
            return "Error: Invalid YouTube URL format."

        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([segment["text"] for segment in transcript])
        return full_text[:4000]  # Truncate if too long
    except Exception as e:
        return f"Error: {str(e)}"

# Summarize the transcript using the Gemini model
def summarize_video(url):
    transcript = get_transcript(url)
    if transcript.startswith("Error:"):
        return transcript
    prompt = f"Summarize this video transcript:\n{transcript}"
    summary = generator(prompt, max_length=200)[0]['generated_text']
    return summary


# Gradio app
gr.Interface(
    fn=summarize_video,
    inputs=gr.Textbox(label="YouTube Video URL"),
    outputs=gr.Textbox(label="Video Summary", lines=10),
    title="📺 YouTube Video Summarizer",
    description="Summarizes YouTube videos using transcripts + Gemini Flash 1.5 (via Gemma 2B Instruct model)."
).launch()
