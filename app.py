import gradio as gr
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load Gemini 1.5 (fine-tuned instruction model from Hugging Face)
model_id = "google/gemma-1.5-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Extract video ID and transcript
def get_transcript(youtube_url):
    try:
        video_id = youtube_url.split("v=")[-1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([segment["text"] for segment in transcript])
        return full_text[:4000]  # keep it short for generation
    except Exception as e:
        return f"Error: {e}"

# Summarize
def summarize_video(url):
    transcript = get_transcript(url)
    if transcript.startswith("Error:"):
        return transcript
    prompt = f"Summarize this video transcript:\n{transcript}\n\nSummary:"
    summary = generator(prompt, max_length=256, do_sample=False)[0]["generated_text"]
    return summary[len(prompt):].strip()

# Gradio UI
gr.Interface(
    fn=summarize_video,
    inputs=gr.Textbox(label="YouTube Video URL"),
    outputs=gr.Textbox(label="Video Summary"),
    title="YouTube Video Summarizer (Gemini Flash 1.5)",
    description="Enter a YouTube video link and get a summary using Gemini Flash 1.5 model hosted on Hugging Face."
).launch()
