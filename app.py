import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Set Streamlit page config
st.set_page_config(page_title="YouTube Video Summarizer", layout="centered")

st.title("🎬 YouTube Video Bullet Point Summarizer")

# Hugging Face summarization model
model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Extract YouTube video ID
def extract_video_id(url):
    import re
    match = re.search(r"(?:v=|youtu\.be/)([^&]+)", url)
    return match.group(1) if match else None

# Summarize transcript text into bullet points
def summarize_text(text):
    prompt = "Summarize the following into concise bullet points:\n\n" + text
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(device)
    outputs = model.generate(inputs, max_length=250, min_length=50, do_sample=False)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# Streamlit input
video_url = st.text_input("Enter YouTube Video URL")

if video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            full_text = " ".join([entry['text'] for entry in transcript])
            st.info("Generating summary... This may take a few seconds.")
            summary = summarize_text(full_text)

            st.success("✅ Summary:")
            for bullet in summary.split("\n"):
                if bullet.strip():
                    st.markdown(f"- {bullet.strip()}")
        except Exception as e:
            st.error(f"⚠️ Could not fetch transcript: {e}")
    else:
        st.error("❌ Invalid YouTube URL format.")
