import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

# Summarizer pipeline from Hugging Face (uses BART or T5)
summarizer = pipeline("summarization")

st.title("🎬 YouTube Video Summarizer")

url = st.text_input("Enter YouTube Video URL (e.g., https://www.youtube.com/watch?v=abc123):")

def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def get_transcript(video_id):
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    transcript = " ".join([d['text'] for d in transcript_list])
    return transcript

if st.button("Generate Summary"):
    video_id = get_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL.")
    else:
        try:
            transcript = get_transcript(video_id)
            st.subheader("📝 Video Transcript")
            st.write(transcript[:2000] + "...")

            # Summarize in chunks (for large inputs)
            max_chunk = 1000
            chunks = [transcript[i:i+max_chunk] for i in range(0, len(transcript), max_chunk)]
            summary = ""
            for chunk in chunks:
                summary_text = summarizer(chunk, max_length=150, min_length=40, do_sample=False)[0]['summary_text']
                summary += summary_text + " "

            st.subheader("✅ Summary")
            st.write(summary)

        except Exception as e:
            st.error(f"Error: {str(e)}")
