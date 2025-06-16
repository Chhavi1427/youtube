import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline

# Load summarization model (no API key needed)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

st.set_page_config(page_title="YouTube Video Summarizer", layout="centered")
st.title("🎥 YouTube Video Summarizer in Note Form")

# Function to extract transcript
def extract_transcript(youtube_url):
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    except NoTranscriptFound:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])
        except:
            st.error("No transcript found in English or Hindi.")
            return None
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None

    full_text = " ".join([i['text'] for i in transcript])
    return full_text

# Function to summarize text
def generate_summary(text):
    max_chunk = 1000
    chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]

    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=150, min_length=30, do_sample=False)
        summary += result[0]['summary_text'] + "\n\n"
    return summary

# Input field
youtube_link = st.text_input("Enter YouTube video URL")

# Show thumbnail
if youtube_link and "v=" in youtube_link:
    video_id = youtube_link.split("v=")[1].split("&")[0]
    st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)

if st.button("Generate Notes"):
    if not youtube_link:
        st.warning("Please enter a valid YouTube URL.")
    else:
        transcript_text = extract_transcript(youtube_link)
        if transcript_text:
            with st.spinner("Summarizing..."):
                summary = generate_summary(transcript_text)
                st.markdown("## 📄 Summary Notes")
                st.write(summary)
