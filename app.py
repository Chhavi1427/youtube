import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Configure API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("API key not found. Please check your .env file.")
else:
    genai.configure(api_key=api_key)

# Correct prompt
prompt = """You are a YOUTUBE video summarizer. You will be taking the transcript text and summarizing the entire video, providing the important summary in points within 1000 words. The transcript text will be appended here: """

# Extract transcript with fallback
def extract_transcript_details(youtube_video_url):
    try:
        video_id = youtube_video_url.split("=")[1].split("&")[0]
       

        # Attempt to fetch English transcript
        try:
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        except NoTranscriptFound:
            
            transcript_text = YouTubeTranscriptApi.get_transcript(video_id, languages=['hi'])

        transcript = " ".join(i["text"] for i in transcript_text)
        return transcript

    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript available for this video in supported languages.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    return None

# Generate summary using Gemini
def generate_gemini_content(transcript_text, prompt):
    try:
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
        response = model.generate_content(prompt + transcript_text)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

# Streamlit app UI
st.title("YouTube Video Summarizer in Form of Notes")
youtube_link = st.text_input("Enter YouTube URL:")

if youtube_link:
    if "=" in youtube_link:
        video_id = youtube_link.split("=")[1].split("&")[0]
        st.image(f"http://img.youtube.com/vi/{video_id}/0.jpg", use_column_width=True)
    else:
        st.error("Invalid YouTube URL. Please enter a valid video link.")

if st.button("Generate Detailed Notes"):
    transcript_text = extract_transcript_details(youtube_link)

    if transcript_text:
        summary = generate_gemini_content(transcript_text, prompt)
        if summary:
            st.markdown("## Detailed Notes:")
            st.write(summary)


