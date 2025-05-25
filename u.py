import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
from transformers import pipeline
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from fpdf import FPDF

# Download necessary NLTK data
nltk.download('stopwords')

# Function to extract video ID from URL
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Function to fetch transcript
def fetch_transcript(video_id, max_segments=1000):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([t['text'] for t in transcript[:max_segments]])
        return transcript_text
    except:
        return None

# Function to split text into smaller chunks
def chunk_text(text, max_tokens=1024):
    words = text.split()
    chunks = [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]
    return [chunk for chunk in chunks if chunk.strip()]  # Ensure non-empty chunks

# Function to summarize text using Hugging Face
def summarize_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    summaries = []
    
    for chunk in chunk_text(text):
        try:
            summary_output = summarizer(chunk, max_length=500, min_length=200, do_sample=False)
            if summary_output and isinstance(summary_output, list) and len(summary_output) > 0:
                summaries.append(summary_output[0].get('summary_text', ''))
            else:
                print(f"Warning: Empty summary for chunk: {chunk[:50]}...")
        except Exception as e:
            print(f"Error processing chunk: {e}")

    return " ".join(summaries) if summaries else "No summary generated."

# Function to extract keywords
def extract_keywords(text, num_keywords=10):
    stop_words = set(stopwords.words('english'))
    words = [word.lower() for word in text.split() if word.isalpha() and word.lower() not in stop_words]
    most_common = Counter(words).most_common(num_keywords)
    return [word[0] for word in most_common]

# Function for topic modeling using LDA
def topic_modeling(text, num_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    text_vectorized = vectorizer.fit_transform([text])
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda_model.fit(text_vectorized)
    
    words = np.array(vectorizer.get_feature_names_out())
    topics = []
    for topic in lda_model.components_:
        top_words = [words[i] for i in topic.argsort()[:-6:-1]]
        topics.append(", ".join(top_words))
    return topics

# Function for sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive 😊"
    elif polarity < 0:
        return "Negative 😞"
    else:
        return "Neutral 😐"

# Function to generate PDF
def generate_pdf(summary, transcript):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style='B', size=16)
    pdf.cell(200, 10, "YouTube Video Summary & Transcript", ln=True, align='C')
    
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Summary:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary.encode("latin-1", "replace").decode("latin-1"))
    
    pdf.ln(5)
    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(0, 10, "Transcript:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, transcript.encode("latin-1", "replace").decode("latin-1"))
    
    pdf_path = "summary_transcript.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Streamlit UI
st.title("🎥 YouTube Video Summarizer")
st.write("Enter a YouTube video URL to generate a transcript, summary .")

# User Input
url = st.text_input("Paste YouTube Video URL:")
transcript_length = st.slider("Select Transcript Length (Segments)", 100, 1000, 500)

if st.button("Summarize"):
    video_id = extract_video_id(url)
    
    if not video_id:
        st.error("Invalid YouTube URL! Please try again.")
    else:
        transcript = fetch_transcript(video_id, max_segments=transcript_length)
        
        if not transcript:
            st.error("Could not fetch transcript!")
        else:
            with st.spinner("Processing..."):
                summary = summarize_text(transcript)
                keywords = extract_keywords(transcript)
                topics = topic_modeling(transcript)
                sentiment = analyze_sentiment(transcript)
                pdf_path = generate_pdf(summary, transcript)
            
            # Display Results
            st.subheader("📜 Summary:")
            st.write(summary)
            
            st.subheader("📜 Full Transcript:")
            st.write(transcript)
            
            
            
            # Provide PDF Download
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="summary_transcript.pdf", mime="application/pdf")