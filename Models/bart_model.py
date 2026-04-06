import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_bart():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_bart(text):
    summarizer = load_bart()

    result = summarizer(
        text,
        max_length=120,
        min_length=30,
        num_beams=2
    )

    return result[0]['summary_text']