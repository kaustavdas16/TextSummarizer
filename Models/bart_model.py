from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_bart(text, max_length=100):
    summarizer = load_model()
    return summarizer(
        text,
        max_length=max_length,
        min_length=max_length // 3,
        do_sample=False
    )[0]['summary_text']