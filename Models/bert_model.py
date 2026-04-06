import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_bert():
    return pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_bert(text):
    summarizer = load_bert()

    result = summarizer(
        text,
        max_length=120,
        min_length=30,
        num_beams=2
    )

    return result[0]['summary_text']