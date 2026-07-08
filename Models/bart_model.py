from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

def summarize_bart(text, max_length=100):

    summarizer = load_model()

    input_words = len(text.split())

    max_length = min(
        max_length,
        max(30, input_words)
    )

    min_length = min(
        max_length // 3,
        max_length - 1
    )

    return summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False
    )[0]["summary_text"]