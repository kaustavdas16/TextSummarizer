from transformers import pipeline
import streamlit as st

@st.cache_resource
def load_model():
    return pipeline("text-generation", model="gpt2")

def summarize_gpt2(text, max_length=100):
    generator = load_model()

    output = generator(
        text,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True
    )

    return output[0]['generated_text']