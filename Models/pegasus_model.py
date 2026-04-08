import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

@st.cache_resource
def load_model():
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    return tokenizer, model

def split_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def summarize_pegasus(text, max_length=100):
    tokenizer, model = load_model()

    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True)

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=max_length // 3,
            num_beams=4
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)