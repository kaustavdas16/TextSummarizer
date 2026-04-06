import streamlit as st
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# ✅ Cache model (VERY IMPORTANT)
@st.cache_resource
def load_pegasus():
    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
    return tokenizer, model

# ✅ Split long text into chunks
def split_text(text, max_words=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks

# ✅ Main summarization function
def summarize_pegasus(text):
    tokenizer, model = load_pegasus()

    chunks = split_text(text, max_words=200)

    final_summary = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True
        )

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=100,
            min_length=30,
            num_beams=4
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        final_summary.append(summary)

    return " ".join(final_summary)