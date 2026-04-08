import streamlit as st
import time

# Models
from Models.bart_model import summarize_bart
from Models.t5_model import summarize_t5
from Models.bert_model import summarize_bert
from Models.gpt2_model import summarize_gpt2
from Models.pegasus_model import summarize_pegasus
from Models.lsa_model import summarize_lsa
from Models.textrank_model import summarize_textrank
from Models.lexrank_model import summarize_lexrank
from Models.sumbasic_model import summarize_sumbasic
from Models.nltk_model import summarize_nltk

from Utils.helpers import calculate_rouge, limit_text

# Page config
st.set_page_config(page_title="Text Summarizer", layout="centered")

# 🎨 Clean styling
st.markdown("""
<style>
h1 {
    text-align: center;
    font-size: 38px;
}

.result-box {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.title("Text Summarization Studio")
st.caption("Multi-model Text Summarization with ROUGE Evaluation")

# INPUT
st.markdown("### Enter Text")
text = st.text_area("Enter your text here:", height=200)

# Limit text size
text = limit_text(text)

# 🎛️ LENGTH SLIDER
st.markdown("### Summary Length")
length = st.select_slider(
    "Choose summary length:",
    options=[50, 100, 150, 200],
    value=100
)

# MODEL SELECT
st.markdown("### Select Model")
model = st.selectbox("Choose a model:", [
    "Pegasus", "T5", "BART", "BERT (Simulated)", "GPT-2",
    "LSA", "TextRank", "LexRank", "SumBasic", "NLTK"
])

# Model Types
model_type = {
    "Pegasus": "Abstractive",
    "T5": "Abstractive",
    "BART": "Abstractive",
    "BERT (Simulated)": "Encoder-based (Simulated)",
    "GPT-2": "Generative (Not optimized)",
    "LSA": "Extractive",
    "TextRank": "Extractive",
    "LexRank": "Extractive",
    "SumBasic": "Extractive",
    "NLTK": "Extractive"
}

# BUTTON
if st.button("Summarize"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):

            start = time.time()

            # Generative models (use slider)
            if model == "Pegasus":
                summary = summarize_pegasus(text, max_length=length)
            elif model == "T5":
                summary = summarize_t5(text, max_length=length)
            elif model == "BART":
                summary = summarize_bart(text, max_length=length)
            elif model == "BERT (Simulated)":
                summary = summarize_bert(text, max_length=length)
            elif model == "GPT-2":
                summary = summarize_gpt2(text, max_length=length)

            # Extractive models
            elif model == "LSA":
                summary = summarize_lsa(text)
            elif model == "TextRank":
                summary = summarize_textrank(text)
            elif model == "LexRank":
                summary = summarize_lexrank(text)
            elif model == "SumBasic":
                summary = summarize_sumbasic(text)
            elif model == "NLTK":
                summary = summarize_nltk(text)

            end = time.time()
            time_taken = round(end - start, 2)

        # ROUGE
        rouge = calculate_rouge(text, summary)

        # OUTPUT
        st.markdown("---")

        st.markdown(f"## Model: {model}")
        st.markdown(f"**Model Type :** {model_type[model]}")
        st.markdown(f"**Time Taken :** {time_taken} sec")

        # SUMMARY
        st.markdown("### Summary")
        st.write(summary)

        # ✅ WORD COUNT (NEW)
        word_count = len(summary.split())
        st.markdown(f"**Word Count :** {word_count}")

        # DOWNLOAD BUTTON
        file_content = (
            f"Model: {model}\n"
            f"Model Type: {model_type[model]}\n\n"
            f"Summary:\n{summary}\n"
        )

        st.download_button(
            label="📥 Download Summary",
            data=file_content,
            file_name=f"{model}_summary.txt",
            mime="text/plain"
        )

        # ROUGE
        st.markdown("### ROUGE Scores")

        st.markdown(f"**ROUGE-1&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{rouge['ROUGE-1']}", unsafe_allow_html=True)
        st.markdown(f"**ROUGE-2&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{rouge['ROUGE-2']}", unsafe_allow_html=True)
        st.markdown(f"**ROUGE-L&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{rouge['ROUGE-L']}", unsafe_allow_html=True)