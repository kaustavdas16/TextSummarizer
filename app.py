import streamlit as st

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

# 🎨 Clean styling (no glitch-causing CSS)
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

# Limit text size (safe handling)
text = limit_text(text)

# MODEL SELECT
st.markdown("### Select Model")
model = st.selectbox("Choose a model:", [
    "Pegasus", "T5", "BART", "BERT (Simulated)", "GPT-2",
    "LSA", "TextRank", "LexRank", "SumBasic", "NLTK"
])

# BUTTON
if st.button("Summarize"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Generating summary..."):

            if model == "Pegasus":
                summary = summarize_pegasus(text)
            elif model == "T5":
                summary = summarize_t5(text)
            elif model == "BART":
                summary = summarize_bart(text)
            elif model == "BERT":
                summary = summarize_bert(text)
            elif model == "GPT-2":
                summary = summarize_gpt2(text)
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

        # Calculate ROUGE
        rouge = calculate_rouge(text, summary)

        # OUTPUT
        st.markdown("---")

        st.markdown(f"## Model: {model}")

        st.markdown("### Summary")
        st.write(summary)

        st.markdown("### ROUGE Scores")

        st.markdown(f"**ROUGE-1&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{rouge['ROUGE-1']}", unsafe_allow_html=True)
        st.markdown(f"**ROUGE-2&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{rouge['ROUGE-2']}", unsafe_allow_html=True)
        st.markdown(f"**ROUGE-L&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{rouge['ROUGE-L']}", unsafe_allow_html=True)