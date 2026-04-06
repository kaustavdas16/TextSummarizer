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

from Utils.helpers import calculate_rouge

st.set_page_config(page_title="Text Summarizer", layout="centered")

st.title("🧠 Text Summarization Studio")

# INPUT
text = st.text_area("Enter your text:", height=200)

# MODEL SELECT
model = st.selectbox("Select Model", [
    "Pegasus", "T5", "BART", "BERT", "GPT-2",
    "LSA", "TextRank", "LexRank", "SumBasic", "NLTK"
])

# BUTTON
if st.button("Summarize"):

    if text.strip() == "":
        st.warning("Enter text first")
    else:
        with st.spinner("Processing..."):

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

        st.markdown("---")
        st.subheader(f"Model: {model}")

        st.write(summary)

        rouge = calculate_rouge(text, summary)

        st.markdown("### ROUGE Scores")
        st.write(rouge)