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


st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: #ffffff;
}

/* Title */
h1 {
    text-align: center;
    font-size: 40px;
    margin-bottom: 10px;
}

/* Labels */
label {
    font-size: 16px !important;
    font-weight: 500;
}

/* Text Area */
textarea {
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Button */
button {
    border-radius: 12px !important;
    height: 45px;
    font-size: 16px !important;
    font-weight: 600;
}

/* Output Card */
.result-box {
    background-color: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# TITLE
st.title("Text Summarization Studio")
st.caption("Compare AI & NLP Models")

# INPUT
st.markdown("### Enter Text")
text = st.text_area("", height=200)

# Limit text size (safety)
from Utils.helpers import limit_text
text = limit_text(text)

# MODEL SELECT
st.markdown("### Select Model")
model = st.selectbox("", [
    "Pegasus", "T5", "BART", "BERT", "GPT-2",
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

        rouge = calculate_rouge(text, summary)

        # OUTPUT CARD
        st.markdown('<div class="result-box">', unsafe_allow_html=True)

        st.markdown(f"### Model: {model}")

        st.markdown("### Summary")
        st.write(summary)

        st.markdown("### ROUGE Scores")
        st.write(f"ROUGE-1: {rouge['ROUGE-1']}")
        st.write(f"ROUGE-2: {rouge['ROUGE-2']}")
        st.write(f"ROUGE-L: {rouge['ROUGE-L']}")

        st.markdown('</div>', unsafe_allow_html=True)