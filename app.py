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

# TITLE
st.title("Text Summarization Studio")
st.caption("Multi-model Text Summarization with ROUGE Evaluation")

# INPUT
st.markdown("### Enter Text")
text = st.text_area("Enter your text here:", height=200)
text = limit_text(text)

# MODE
mode = st.radio("Choose Mode:", ["Single Model", "Compare All Models"])

# SLIDER
st.markdown("### Summary Length")
length = st.select_slider(
    "Choose summary length:",
    options=[50, 100, 150, 200],
    value=100
)

# MODEL SELECT
if mode == "Single Model":
    st.markdown("### Select Model")
    model = st.selectbox("Choose a model:", [
        "Pegasus", "T5", "BART", "BERT (Simulated)", "GPT-2",
        "LSA", "TextRank", "LexRank", "SumBasic", "NLTK"
    ])

# MODEL TYPES
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

# RUN FUNCTION
def run_model(model, text, length):
    start = time.time()

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

    
    # ✅ CLEANING + FORMATTING
    summary = summary.strip()

    if summary:
        summary = summary[0].upper() + summary[1:]

        # Optional: ensure it ends with a period
        if not summary.endswith("."):
            summary += "."


    end = time.time()
    return summary, round(end - start, 2)

# BUTTON
if st.button("Summarize"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        # 🔹 SINGLE MODE
        if mode == "Single Model":

            with st.spinner("Generating summary..."):
                summary, time_taken = run_model(model, text, length)

            rouge = calculate_rouge(text, summary)
            word_count = len(summary.split())

            st.markdown(f"## Model: {model}")
            st.markdown(f"**Model Type :** {model_type[model]}")
            st.markdown(f"**Time Taken :** {time_taken} sec")

            st.markdown("### Summary")
            st.write(summary)

            st.markdown(f"**Word Count :** {word_count}")

            
            st.markdown("### ROUGE Scores")
            st.markdown(f"**ROUGE-1&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{rouge['ROUGE-1']}", unsafe_allow_html=True)
            st.markdown(f"**ROUGE-2&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{rouge['ROUGE-2']}", unsafe_allow_html=True)
            st.markdown(f"**ROUGE-L&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{rouge['ROUGE-L']}", unsafe_allow_html=True)


            file_content = (
                f"Model       : {model}\n"
                f"Model Type  : {model_type[model]}\n"
                f"Word Count  : {word_count}\n\n"
                f"Summary: \n\n{summary}\n"
            )

            st.download_button(
                label="📥 Download Summary",
                data=file_content,
                file_name=f"{model}_summary.txt",
                mime="text/plain"
            )


        # 🔹 COMPARE MODE
        else:

            models_list = list(model_type.keys())
            results = {}

            with st.spinner("Running all models..."):
                for m in models_list:
                    summary, time_taken = run_model(m, text, length)
                    rouge = calculate_rouge(text, summary)
                    word_count = len(summary.split())

                    results[m] = {
                        "summary": summary,
                        "rouge": rouge,
                        "time": time_taken,
                        "words": word_count
                    }

            # BEST MODEL (UNCHANGED)
            def adjusted_score(data, model_name):
                original_len = len(text.split())
                if original_len == 0:
                    return 0

                compression = data["words"] / original_len
                rouge = data["rouge"]["ROUGE-1"]

                if compression > 0.8:
                    score = rouge * 0.1
                elif compression > 0.6:
                    score = rouge * 0.4
                else:
                    score = rouge * (1 - compression)

                if model_name in ["Pegasus", "T5", "BART"]:
                    score *= 1.3

                return score

            best_model = max(results, key=lambda x: adjusted_score(results[x], x))

            st.markdown(f"## 🏆 Best Model: {best_model}")

            # SIMPLE SEPARATOR
            st.markdown("---")

            # SIMPLE DISPLAY
            for m in models_list:

                st.markdown(f"## Model: {m}")
                st.markdown(f"**Model Type :** {model_type[m]}")
                st.markdown(f"**Time Taken :** {results[m]['time']} sec")

                st.markdown("### Summary")
                st.write(results[m]["summary"])

                st.markdown(f"**Word Count :** {results[m]['words']}")

                st.markdown("### ROUGE Scores")
                st.markdown(f"**ROUGE-1&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{results[m]['rouge']['ROUGE-1']}", unsafe_allow_html=True)
                st.markdown(f"**ROUGE-2&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{results[m]['rouge']['ROUGE-2']}", unsafe_allow_html=True)
                st.markdown(f"**ROUGE-L&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{results[m]['rouge']['ROUGE-L']}", unsafe_allow_html=True)

                file_content = (
                    f"Model       : {m}\n"
                    f"Model Type  : {model_type[m]}\n"
                    f"Word Count  : {results[m]['words']}\n\n"
                    f"Summary: \n\n{results[m]['summary']}\n"
                )

                st.download_button(
                    label=f"📥 Download {m} Summary",
                    data=file_content,
                    file_name=f"{m}_summary.txt",
                    mime="text/plain"
                )

                # SIMPLE SEPARATOR
                st.markdown("---")