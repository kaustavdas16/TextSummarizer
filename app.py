import streamlit as st
import time
import matplotlib.pyplot as plt

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

# ✅ SESSION STATE INIT
if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "compare_results" not in st.session_state:
    st.session_state.compare_results = None

if "scores" not in st.session_state:
    st.session_state.scores = None

if "best_model" not in st.session_state:
    st.session_state.best_model = None


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

    summary = summary.strip()

    if summary:
        summary = summary[0].upper() + summary[1:]
        if not summary.endswith("."):
            summary += "."

    end = time.time()
    return summary, round(end - start, 2)

# BUTTON
if st.button("Summarize"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        # SINGLE MODE
        if mode == "Single Model":

            with st.spinner("Generating summary..."):
                summary, time_taken = run_model(model, text, length)

            rouge = calculate_rouge(text, summary)
            word_count = len(summary.split())

            # ✅ STORE
            st.session_state.single_result = {
                "model": model,
                "summary": summary,
                "time": time_taken,
                "rouge": rouge,
                "words": word_count
            }

        # COMPARE MODE
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

            scores = {m: adjusted_score(results[m], m) for m in models_list}
            best_model = max(scores, key=scores.get)

            # ✅ STORE
            st.session_state.compare_results = results
            st.session_state.scores = scores
            st.session_state.best_model = best_model


# =========================
# DISPLAY (PERSISTENT)
# =========================

# SINGLE MODE DISPLAY
if mode == "Single Model" and st.session_state.single_result:

    data = st.session_state.single_result

    st.markdown("---")

    st.markdown(f"## Model: {data['model']}")
    st.markdown(f"**Model Type :** {model_type[data['model']]}")
    st.markdown(f"**Time Taken :** {data['time']} sec")

    st.markdown("### Summary")
    st.write(data["summary"])

    st.markdown(f"**Word Count :** {data['words']}")

    st.markdown("### ROUGE Scores")
    st.markdown(f"**ROUGE-1&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{data['rouge']['ROUGE-1']}", unsafe_allow_html=True)
    st.markdown(f"**ROUGE-2&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{data['rouge']['ROUGE-2']}", unsafe_allow_html=True)
    st.markdown(f"**ROUGE-L&nbsp;&nbsp;:** &nbsp;&nbsp;&nbsp;{data['rouge']['ROUGE-L']}", unsafe_allow_html=True)

    file_content = (
        f"Model       : {data['model']}\n"
        f"Model Type  : {model_type[data['model']]}\n"
        f"Word Count  : {data['words']}\n\n"
        f"Summary: \n\n{data['summary']}\n"
    )

    st.download_button(
        label="📥 Download Summary",
        data=file_content,
        file_name=f"{data['model']}_summary.txt",
        mime="text/plain"
    )


# COMPARE MODE DISPLAY
if mode == "Compare All Models" and st.session_state.compare_results:

    results = st.session_state.compare_results
    scores = st.session_state.scores
    best_model = st.session_state.best_model

    models_list = list(results.keys())

    st.markdown("---")
    st.write("")

    # BEST MODEL BOX
    st.markdown(f"""
    <div style="
        display: block;
        width: 100%;
        padding: 18px;
        border-radius: 14px;
        background: linear-gradient(
            rgba(0, 255, 150, 0.12),
            rgba(0, 255, 150, 0.08)
        ), var(--secondary-background-color);
        border: 1px solid rgba(0, 255, 150, 0.5);
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        color: var(--text-color);
        box-shadow:
            0 0 10px rgba(0,255,150,0.4),
            0 0 20px rgba(0,255,150,0.3),
            0 0 30px rgba(0,255,150,0.2);
    ">
        🏆 BEST MODEL → {best_model}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

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

        st.markdown("---")

    total_score = sum(scores.values())
    performance_percent = {
        m: round((scores[m] / total_score) * 100, 2) if total_score > 0 else 0
        for m in models_list
    }

    sorted_models = sorted(performance_percent.items(), key=lambda x: x[1], reverse=True)

    st.markdown("## 📊 Overall Model Performance (%)")

    for i, (m, score) in enumerate(sorted_models, start=1):
        st.markdown(f"**{i}. {m}** : {score}%")

    labels = [m for m, _ in sorted_models]
    sizes = [score for _, score in sorted_models]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%')
    ax.set_title("Overall Model Performance (Quality + Efficiency)")

    st.pyplot(fig)

    st.markdown("---")

    all_content = f"🏆 BEST MODEL: {best_model}\n\n"

    for m in models_list:
        all_content += (
            f"========== {m} ==========\n"
            f"Model Type  : {model_type[m]}\n"
            f"Word Count  : {results[m]['words']}\n\n"
            f"Summary:\n{results[m]['summary']}\n\n"
            f"----------------------------------------\n\n"
        )

    st.download_button(
        label="📥 Download All Summaries",
        data=all_content,
        file_name="all_models_summary.txt",
        mime="text/plain"
    )