import streamlit as st
import time
import matplotlib.pyplot as plt
import re
import pandas as pd
import numpy as np
import html

from nltk.translate.bleu_score import (SmoothingFunction, sentence_bleu)

# =========================
# ENGLISH MODELS
# =========================

from Models.bart_model import summarize_bart
from Models.t5_model import summarize_t5
from Models.gpt2_model import summarize_gpt2
from Models.pegasus_model import summarize_pegasus
from Models.lsa_model import summarize_lsa
from Models.textrank_model import summarize_textrank
from Models.lexrank_model import summarize_lexrank
from Models.sumbasic_model import summarize_sumbasic
from Models.nltk_model import summarize_nltk

# =========================
# MULTILINGUAL MODELS
# =========================

from Models.xlsum_mt5_model import summarize_xlsum_mt5
# from Models.mbart_model import summarize_mbart
from Models.mt5_small_model import summarize_mt5_small

from Utils.helpers import calculate_rouge, limit_text


from screenshot_mode import HIDE_STREAMLIT_STYLE    #For screenshots
SCREENSHOT_MODE = True     #Set to True when taking screenshots


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Multilingual Text Summarizer",
    layout="centered",
    initial_sidebar_state="expanded" if SCREENSHOT_MODE else "auto"
)

if SCREENSHOT_MODE:
    st.markdown(
        HIDE_STREAMLIT_STYLE,
        unsafe_allow_html=True
    )


# ========================
# DOWNLOAD BUTTON CSS
# ========================

st.markdown("""
<style>

div.stDownloadButton > button {

    width:100%;
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.10);
    border-radius:12px;
    font-weight:600;
    font-size:16px;
    padding:10px 18px;
    margin-left:10px;
    transition:all 0.2s ease;

}

div.stDownloadButton > button:hover {

    background:rgba(0,255,150,0.08);
    border:1px solid rgba(0,255,150,0.45);
            
    # color:white;

}

div.stDownloadButton > button:focus {

    box-shadow:none;

}

</style>
""", unsafe_allow_html=True)


#=========================
#GENERATE BUTTON CSS
#=========================

st.markdown("""
<style>

/* Generate Summary Button */
div.stButton > button {

    width:100%;
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.10);
    border-radius:12px;
    # color:white;
    font-size:17px;
    font-weight:600;
    padding:12px 18px;
    box-shadow:0 2px 8px rgba(0,0,0,0.18);
    margin-bottom:20px;
    );
            
    transition:
        background-color .18s ease,
        border-color .18s ease,
        box-shadow .18s ease;
}

/* Hover */
div.stButton > button:hover {
    background:rgba(255,255,255,0.09);
    border:1px solid rgba(255,255,255,0.18);
    box-shadow:0 5px 14px rgba(0,0,0,0.28);
}

/* Click */
div.stButton > button:active {
    background:rgba(255,255,255,0.11);
    box-shadow:0 2px 6px rgba(0,0,0,0.18);
}

/* Remove blue outline */
div.stButton > button:focus{
    outline:none;
}
</style>
""", unsafe_allow_html=True)


# =========================
# SESSION STATE
# =========================

if "compare_results" not in st.session_state:
    st.session_state.compare_results = None

if "scores" not in st.session_state:
    st.session_state.scores = None

if "best_model" not in st.session_state:
    st.session_state.best_model = None


# =========================
# TITLE
# =========================

st.markdown("""
<div style="
    font-size:2.7rem;
    font-weight:700;
    margin-top:20px;
    margin-bottom:5px;
    line-height:1.2;
    text-align:center;
">
🌍 Multilingual Text Summarization Studio
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="
    font-size:1rem;
    font-weight:500;
    margin-top:15px;
    margin-bottom:40px;
    line-height:1.2;
    text-align:center;
    color:rgba(255,255,255,0.75);
">
AI-powered multilingual text summarization with model comparison
</div>
""", unsafe_allow_html=True)

# ===========================
# LANGUAGE & MODE SELECTION
# ===========================

col1, col2 = st.columns([1.2, 1])

with col1:

    st.markdown("""
    <div style="
        font-size:1.2rem;
        font-weight:500;
        margin-top:20px;
        margin-bottom:10px;
    ">
        🌐 Language
    </div>
    """, unsafe_allow_html=True)

    

    language = st.selectbox(
        "",
        [
            "English",
            "Bengali",
            "Hindi"
        ],
        label_visibility="collapsed"
    )

with col2:

    st.markdown("""
    <div style="
        font-size:1.2rem;
        font-weight:500;
        margin-top:20px;
        margin-bottom:10px;
    ">
        ⚙️ Mode
    </div>
    """, unsafe_allow_html=True)

    mode = st.selectbox(
        "",
        [
            "Single Model",
            "Compare All Models"
        ],
        label_visibility="collapsed"
    )


# =========================
# INPUT
# =========================

st.markdown("""
<div style="
    font-size:1.4rem;
    font-weight:500;
    margin-top:20px;
    margin-bottom:10px;
    margin-left:5px;
">
Enter Text
</div>
""", unsafe_allow_html=True)

text = st.text_area(
    label="",
    placeholder="Paste or type your text here...",
    height=220,
    label_visibility="collapsed"
)

text = limit_text(text)


# =========================
# AVAILABLE MODELS
# =========================

if language == "English":

    available_models = [

        "Pegasus",
        "T5",
        "BART",
        "GPT-2",
        "LSA",
        "TextRank",
        "LexRank",
        "SumBasic",
        "NLTK"
    ]

else:

    available_models = [

        "XLSum-mT5",
        # "mBART-50",
        "mT5-small"
    ]

# =========================
# MODEL SELECTION
# =========================

if mode == "Single Model":

    st.markdown("""
    <div style="
        font-size:1.2rem;
        font-weight:500;
        margin-top:20px;
        margin-bottom:10px;
        margin-left:5px;
    ">
        Choose Model
    </div>
    """, unsafe_allow_html=True)

    selected_model = st.selectbox(
        "",
        available_models,
        label_visibility="collapsed"
    )

# =========================
# SUMMARY LENGTH
# =========================

st.markdown("""
<div style="
    font-size:1.8rem;
    font-weight:500;
    margin-top:20px;
    margin-bottom:10px;
    margin-left:5px;
">
Summary Length
</div>
""", unsafe_allow_html=True)

# ENGLISH
st.markdown("""
<div style="
    font-size:1.2rem;
    font-weight:500;
    margin-top:20px;
    margin-bottom:5px;
">
    📏 Choose Summary Length
</div>
""", unsafe_allow_html=True)

if language == "English":

    length = st.select_slider(
        "",
        options=[50, 100, 150, 200],
        value=100,
        label_visibility="collapsed"
    )

# MULTILINGUAL
else:

    summary_size = st.select_slider(
        "",
        options=["Short", "Medium", "Long"],
        value="Medium",
        label_visibility="collapsed"
    )

    if summary_size == "Short":

        length = 100

    elif summary_size == "Medium":

        length = 150

    else:

        length = 220

# =========================
# MODEL TYPES
# =========================

model_type = {

    # ENGLISH
    "Pegasus": "Abstractive",
    "T5": "Abstractive",
    "BART": "Abstractive",
    "GPT-2": "Generative (Not optimized)",
    "LSA": "Extractive",
    "TextRank": "Extractive",
    "LexRank": "Extractive",
    "SumBasic": "Extractive",
    "NLTK": "Extractive",

    # MULTILINGUAL
    "XLSum-mT5":
        "Multilingual Transformer",

    # "mBART-50":
    #     "Multilingual Seq2Seq Transformer",

    "mT5-small":
        "Lightweight Multilingual Transformer"
}

# =========================
# RUN MODEL
# =========================

def run_model(model, text, length):

    start = time.time()

    # =========================
    # ENGLISH MODELS
    # =========================

    if model == "Pegasus":

        summary = summarize_pegasus(
            text,
            max_length=length
        )

    elif model == "T5":

        summary = summarize_t5(
            text,
            max_length=length
        )

    elif model == "BART":

        summary = summarize_bart(
            text,
            max_length=length
        )

    elif model == "GPT-2":

        summary = summarize_gpt2(
            text,
            max_length=length
        )

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

    # =========================
    # MULTILINGUAL MODELS
    # =========================

    elif model == "XLSum-mT5":

        summary = summarize_xlsum_mt5(
            text,
            max_length=length
        )

    # elif model == "mBART-50":

    #     summary = summarize_mbart(
    #         text,
    #         max_length=length
    #     )

    elif model == "mT5-small":

        summary = summarize_mt5_small(
            text,
            max_length=length
        )

    else:

        summary = "Model not found."

    summary = summary.strip()

    if summary:

        summary = (
            summary[0].upper()
            + summary[1:]
        )

        # ENGLISH
        if language == "English":

            if not summary.endswith("."):
                summary += "."

        # MULTILINGUAL
        else:

            if not summary.endswith("।"):
                summary += "।"

    end = time.time()

    return summary, round(end - start, 2)


# =========================
# MULTILINGUAL SCORING
# =========================

def multilingual_quality_score(
    summary,
    original_text
):

    if not summary.strip():
        return 0

    original_words = len(
        original_text.split()
    )

    summary_words = len(
        summary.split()
    )

    if original_words == 0:
        return 0

    # COMPRESSION
    compression = (
        summary_words / original_words
    )

    if compression > 0.8:

        compression_score = 0.2

    elif compression > 0.6:

        compression_score = 0.5

    else:

        compression_score = (
            1.0 - compression
        )

    # REPETITION
    words = summary.split()

    unique_ratio = (
        len(set(words)) / len(words)
        if words else 0
    )

    # ENGLISH CONTAMINATION
    english_words = re.findall(
        r'[A-Za-z]+',
        summary
    )

    english_ratio = (
        len(english_words) / len(words)
        if words else 0
    )

    language_purity = (
        1 - english_ratio
    )

    final_score = (
        compression_score * 0.4
        +
        unique_ratio * 0.3
        +
        language_purity * 0.3
    )

    return round(final_score, 4)

# =========================
# CHANGE PERCENTAGE
# =========================

def calculate_change_percentage(
    original_text,
    summary_text
):

    original_words = len(
        original_text.split()
    )

    summary_words = len(
        summary_text.split()
    )

    if original_words == 0:
        return 0

    reduction = max(
        0,
        (
            (
                original_words - summary_words
            )
            /
            original_words
        ) * 100
    )

    return round(reduction, 2)


#=========================
#COMPARE ALL LOADING CARD
#=========================

def render_compare_loading_card(title, progress, statuses, pipeline_messages=None, times=None):

    status_html = ""

    message_html = ""

    if pipeline_messages is None:
        pipeline_messages = []

    if len(pipeline_messages) > 0:

        message_html += (
            '<div style="'
            'margin-top:22px;'
            'padding-top:18px;'
            'border-top:1px solid rgba(255,255,255,0.08);'
            '">'
        )

        for msg in pipeline_messages:

            message_html += (
                f'<div style="'
                f'margin-top:10px;'
                f'font-size:17px;'
                f'font-weight:500;'
                f'color:rgba(255,255,255,0.92);'
                f'">'
                f'{msg}'
                f'</div>'
            )

        message_html += "</div>"

    if times is None:
        times = {}

    for model, state in statuses.items():

        if state == "done":
            icon = "✅"

            row_style = (
                "font-weight:500;"
                "transform:translateX(0px);"
                "background:transparent;"
                "border-left:4px solid transparent;"
                "border-top:1px solid transparent;"
                "border-right:1px solid transparent;"
                "border-bottom:1px solid transparent;"
                "text-shadow:none;"
                "box-shadow:none;"
                "border-radius:8px;"
                "padding:6px 10px;"
            )

        elif state == "running":
            icon = "🔄"

            row_style = (
                "font-weight:700;"
                "transform:translateX(5px);"
                # "background:rgba(100,221,23,.05);"
                "background:rgba(76,175,80,.08);"

                "border-left:4px solid #4CAF50;"
                "border-top:1px solid rgba(255,255,255,.08);"
                "border-right:1px solid rgba(255,255,255,.08);"
                "border-bottom:1px solid rgba(255,255,255,.08);"

                "text-shadow:0 0 5px rgba(100,221,23,.18);"
                "box-shadow:0 0 16px rgba(76,175,80,.12);"
                "border-radius:8px;"
                "padding:6px 10px;"
            )

        else:
            icon = "⏳"

            row_style = (
                "opacity:0.72;"
                "font-weight:450;"
                "transform:translateX(0px);"
                "background:transparent;"
                "border-left:4px solid transparent;"
                "border-top:1px solid transparent;"
                "border-right:1px solid transparent;"
                "border-bottom:1px solid transparent;"
                "text-shadow:none;"
                "box-shadow:none;"
                "border-radius:8px;"
                "padding:6px 10px;"
            )

        time_text = ""

        if state == "done" and model in times:
            time_text = f"{times[model]:.2f} s"

        status_html += (
            f'<div style="display:flex;'
            f'justify-content:space-between;'
            f'align-items:center;'
            f'margin:8px 0;'
            f'font-size:17px;'

            f'transition:'
            f'transform .45s cubic-bezier(.22,.61,.36,1),'
            f'background-color .45s ease,'
            f'border-color .45s ease,'
            f'box-shadow .45s ease,'
            f'opacity .45s ease;'

            f'{row_style}">'
            f'<span>{icon} {model}</span>'
            f'<span style="opacity:.65;">{time_text}</span>'
            f'</div>'
        )

    return f"""
    <div style="
        padding:28px;
        border-radius:16px;
        background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.08);
        margin-top:10px;
        margin-bottom:20px;
    ">

    <div style="
        font-size:24px;
        font-weight:700;
        margin-bottom:22px;
    ">
        {title}
    </div>

    <div style="
        width:100%;
        height:12px;
        background:rgba(255,255,255,0.08);
        border-radius:20px;
        overflow:hidden;
        margin-bottom:24px;
    ">

    <div style="
        width:{progress}%;
        height:100%;
        background:linear-gradient(
            90deg,
            #00c853,
            #64dd17
        );
        border-radius:20px;
        transition:width .35s ease;
    ">
    </div>

    </div>


    {status_html}

    {message_html}

    </div>
    """


#==========================
#SINGLE MODEL LOADING CARD
#==========================

def render_single_loading_card(progress, stages,summary_time=None):

    status_html = ""

    for stage, state in stages.items():

        if state == "done":
            icon = "✅"

            row_style = (
                "font-weight:500;"
                "transform:translateX(0px);"
                "background:transparent;"
                "border-left:4px solid transparent;"
                "border-top:1px solid transparent;"
                "border-right:1px solid transparent;"
                "border-bottom:1px solid transparent;"
                "text-shadow:none;"
                "box-shadow:none;"
                "border-radius:8px;"
                "padding:6px 10px;"
            )

        elif state == "running":
            icon = "🔄"

            row_style = (
                "font-weight:700;"
                "transform:translateX(5px);"

                "background:rgba(76,175,80,.08);"

                "border-left:4px solid #4CAF50;"
                "border-top:1px solid rgba(255,255,255,.08);"
                "border-right:1px solid rgba(255,255,255,.08);"
                "border-bottom:1px solid rgba(255,255,255,.08);"

                "text-shadow:0 0 3px rgba(100,221,23,.18);"
                "box-shadow:0 0 16px rgba(76,175,80,.12);"

                "border-radius:8px;"
                "padding:6px 10px;"
            )

        else:
            icon = "⏳"
            
            row_style = (
                "opacity:0.72;"
                "font-weight:450;"
                "transform:translateX(0px);"
                "background:transparent;"
                "border-left:4px solid transparent;"
                "border-top:1px solid transparent;"
                "border-right:1px solid transparent;"
                "border-bottom:1px solid transparent;"
                "text-shadow:none;"
                "box-shadow:none;"
                "border-radius:8px;"
                "padding:6px 10px;"
            )

        time_text = ""

        if (
            stage == "Generating Summary"
            and state == "done"
            and summary_time is not None
        ):
            time_text = f"{summary_time:.2f} s"

        status_html += (
            f'<div style="display:flex;'
            f'justify-content:space-between;'
            f'align-items:center;'
            f'margin:8px 0;'
            f'font-size:17px;'

            f'transition:'
            f'transform .45s cubic-bezier(.22,.61,.36,1),'
            f'background-color .45s ease,'
            f'border-color .45s ease,'
            f'box-shadow .45s ease,'
            f'opacity .45s ease;'

            f'{row_style}">'
            f'<span>{icon} {stage}</span>'
            f'<span style="opacity:.65;">{time_text}</span>'
            f'</div>'
        )

    return f"""
    <div style="
        padding:28px;
        border-radius:16px;
        background:rgba(255,255,255,0.04);
        border:1px solid rgba(255,255,255,0.08);
        margin-top:10px;
        margin-bottom:20px;
    ">

    <div style="
        font-size:24px;
        font-weight:700;
        margin-bottom:22px;
    ">
        🧠 Multilingual Summarization Pipeline
    </div>

    <div style="
        width:100%;
        height:12px;
        background:rgba(255,255,255,0.08);
        border-radius:20px;
        overflow:hidden;
        margin-bottom:24px;
    ">

    <div style="
        width:{progress}%;
        height:100%;
        background:linear-gradient(
            90deg,
            #00c853,
            #64dd17
        );
        border-radius:20px;
        transition:width .35s ease;
    ">
    </div>

    </div>


    {status_html}

    </div>
    """


# =========================
# BUTTON
# =========================

button_text = (
    "🚀 Generate Summary"
    if mode == "Single Model"
    else
    "📊 Compare Models"
)

if st.button(button_text):

    if text.strip() == "":

        st.warning(
            "Please enter some text."
        )

    else:

        if mode == "Single Model":

            models_list = [
                selected_model
            ]

        else:

            models_list = available_models

        results = {}

        loading_placeholder = st.empty()

        statuses = {
            model: "waiting"
            for model in models_list
        }

        model_times = {}

        pipeline_messages = []

        if mode == "Single Model":

            single_stages = {

                "Initializing Model": "running",

                "Generating Summary": "waiting",

                "Calculating Evaluation Metrics": "waiting",

                "Finalizing Results": "waiting"

            }

            loading_placeholder.markdown(
                render_single_loading_card(
                    0,
                    single_stages
                ),
                unsafe_allow_html=True
            )

        else:

            loading_placeholder.markdown(
                render_compare_loading_card(
                    "🤖 Running AI Models",
                    0,
                    statuses,
                    times=model_times
                ),
                unsafe_allow_html=True
            )

        for current_index, m in enumerate(models_list):

            if mode == "Single Model":

                single_stages["Initializing Model"] = "done"
                single_stages["Generating Summary"] = "running"

                loading_placeholder.markdown(
                    render_single_loading_card(
                        15,
                        single_stages
                    ),
                    unsafe_allow_html=True
                )

            else:

                statuses[m] = "running"

                progress = int(
                    (current_index + 0.25) /
                    len(models_list) * 100
                )

                loading_placeholder.markdown(
                    render_compare_loading_card(
                        "🤖 Running AI Models",
                        progress,
                        statuses,
                        times=model_times
                    ),
                    unsafe_allow_html=True
                )

            try:

                summary, time_taken = (
                    run_model(
                        m,
                        text,
                        length
                    )
                )

                model_times[m] = time_taken

                if mode == "Single Model":

                    single_stages["Generating Summary"] = "done"

                    single_stages["Calculating Evaluation Metrics"] = "running"

                    loading_placeholder.markdown(
                        render_single_loading_card(
                            70,
                            single_stages,
                            summary_time=time_taken
                        ),
                        unsafe_allow_html=True
                    )

                word_count = len(
                    summary.split()
                )

                # ROUGE
                try:

                    rouge = (
                        calculate_rouge(
                            text,
                            summary
                        )
                    )

                except:

                    rouge = {
                        "ROUGE-1": 0,
                        "ROUGE-2": 0,
                        "ROUGE-L": 0
                    }

                # BLEU
                try:
                    
                    smooth = SmoothingFunction().method1

                    bleu = sentence_bleu(
                        [text.split()],
                        summary.split(),
                        smoothing_function=smooth
                    )

                except:

                    bleu = 0

                if mode == "Single Model":

                    single_stages["Calculating Evaluation Metrics"] = "done"

                    single_stages["Finalizing Results"] = "running"

                    loading_placeholder.markdown(
                        render_single_loading_card(
                            90,
                            single_stages,
                            summary_time=time_taken
                        ),
                        unsafe_allow_html=True
                    )

                results[m] = {

                    "summary": summary,

                    "rouge": rouge,

                    "bleu": round(
                        bleu,
                        4
                    ),

                    "time": time_taken,

                    "words": word_count
                }

                if mode == "Single Model":

                    single_stages["Finalizing Results"] = "done"

                    loading_placeholder.markdown(
                        render_single_loading_card(
                            100,
                            single_stages,
                            summary_time=time_taken
                        ),
                        unsafe_allow_html=True
                    )

                    time.sleep(2.5)  #Single-Model-Timeout for user to see the final loading card

                    loading_placeholder.empty()

            except Exception as e:

                results[m] = {

                    "summary":
                        (
                            "⚠ Unable to generate summary.\n\n"
                            f"Reason:\n{str(e)}"
                        ),

                    "rouge": {
                        "ROUGE-1": 0,
                        "ROUGE-2": 0,
                        "ROUGE-L": 0
                    },

                    "bleu": 0,

                    "time": 0,

                    "words": 0
                }


            # -------------------------
            # Update Loading Card (Done)
            # -------------------------

            if mode == "Compare All Models":

                statuses[m] = "done"

                progress = int(
                    (current_index + 1) /
                    len(models_list) * 100
                )

                loading_placeholder.markdown(
                    render_compare_loading_card(
                        "🤖 Running AI Models",
                        progress,
                        statuses,
                        pipeline_messages=pipeline_messages,
                        times=model_times
                    ),
                    unsafe_allow_html=True
                )

        # Remove loading card
        if mode == "Compare All Models":

            pipeline_messages.append(
                "🔄 Calculating Evaluation Metrics..."
            )

            loading_placeholder.markdown(
                render_compare_loading_card(
                    "🤖 Running AI Models",
                    100,
                    statuses,
                    pipeline_messages=pipeline_messages,
                    times=model_times
                ),
                unsafe_allow_html=True
            )

            time.sleep(0.8)

            pipeline_messages[-1] = "✅ Calculating Evaluation Metrics"

            pipeline_messages.append(
                "🔄 Ranking Models..."
            )

            loading_placeholder.markdown(
                render_compare_loading_card(
                    "🤖 Running AI Models",
                    100,
                    statuses,
                    pipeline_messages=pipeline_messages,
                    times=model_times
                ),
                unsafe_allow_html=True
            )

            time.sleep(0.8)

            pipeline_messages[-1] = "✅ Ranking Models"

            pipeline_messages.append(
                "🔄 Preparing Comparison Dashboard..."
            )

            loading_placeholder.markdown(
                render_compare_loading_card(
                    "🤖 Running AI Models",
                    100,
                    statuses,
                    pipeline_messages=pipeline_messages,
                    times=model_times
                ),
                unsafe_allow_html=True
            )

            time.sleep(1.0)

            pipeline_messages[-1] = "✅ Preparing Comparison Dashboard"

            loading_placeholder.markdown(
                render_compare_loading_card(
                    "🤖 Running AI Models",
                    100,
                    statuses,
                    pipeline_messages=pipeline_messages,
                    times=model_times
                ),
                unsafe_allow_html=True
            )

            time.sleep(2.5)   #All-Models-Timeout for user to see the final loading card

            loading_placeholder.empty()


        # ENGLISH
        
        if language == "English":

            scores = {}

            for m in models_list:

                rouge1 = results[m]["rouge"]["ROUGE-1"]
                rouge2 = results[m]["rouge"]["ROUGE-2"]
                rougeL = results[m]["rouge"]["ROUGE-L"]
                bleu = results[m]["bleu"]

                accuracy = (
                    0.4 * rouge1 +
                    0.2 * rouge2 +
                    0.2 * rougeL +
                    0.2 * bleu
                )

                scores[m] = round(accuracy, 4)

                change_percentage = calculate_change_percentage(
                    text,
                    results[m]["summary"]
                )

                if change_percentage < 10:

                    scores[m] = 0

        # MULTILINGUAL

        else:

            scores = {}

            for m in models_list:

                scores[m] = multilingual_quality_score(
                    results[m]["summary"],
                    text
                )

        best_model = max(
            scores,
            key=scores.get
        )

        st.session_state.compare_results = results

        st.session_state.scores = scores

        st.session_state.best_model = best_model

# =========================
# DISPLAY RESULTS
# =========================

if st.session_state.compare_results:

    results = (
        st.session_state.compare_results
    )

    scores = (
        st.session_state.scores
    )

    best_model = (
        st.session_state.best_model
    )

    models_list = list(results.keys())

    if mode == "Compare All Models":

        st.markdown("---")

        # BEST MODEL BOX

        st.markdown(f"""
        <div style="
            width:100%;
            padding:30px;
            border-radius:14px;
            background:linear-gradient(
                rgba(0,255,150,0.12),
                rgba(0,255,150,0.08)
            );
            border:1px solid rgba(0,255,150,0.5);
            text-align:center;
        ">
            <div style="
                font-size:30px;
                font-weight:700;
                color:#FFD700;
                margin-bottom:10px;
            ">
                🏆 Best Performing Model
            </div>

        <div style="
            font-size:40px;
            font-weight:600;
            margin-bottom:20px;
        ">
            {best_model}
        </div>

        <div style="
            font-size:35px;
            font-weight:500;
            line-height:0.5;
        ">
            {round(scores[best_model] * 100, 2)}%
        </div>

        <div style="
            font-size:20px;
            color:rgba(255,255,255,0.75);
            margin-top:8px;
            font-weight:600;
        ">
            Accuracy
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")


    change_percentages = {}

    # MODEL OUTPUTS
    for m in models_list:

        change_percentages[m] = calculate_change_percentage(
            text,
            results[m]["summary"]
        )

        if mode == "Single Model":
            st.markdown("---")

        with st.container(border=True):

            # Everything for one model goes here
            if mode == "Compare All Models" and m == best_model:
                st.markdown(f"""
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:10px;
                margin-left:10px;
            ">

            <div style="
                font-size:1.75rem;
                font-weight:600;
            ">
            🤖 {m}
            </div>

            <div style="
                font-size:1.5rem;
            ">
            🏆
            </div>

            </div>
            """, unsafe_allow_html=True)
                
            else:

                st.subheader(f"🤖 {m}")

            st.caption(f"""
                <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:10px;
                margin-top:-10px;
                margin-left:10px;
            ">

            <div style="
                font-size:1rem;
                font-weight:600;
            ">
            Model Type: {model_type[m]}
            </div>

            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="
                display:flex;
                align-items:center;
                margin-top:8px;
                margin-bottom:5px;
                margin-left:10px;
            ">

            <div style="
                font-size:1.6rem;
                font-weight:500;
            ">
                📄 Summary
            </div>

            </div>
            """, unsafe_allow_html=True)

            safe_summary = html.escape(results[m]["summary"])
            
            st.markdown(
                f"""
            <div style="
                padding:10px 20px;
                margin-top:5px;
                margin-bottom:10px;
                line-height:1.8;
                font-size:16px;
                text-align:left;
            ">
            {results[m]["summary"]}
            </div>
            """,
                unsafe_allow_html=True
            )

            if m == "GPT-2":

                st.markdown("""
            <div style="
                padding:16px 20px;
                margin-top:12px;
                margin-bottom:18px;
                border-radius:12px;
                margin-left:10px;
                margin-right:10px;
                background:rgba(255,193,7,0.08);
                border:1px solid rgba(255,193,7,0.35);
            ">

            <div style="
                font-size:1.1rem;
                font-weight:600;
                margin-bottom:10px;
            ">
            ⚠️ Expected Behaviour
            </div>

            <div style="
                line-height:1.7;
                font-size:15px;
            ">

            GPT-2 is a <b>general-purpose text generation model</b> and is <b>not fine-tuned for text summarization</b>. Unlike Pegasus, T5 and BART, it is designed to generate text rather than concise summaries. Therefore, it may reproduce parts of the original input, continue the text, or generate unrelated content. It is included in this project to demonstrate the limitations of using a non-specialized language model for summarization.

            </div>

            </div>
            """, unsafe_allow_html=True)

            #=============================
            # STATISTICS
            #=============================

            st.markdown(f"""
            <div style="
                display:flex;
                align-items:center;
                margin-top:8px;
                margin-bottom:5px;
                margin-left:10px;
            ">

            <div style="
                font-size:1.6rem;
                font-weight:500;
            ">
                📊 Statistics
            </div>

            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="
                display:flex;
                justify-content:left;
                gap:24px;
                margin-top:15px;
                margin-bottom:10px;
                margin-left:10px;
            ">

            <div style="
                padding:16px 22px;
                border-radius:12px;
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.10);
                text-align:center;
            ">

            <div style="
                font-size:2.1rem;
                font-weight:500;
                line-height:1;
            ">
                {results[m]["words"]}
            </div>

            <div style="
                margin-top:10px;
                font-size:0.95rem;
                color:gray;
                font-weight:500;
            ">
                📝 Word Count
            </div>

            </div>


            <div style="
                padding:16px 26px;
                border-radius:12px;
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.10);
                text-align:center;
            ">

            <div style="
                font-size:2.1rem;
                font-weight:500;
                line-height:1;
            ">
                {change_percentages[m]}%
            </div>

            <div style="
                margin-top:10px;
                font-size:0.95rem;
                color:gray;
                font-weight:500;
            ">
                🔄 Reduction
            </div>

            </div>


            <div style="
                padding:16px 26px;
                border-radius:12px;
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.10);
                text-align:center;
            ">

            <div style="
                font-size:2.1rem;
                font-weight:500;
                line-height:1;
            ">
                {results[m]["time"]:.2f}s
            </div>

            <div style="
                margin-top:10px;
                font-size:0.95rem;
                color:gray;
                font-weight:500;
            ">
                ⏱ Time Taken
            </div>

            </div>

            </div>
            """, unsafe_allow_html=True)

            st.write("")


            #=============================
            # EVALUATION METRICS
            #=============================

            st.markdown(f"""
            <div style="
                display:flex;
                align-items:center;
                margin-top:8px;
                margin-bottom:5px;
                margin-left:10px;
            ">

            <div style="
                font-size:1.6rem;
                font-weight:500;
            ">
                📈 Evaluation Metrics
            </div>

            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="
                display:flex;
                justify-content:left;
                gap:24px;
                margin-top:15px;
                margin-bottom:10px;
                margin-left:10px;
            ">

            <div style="
                padding:18px 28px;
                border-radius:12px;
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.10);
                text-align:center;
            ">

            <div style="
                font-size:2.1rem;
                font-weight:500;
                line-height:1;
            ">
                {round(results[m]["rouge"]["ROUGE-1"], 4)}
            </div>

            <div style="
                margin-top:10px;
                font-size:0.95rem;
                color:gray;
                font-weight:500;
            ">
                ROUGE-1
            </div>

            </div>


            <div style="
                padding:18px 28px;
                border-radius:12px;
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.10);
                text-align:center;
            ">

            <div style="
                font-size:2.1rem;
                font-weight:500;
                line-height:1;
            ">
                {round(results[m]["rouge"]["ROUGE-2"], 4)}
            </div>

            <div style="
                margin-top:10px;
                font-size:0.95rem;
                color:gray;
                font-weight:500;
            ">
                ROUGE-2
            </div>

            </div>


            <div style="
                padding:18px 28px;
                border-radius:12px;
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.10);
                text-align:center;
            ">

            <div style="
                font-size:2.1rem;
                font-weight:500;
                line-height:1;
            ">
                {round(results[m]["rouge"]["ROUGE-L"], 4)}
            </div>

            <div style="
                margin-top:10px;
                font-size:0.95rem;
                color:gray;
                font-weight:500;
            ">
                ROUGE-L
            </div>

            </div>


            <div style="
                padding:18px 28px;
                border-radius:12px;
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.10);
                text-align:center;
            ">

            <div style="
                font-size:2.1rem;
                font-weight:500;
                line-height:1;
            ">
                {round(results[m]["bleu"], 4)}
            </div>

            <div style="
                margin-top:10px;
                font-size:0.95rem;
                color:gray; 
                font-weight:500;
            ">
                BLEU
            </div>

            </div>

            </div>
            """, unsafe_allow_html=True)

            st.write("")


            # ==============================
            # Accuracy / Overall Evaluation
            # ==============================

            card_bg = "rgba(255,255,255,0.05)"
            card_border = "1px solid rgba(255,255,255,0.10)"

            if language == "English":

                st.markdown(f"""
            <div style="
                display:flex;
                align-items:center;
                margin-top:8px;
                margin-bottom:5px;
                margin-left:10px;
            ">

            <div style="
                font-size:1.6rem;
                font-weight:500;
            ">
                🎯 Accuracy
            </div>

            </div>
            """, unsafe_allow_html=True)

                st.markdown(f"""
            <div style="
                display:flex;
                justify-content:left;
                margin-top:15px;
                margin-left:10px;
            ">

            <div style="
                padding:18px 24px;
                border-radius:14px;
                background:{card_bg};
                border:{card_border};
                text-align:center;
            ">

            <div style="
                font-size:2.1rem;
                font-weight:500;
                line-height:1;
            ">
                {round(scores[m] * 100, 2)}%
            </div>

            <div style="
                margin-top:10px;
                font-size:1rem;
                color:gray;
                font-weight:500;
            ">
                Accuracy
            </div>

            </div>

            </div>
            """, unsafe_allow_html=True)

            else:

                st.markdown(f"""
            <div style="
                display:flex;
                align-items:center;
                margin-top:8px;
                margin-bottom:5px;
                margin-left:10px;
            ">

            <div style="
                font-size:1.6rem;
                font-weight:500;
            ">
                🎯 Overall Evaluation
            </div>

            </div>
            """, unsafe_allow_html=True)

                st.markdown(f"""
            <div style="
                display:flex;
                justify-content:left;
                gap:24px;
                margin-top:15px;
                margin-left:10px;
            ">

            <div style="
                padding:18px 24px;
                border-radius:14px;
                background:{card_bg};
                border:{card_border};
                text-align:center;
            ">

            <div style="
                font-size:2.1rem;
                font-weight:500;
                line-height:1;
            ">
                {round(scores[m] * 100, 2)}%
            </div>

            <div style="
                margin-top:10px;
                font-size:1rem;
                color:gray;
                font-weight:500;
            ">
                Accuracy
            </div>

            </div>


            <div style="
                padding:18px 24px;
                border-radius:14px;
                background:{card_bg};
                border:{card_border};
                text-align:center;
            ">

            <div style="
                font-size:2.1rem;
                font-weight:500;
                line-height:1;
            ">
                {scores[m]}
            </div>

            <div style="
                margin-top:10px;
                font-size:1rem;
                color:gray;
                font-weight:500;
            ">
                Multilingual Quality
            </div>

            </div>

            </div>
            """, unsafe_allow_html=True)


            #=============================
            # DOWNLOAD BUTTON
            #=============================
            
            file_content = (

                f"Model : {m}\n"
                f"Model Type : {model_type[m]}\n"
                f"Word Count : {results[m]['words']}\n\n"

                f"ROUGE-1 : "
                f"{results[m]['rouge']['ROUGE-1']}\n"

                f"ROUGE-2 : "
                f"{results[m]['rouge']['ROUGE-2']}\n"

                f"ROUGE-L : "
                f"{results[m]['rouge']['ROUGE-L']}\n"

                f"BLEU : "
                f"{results[m]['bleu']}\n\n"

                f"\nAccuracy : "
                f"{round(scores[m] * 100, 2)}%\n"

                f"Summary:\n\n"
                f"{results[m]['summary']}\n"
            )

            if language != "English":

                file_content += (

                    f"\nMultilingual Quality Score : "
                    f"{scores[m]}\n"

                    f"Accuracy : "
                    f"{round(scores[m] * 100, 2)}%\n"

                )

            st.divider()

            st.download_button(
                label=f"📥 Download {m} Summary",

                data=file_content,

                file_name=f"{m}_summary.txt",

                mime="text/plain"
            )

            st.write("")



    if mode == "Compare All Models":

        st.divider()

        # st.markdown("""
        # <div style="
        #     text-align:center;
        #     font-size:2.2rem;
        #     font-weight:600;
        #     margin-top:20px;
        #     margin-bottom:28px;
        #     background:rgba(127,127,127,.4);
        #     border:1px solid rgba(127,127,127,.08);
        #     border-radius:12px;
        #     padding:14px 18px;
        # ">
        #     📊 Comparison Dashboard
        # </div>
        # """, unsafe_allow_html=True)

        # st.divider()

        # =========================
        # PERFORMANCE %
        # =========================

        total_score = sum(scores.values())

        if total_score == 0:

            performance_percent = {
                m: 0
                for m in models_list
            }

        else:

            performance_percent = {

                m: round(
                    (
                        scores[m]
                        / total_score
                    ) * 100,
                    2
                )

                for m in models_list
            }

        sorted_models = sorted(
            performance_percent.items(),

            key=lambda x: x[1],

            reverse=True
        )

        st.markdown("""
        <div style="
            font-size:1.8rem;
            font-weight:600;
            margin-top:20px;
            margin-bottom:20px;
        ">
            🥯 Overall Model Performance
        </div>
        """, unsafe_allow_html=True)

        # for i, (m, score) in enumerate(
        #     sorted_models,
        #     start=1
        # ):

        #     st.markdown(
        #         f"**{i}. {m}** : {score}%"
        #     )

        # PIE CHART
        labels = [m for m, _ in sorted_models]

        sizes = [score for _, score in sorted_models]

        if sum(sizes) == 0:

            st.warning(
                "Performance chart unavailable because all model scores are 0."
            )

        else:

            fig, ax = plt.subplots()

            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.80,
                wedgeprops=dict(
                    width=0.42,
                    edgecolor="white"
                )
            )

            ax.text(
                0,
                0,
                "Overall\nPerformance",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold"
            )

            # ax.set_title(
            #     "Overall Model Performance"
            # )

            ax.axis("equal")
            st.pyplot(fig)

        st.markdown("---")

    # =========================
    # EVALUATION METRICS
    # =========================

    if mode == "Compare All Models":
        st.markdown("""
        <div style="
            font-size:1.8rem;
            font-weight:600;
            margin-top:20px;
            margin-bottom:20px;
        ">
            📋 Evaluation Metrics
        </div>
        """, unsafe_allow_html=True)


        evaluation_data = []

        for m in models_list:

            accuracy = round(
                scores[m] * 100,
                2
            )

            evaluation_data.append({

                "Model": m,

                "ROUGE-1":
                    results[m]["rouge"]["ROUGE-1"],

                "ROUGE-2":
                    results[m]["rouge"]["ROUGE-2"],

                "ROUGE-L":
                    results[m]["rouge"]["ROUGE-L"],

                "BLEU":
                    results[m]["bleu"],

                "Accuracy (%)":
                    accuracy
            })

        # TABLE
        df = pd.DataFrame(evaluation_data)

        # Sort by Accuracy (Highest first)
        df = df.sort_values(
            by="Accuracy (%)",
            ascending=False
        ).reset_index(drop=True)

        df.insert(
            0,
            "Rank",
            range(1, len(df) + 1)
        )

        # Display
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

    # ACCURACY COMPARISON CHART

    if mode == "Compare All Models":

        st.markdown("""
        <div style="
            font-size:1.8rem;
            font-weight:600;
            margin-top:20px;
            margin-bottom:20px;
        ">
            📊 Accuracy Comparison
        </div>
        """, unsafe_allow_html=True)

        # Sort by accuracy (Highest first)
        accuracy_df = df.sort_values(
            by="Accuracy (%)",
            ascending=True
        )

        spacing = 1.6
        bar_height = 1.2 

        positions = np.arange(len(accuracy_df)) * spacing
        fig2, ax2 = plt.subplots(figsize=(7, 5))

        bars= ax2.barh(
            positions,
            accuracy_df["Accuracy (%)"],
            height=bar_height,
        )

        ax2.set_xlim(0, max(accuracy_df["Accuracy (%)"]) + 3)
        
        # Display value beside each bar
        for bar, value in zip(bars, accuracy_df["Accuracy (%)"]):

            ax2.text(
                value + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}%",
                va="center",
                fontsize=9
            )

        ax2.set_yticks(positions)

        ax2.set_yticklabels(accuracy_df["Model"])
        
        ax2.set_xlabel("Accuracy (%)")

        ax2.set_title("Model Accuracy Comparison")

        ax2.grid(axis="x", linestyle="--", alpha=0.3)

        st.pyplot(fig2)

        st.markdown("---")


        # TEXT REDUCTION COMPARISON

        if mode == "Compare All Models":

            st.markdown("""
            <div style="
                font-size:1.8rem;
                font-weight:600;
                margin-top:20px;
                margin-bottom:20px;
            ">
                📉 Text Reduction Comparison
            </div>
            """, unsafe_allow_html=True)

            reduction_df = pd.DataFrame({

                "Model": list(change_percentages.keys()),

                "Reduction (%)": list(change_percentages.values())

            })

            # Sort by reduction (Highest first)
            reduction_df = reduction_df.sort_values(
                by="Reduction (%)",
                ascending=True
            )

            spacing = 1.6
            bar_height = 1.2

            positions = np.arange(len(reduction_df)) * spacing

            fig3, ax3 = plt.subplots(figsize=(7, 5))

            bars = ax3.barh(
                positions,
                reduction_df["Reduction (%)"],
                height=bar_height
            )

            ax3.set_xlim(0, max(reduction_df["Reduction (%)"]) + 15)

            # Display value beside each bar
            for bar, value in zip(bars, reduction_df["Reduction (%)"]):

                ax3.text(
                    value + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.2f}%",
                    va="center",
                    fontsize=8
                )

            ax3.set_yticks(positions)

            ax3.set_yticklabels(reduction_df["Model"])

            ax3.set_xlabel("Reduction (%)")

            ax3.set_title("Text Reduction Comparison")

            ax3.grid(
                axis="x",
                linestyle="--",
                alpha=0.3
            )

            st.pyplot(fig3)

            st.markdown("---")


    # DOWNLOAD ALL

    if mode == "Compare All Models":

        all_content = (
            f"🏆 BEST MODEL: "
            f"{best_model}\n\n"
        )

        for m in models_list:

            all_content += (

                f"========== {m} ==========\n"

                f"Model Type : "
                f"{model_type[m]}\n"

                f"Word Count : "
                f"{results[m]['words']}\n\n"

                f"ROUGE-1 : "
                f"{results[m]['rouge']['ROUGE-1']}\n"

                f"ROUGE-2 : "
                f"{results[m]['rouge']['ROUGE-2']}\n"

                f"ROUGE-L : "
                f"{results[m]['rouge']['ROUGE-L']}\n"

                f"BLEU : "
                f"{results[m]['bleu']}\n\n"

                f"\nAccuracy : "
                f"{round(scores[m] * 100, 2)}%\n"

                f"Summary:\n"
                f"{results[m]['summary']}\n\n"

                f"--------------------------------\n\n"
            )

            if language != "English":

                all_content += (
                    f"Multilingual Quality Score : "
                    f"{scores[m]}\n\n"

                    f"Accuracy : "
                    f"{round(scores[m] * 100, 2)}%\n"
                )

        st.download_button(
            label="📥 Download All Summaries",

            data=all_content,

            file_name="all_models_summary.txt",

            mime="text/plain"
        )