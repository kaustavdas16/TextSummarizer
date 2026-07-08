# =========================
# Models/xlsum_mt5_model.py
# =========================

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

import torch
import re

model_name = "csebuetnlp/mT5_multilingual_XLSum"

tokenizer = None
model = None

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_model():

    global tokenizer
    global model

    if tokenizer is None:

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

    if model is None:

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

        model.to(device)

        model.eval()


def clean_summary(summary):

    # Remove URL-like artifacts
    summary = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        summary
    )

    summary = re.sub(
        r'\S+\.(com|org|net)',
        '',
        summary
    )

    # Remove extra spaces
    summary = re.sub(
        r'\s+',
        ' ',
        summary
    )

    return summary.strip()


def summarize_xlsum_mt5(
    text,
    max_length=120
):

    load_model()

    inputs = tokenizer(

        text,

        return_tensors="pt",

        truncation=True,

        max_length=1024
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        summary_ids = model.generate(

            input_ids=inputs["input_ids"],

            attention_mask=inputs["attention_mask"],

            max_new_tokens=max_length,

            min_new_tokens=max_length // 2,

            num_beams=5,

            repetition_penalty=1.2,

            no_repeat_ngram_size=2,

            length_penalty=0.7,

            early_stopping=False
        )

    summary = tokenizer.decode(

        summary_ids[0],

        skip_special_tokens=True

    )

    summary = clean_summary(summary)

    return summary