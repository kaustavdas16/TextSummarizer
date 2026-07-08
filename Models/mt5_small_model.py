from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer
)

import torch
import re

model_name = "google/mt5-small"

tokenizer = None
model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():

    global tokenizer
    global model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model is None:
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
        model.eval()


def detect_language(text):

    if re.search(r'[\u0980-\u09FF]', text):
        return "bn"

    elif re.search(r'[\u0900-\u097F]', text):
        return "hi"

    return "en"


def clean_summary(summary):

    # Remove "Tags:"
    summary = re.sub(r'Tags?:.*','',summary,flags=re.IGNORECASE  | re.DOTALL)

    # Remove "Read more"
    summary = re.sub(r'Read\s*more.*','',summary,flags=re.IGNORECASE)

    # Remove "Continue reading"
    summary = re.sub(r'Continue\s*reading.*','',summary,flags=re.IGNORECASE)

    summary = re.sub(r'<extra_id_\d+>', '', summary)
    summary = re.sub(r'https?://\S+|www\.\S+', '', summary)
    summary = re.sub(r'\S+\.com', '', summary)
    summary = re.sub(r'0x[A-Fa-f0-9]+', '', summary)
    summary = re.sub(r'\.{2,}', '.', summary)
    summary = re.sub(r'\s+', ' ', summary)

    return summary.strip()


def summarize_mt5_small(text, max_length=180):

    load_model()

    lang = detect_language(text)

    prompt = "summarize: " + text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():

        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],

            max_new_tokens=min(max_length, 150),
            min_new_tokens=50,

            num_beams=4,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            length_penalty=0.8,

            temperature=0.7,

            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return clean_summary(summary)