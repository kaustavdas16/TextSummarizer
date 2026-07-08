# =========================
# Models/xlsum_finetuned_model.py
# =========================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Path to the fine-tuned model
MODEL_PATH = "Models/XLSum_fine_tuned_model"

tokenizer = None
model = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    global tokenizer, model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if model is None:
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()


def summarize_xlsum_finetuned(text, max_length=180):

    load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],

            max_new_tokens=min(
                max_length,
                150
            ),

            min_new_tokens=min(
                max_length // 2,
                60
            ),

            num_beams=6,

            repetition_penalty=2.0,

            no_repeat_ngram_size=3,

            length_penalty=1.5,

            early_stopping=True
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    return summary.strip()