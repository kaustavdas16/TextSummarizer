from transformers import pipeline
import streamlit as st


@st.cache_resource
def load_model():

    return pipeline(
        "text-generation",
        model="gpt2"
    )


def summarize_gpt2(text, max_length=100):

    generator = load_model()

    # Limit the input because GPT-2 has a limited context window
    # max_input_chars = 1800
    # truncated_text = text[:max_input_chars]

    inputs = generator.tokenizer(
        text,
        truncation=True,
        max_length=850,   # Leave room for the prompt and generated output
        return_tensors="pt"
    )

    truncated_text = generator.tokenizer.decode(
        inputs["input_ids"][0],
        skip_special_tokens=True
    )

    prompt = (
        "Summarize the following text:\n\n"
        + truncated_text +
        "\n\nSummary:"
    )

    output = generator(
        prompt,
        max_new_tokens=max_length,
        do_sample=False,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        truncation=True,
        pad_token_id=50256,
        num_return_sequences=1,
        return_full_text=False
    )

    summary = output[0]["generated_text"].strip()

    if not summary:
        return truncated_text

    return summary