import streamlit as st

import time #newly added

from transformers import PegasusTokenizer, PegasusForConditionalGeneration

@st.cache_resource
def load_model():

    start = time.time()   #newly added

    print(">>> LOADING PEGASUS MODEL <<<")   #newly added

    tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

    print(f"Tokenizer loaded in {time.time()-start:.2f} sec")       #newly added
    start = time.time()

    model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

    print(f"Model loaded in {time.time()-start:.2f} sec")   #newly added

    return tokenizer, model

def split_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def summarize_pegasus(text, max_length=100):
    tokenizer, model = load_model()

    chunks = split_text(text)
    summaries = []

    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True)

        start = time.time()     #newly added


        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            min_length=max_length // 3,
            num_beams=4
        )

        print(f"Generation took {time.time() - start:.2f} sec")    #newly added

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)