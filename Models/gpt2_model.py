from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def summarize_gpt2(text):
    result = generator("Summarize: " + text, max_length=100)
    return result[0]['generated_text']