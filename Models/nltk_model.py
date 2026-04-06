import nltk
from nltk.tokenize import sent_tokenize

# Auto download (runs only once)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def summarize_nltk(text):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:3])