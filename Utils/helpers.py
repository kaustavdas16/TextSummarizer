from rouge_score import rouge_scorer

# ✅ ROUGE Score Calculation
def calculate_rouge(original, summary):
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(original, summary)

    return {
        "ROUGE-1": round(scores['rouge1'].fmeasure, 3),
        "ROUGE-2": round(scores['rouge2'].fmeasure, 3),
        "ROUGE-L": round(scores['rougeL'].fmeasure, 3)
    }


# ✅ Split text into chunks (for large input handling)
def split_text(text, max_words=200):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)

    return chunks


# ✅ Optional: Limit text length (safety)
def limit_text(text, max_words=1000):
    words = text.split()

    if len(words) > max_words:
        return " ".join(words[:max_words])
    
    return text