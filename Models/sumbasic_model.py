from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer

def summarize_sumbasic(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = SumBasicSummarizer()

    summary = summarizer(parser.document, 3)
    return " ".join(str(sentence) for sentence in summary)