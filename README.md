# 🧠 Text Summarization Studio

A Generative AI Project that implements multiple NLP and Transformer-based models to perform text summarization and compare their outputs using ROUGE evaluation.

---

## 🚀 Features

- 🔹 Multiple Summarization Models:
  - Pegasus
  - T5
  - BART
  - BERT (simulated)
  - GPT-2
  - LSA
  - TextRank
  - LexRank
  - SumBasic
  - NLTK

- 🔹 Clean and Modern UI using Streamlit
- 🔹 ROUGE Score Evaluation (ROUGE-1, ROUGE-2, ROUGE-L)
- 🔹 Handles long text using chunking
- 🔹 Model comparison capability

---

## 🧠 Tech Stack

- Python
- Streamlit
- Hugging Face Transformers
- NLTK
- Sumy

---

## 📂 Project Structure

TextSummarizer/
│
├── app.py
├── requirements.txt
│
├── Models/
│ ├── pegasus_model.py
│ ├── t5_model.py
│ ├── bart_model.py
│ ├── bert_model.py
│ ├── gpt2_model.py
│ ├── lsa_model.py
│ ├── textrank_model.py
│ ├── lexrank_model.py
│ ├── sumbasic_model.py
│ ├── nltk_model.py
│
└── Utils/
└── helpers.py

---

## ⚙️ How to Run the Project (Step-by-Step)

### 🔹 Step 1: Clone the Repository

Open terminal and run:

git clone https://github.com/kaustavdas16/TextSummarizer.git

cd TextSummarizer

🔹 Step 2: Create Virtual Environment

python -m venv venv

🔹 Step 3: Activate Virtual Environment
On Windows (PowerShell):

venv\Scripts\activate

On Windows (Command Prompt):

venv\Scripts\activate.bat

🔹 Step 4: Install Dependencies

pip install -r requirements.txt

🔹 Step 5: Run the Application

streamlit run app.py



🔹 If Error Comes

pip install transformers==4.37.2