# Multilingual Translator

# 🌐 Neural Machine Translation for Indian Languages

## 📌 Project Overview

This project implements a **Neural Machine Translation (NMT)** system that provides real-time translation between multiple Indian languages using a web-based interface. The model supports translations among **Hindi, English, Marathi, Tamil, and Kannada**, leveraging the **IndicTrans2** transformer model fine-tuned on the 
dataset refer [Indian Parallel Corpus](https://github.com/Kartikaggarwal98/Indian_ParallelCorpus).

---

## 🧩 Features

- ✅ Web-based UI for instant translation
- ✅ Translation support for Hindi, English, Marathi, Tamil, and Kannada
- ✅ IndicTrans2-based transformer model fine-tuned on Indian language pairs
- ✅ Language detection and transliteration for Romanized inputs (e.g., "namaste" → "नमस्ते")
- ✅ Evaluation using BLEU, METEOR, and TER scores
- ✅ Comparison against Google Translate outputs

---

## 🚀 Technologies Used

| Component     | Tool / Library                                               |
|---------------|--------------------------------------------------------------|
| Model         | `ai4bharat/IndicTrans2` (Transformers)  |                    |
| Web Framework | Fast API                                |                    |
| Tokenizer     | Hugging Face Tokenizers                 |                    |
| Evaluation    | SacreBLEU, NLTK (METEOR), PyTER         |                    |
| Dataset       | Indian Parallel Corpus (IIT Bombay), ai4bharat/samanantar"   |
|               |                                         |                    |

---

## 🖥️ How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Chandrashekhar-26/multilingual-translator.git
```

### 2. Set Up the Environment
```bash
cd multilingual-translator
python -m venv venv
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python main.py
```

## Test cases include:

✅ Valid input text across Hindi, English, Marathi, Tamil, Kannada

✅ Romanized inputs (e.g., "vanakkam", "namaste")

✅ Empty text input handling

✅ Unsupported language pair error handling

✅ Long text translations

✅ Non-translatable symbols/emojis

## 📚 Supported Languages and Pairs
| Source Language | Target Languages                 |
| --------------- |----------------------------------|
| Hindi           | English, Marathi, Tamil, Kannada |
| English         | Hindi, Marathi, Tamil, Kannada   |
| Marathi         | Hindi, English, Tamil, Kannada   |
| Tamil           | Kannada, Hindi, Marathi, English |
| Kannada         | Tamil, Hindi, Tamil, English     |


## 🏛️ Dataset Used (For Finetuning)
Dataset referred - [Indian Parallel Corpus](https://github.com/Kartikaggarwal98/Indian_ParallelCorpus).
1. Hindi → English = cfilt/iitb-english-hindi (Hugging face)
2. English → Hindi = cfilt/iitb-english-hindi (Hugging face)
3. English → Marathi = ai4bharat/samanantar (Hugging face)

