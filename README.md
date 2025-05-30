## TextOrigin-AI-vs-Human-Text-Detection ##

TextOrigin ML Pipeline — Visual Overview

               ┌─────────────────────┐
               │   Input DataFrame   │
               │  ─────────────────  │
               │  text, word_count,  │
               │  char_count, etc.   │
               └────────┬────────────┘
                        │
                        ▼
          ┌────────────────────────────┐
          │     FeatureUnion Block     │
          │ Combines text + metadata   │
          └────────┬────────────┬──────┘
                   │            │
     ┌─────────────┘            └────────────────────────────────────┐
     ▼                                                              ▼
┌──────────────┐                                          ┌────────────────────┐
│ TF-IDF Text  │                                          │  Metadata Features │
│ ──────────── │                                          │ word_count,        │
│ FunctionTransformer: text →                            │ char_count,        │
│  TfidfVectorizer (5000)                                │ punctuation_ratio, │
└──────┬────────┘                                          │ has_hashtags,     │
       │                                                  │ slang_words, etc. │
       ▼                                                  └────────┬──────────┘
       └─────────────────────┬─────────────────────────────────────┘
                             ▼
               ┌────────────────────────────┐
               │ Concatenated Feature Vector│
               └────────┬───────────────────┘
                        ▼
              ┌─────────────────────────────┐
              │  Multinomial Naive Bayes    │
              └────────┬────────────────────┘
                        ▼
           ┌──────────────────────────────┐
           │ Prediction: AI or Human Text │
           └──────────────────────────────┘


from pathlib import Path

# Define the README content
readme_content = """
# TextOrigin: AI vs Human Text Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Enabled-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🧠 Overview

**TextOrigin** is a hybrid AI vs Human text detection system built with:
- **Machine Learning (Naive Bayes + TF-IDF)**
- **Metadata features (e.g., word count, sentence count, etc.)**
- **Groq’s LLaMA3 LLM for deep text understanding**
- **Explainability (LIME / SHAP ready)**

## 📦 Features

| Component        | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `Naive Bayes`    | Classifies text using TF-IDF + metadata features                            |
| `TF-IDF`         | Transforms input into numerical features for ML                             |
| `LSTM (Optional)`| For deep learning-based classification (coming soon)                        |
| `Groq LLM`       | Provides rationale, analysis, and recommendations based on LLaMA3           |
| `Gradio / Streamlit` | Interactive interface for upload, detection, and export                 |
| `CSV Export`     | Save predictions, confidence, and rationale in a downloadable format         |
| `PDF Upload`     | Allows multi-text file input and comparisons                                 |

## 🚀 How It Works

1. **Upload text or PDF**
2. **Run feature extraction (metadata + TF-IDF)**
3. **Classify with Naive Bayes**
4. **(Optional) Analyze with LLM (Groq API key needed)**
5. **Explain prediction and export results**

## 🛠 Installation

```bash
git clone https://github.com/yourusername/textorigin.git
cd textorigin
pip install -r requirements.txt








