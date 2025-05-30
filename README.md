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
