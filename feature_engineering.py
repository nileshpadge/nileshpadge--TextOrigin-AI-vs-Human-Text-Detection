import pandas as pd
import string
import re

# List of informal/emotional/human-like words
emotional_word_set = set([
    "amazing", "unreal", "love", "awesome", "beautiful", "fun", "exciting", "happy", "great", "cool"
])

# List of common slang words
slang_word_set = set([
    "yo", "bruh", "lit", "sick", "dope", "nah", "yolo", "omg", "lol", "btw", "tbh", "idk", "fomo", "chill"
])

def compute_metadata_features(df):
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['text'].apply(len)
    df['punctuation_ratio'] = df['text'].apply(lambda x: sum([1 for c in x if c in string.punctuation]) / (len(x) + 1))
    df['has_hashtags'] = df['text'].apply(lambda x: 1 if '#' in x else 0)
    df['emotional_words'] = df['text'].apply(lambda x: sum(1 for word in re.findall(r'\w+', x.lower()) if word in emotional_word_set))
    df['slang_words'] = df['text'].apply(lambda x: sum(1 for word in re.findall(r'\w+', x.lower()) if word in slang_word_set))
    return df