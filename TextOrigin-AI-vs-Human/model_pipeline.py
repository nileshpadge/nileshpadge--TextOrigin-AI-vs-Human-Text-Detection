# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import FunctionTransformer
# from sklearn.naive_bayes import MultinomialNB
# from feature_engineering import compute_metadata_features
# import streamlit as st

# @st.cache_resource
# def load_pipeline():
#     df = pd.read_csv("Data\AI_Human.csv")
#     df.columns = ['text', 'label']
#     df.dropna(subset=['text', 'label'], inplace=True)
#     df = compute_metadata_features(df)

#     X = df[['text', 'word_count', 'char_count', 'punctuation_ratio', 'has_hashtags', 'emotional_words', 'slang_words']]
#     y = df['label']
#     X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

#     pipeline = Pipeline([
#         ('features', FeatureUnion([
#             ('tfidf', Pipeline([
#                 ('selector', FunctionTransformer(lambda x: x['text'], validate=False)),
#                 ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english'))
#             ])),
#             ('word_count', Pipeline([
#                 ('selector', FunctionTransformer(lambda x: x[['word_count']], validate=False))
#             ])),
#             ('char_count', Pipeline([
#                 ('selector', FunctionTransformer(lambda x: x[['char_count']], validate=False))
#             ])),
#             ('punctuation_ratio', Pipeline([
#                 ('selector', FunctionTransformer(lambda x: x[['punctuation_ratio']], validate=False))
#             ])),
#             ('has_hashtags', Pipeline([
#                 ('selector', FunctionTransformer(lambda x: x[['has_hashtags']], validate=False))
#             ])),
#             ('emotional_words', Pipeline([
#                 ('selector', FunctionTransformer(lambda x: x[['emotional_words']], validate=False))
#             ])),
#             ('slang_words', Pipeline([
#                 ('selector', FunctionTransformer(lambda x: x[['slang_words']], validate=False))
#             ]))
#         ])),
#         ('classifier', MultinomialNB())
#     ])

#     pipeline.fit(X_train, y_train)
#     return pipeline


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from feature_engineering import compute_metadata_features
import streamlit as st

@st.cache_resource
def load_pipeline():
    # Load and clean data
    df = pd.read_csv("Data/AI_Human.csv", nrows=100000)
    df.columns = ['text', 'label']
    df = df.dropna().reset_index(drop=True)

    # Balance the dataset
    min_count = df['label'].value_counts().min()
    df_0 = df[df['label'] == 0].sample(min_count, random_state=42)
    df_1 = df[df['label'] == 1].sample(min_count, random_state=42)
    df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Feature engineering
    df = compute_metadata_features(df)

    # Define features and labels
    X = df[['text', 'word_count', 'char_count', 'punctuation_ratio', 
            'has_hashtags', 'emotional_words', 'slang_words']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', Pipeline([
                ('selector', FunctionTransformer(lambda x: x['text'], validate=False)),
                ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english'))
            ])),
            ('word_count', Pipeline([
                ('selector', FunctionTransformer(lambda x: x[['word_count']], validate=False))
            ])),
            ('char_count', Pipeline([
                ('selector', FunctionTransformer(lambda x: x[['char_count']], validate=False))
            ])),
            ('punctuation_ratio', Pipeline([
                ('selector', FunctionTransformer(lambda x: x[['punctuation_ratio']], validate=False))
            ])),
            ('has_hashtags', Pipeline([
                ('selector', FunctionTransformer(lambda x: x[['has_hashtags']], validate=False))
            ])),
            ('emotional_words', Pipeline([
                ('selector', FunctionTransformer(lambda x: x[['emotional_words']], validate=False))
            ])),
            ('slang_words', Pipeline([
                ('selector', FunctionTransformer(lambda x: x[['slang_words']], validate=False))
            ]))
        ])),
        ('classifier', MultinomialNB())
        # ('classifier', LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)
    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write("### Model Evaluation")
    st.write(f"**Accuracy:** {acc:.4f}")
    return pipeline