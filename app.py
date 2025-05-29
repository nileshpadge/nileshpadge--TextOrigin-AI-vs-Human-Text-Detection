# import streamlit as st
# import pandas as pd
# import PyPDF2
# from feature_engineering import compute_metadata_features, emotional_word_set, slang_word_set
# from model_pipeline import load_pipeline
# from llm_analysis import generate_llm_analysis_groq
# import re
# import string
# # Streamlit App UI
# st.set_page_config(page_title="TextOrigin: AI vs Human Detector", layout="centered")
# st.title("🧠 TextOrigin: AI vs Human Detector")
# st.markdown("Upload a PDF or enter text below. We'll detect if it's AI-generated or human-written and explain why.")

# # PDF Upload
# uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
# user_input = st.text_area("✍ Or paste your text here:", height=200)

# # Process input (PDF or text)
# input_text = ""
# if uploaded_file is not None:
#     try:
#         # Extract text from PDF
#         pdf_reader = PyPDF2.PdfReader(uploaded_file)
#         input_text = ""
#         for page in pdf_reader.pages:
#             input_text += page.extract_text() or ""
#         st.success("PDF uploaded and text extracted successfully!")
#     except Exception as e:
#         st.error(f"Error processing PDF: {e}")
# elif user_input.strip():
#     input_text = user_input

# # Analyze button
# if st.button("Analyze Text"):
#     if not input_text.strip():
#         st.warning("Please upload a PDF or enter some text to analyze.")
#     else:
#         pipeline = load_pipeline()

#         # Prepare input DataFrame
#         input_df = pd.DataFrame([{
#             'text': input_text,
#             'word_count': len(input_text.split()),
#             'char_count': len(input_text),
#             'punctuation_ratio': sum([1 for c in input_text if c in string.punctuation]) / (len(input_text) + 1),
#             'has_hashtags': 1 if '#' in input_text else 0,
#             'emotional_words': sum(1 for word in re.findall(r'\w+', input_text.lower()) if word in emotional_word_set),
#             'slang_words': sum(1 for word in re.findall(r'\w+', input_text.lower()) if word in slang_word_set)
#         }])

#         # Predict and display results
#         pred_label = pipeline.predict(input_df)[0]
#         pred_proba = pipeline.predict_proba(input_df)[0]
#         confidence = round(max(pred_proba) * 100, 2)

#         label_map = {0: "Human-written", 1: "AI-generated"}
#         label = label_map[int(pred_label)]

#         st.subheader("🔍 Classification Result")
#         st.success(f"*Prediction:* {label}  \n*Confidence:* {confidence}%")

#         # Get LLM analysis
#         with st.spinner("💬 Getting detailed analysis from LLM..."):
#             llm_output = generate_llm_analysis_groq(input_text, label, confidence)
#             st.markdown("### 📊 LLM Analysis")
#             st.markdown(llm_output)


import streamlit as st
import pandas as pd
import PyPDF2
from feature_engineering import compute_metadata_features, emotional_word_set, slang_word_set
from model_pipeline import load_pipeline
from llm_analysis import generate_llm_analysis_groq
import re
import string

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Streamlit App UI
st.set_page_config(page_title="TextOrigin: AI vs Human Detector", layout="centered")
st.title("🧠 TextOrigin: AI vs Human Detector")
st.markdown("Upload a PDF or enter text below. We'll detect if it's AI-generated or human-written and explain why.")

# PDF Upload
uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])
user_input = st.text_area("✍ Or paste your text here:", height=200)

# Process input (PDF or text)
input_text = ""
if uploaded_file is not None:
    try:
        # Extract text from PDF
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        input_text = ""
        for page in pdf_reader.pages:
            input_text += page.extract_text() or ""
        st.success("PDF uploaded and text extracted successfully!")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
elif user_input.strip():
    input_text = user_input

# Analyze button
if st.button("Analyze Text"):
    if not input_text.strip():
        st.warning("Please upload a PDF or enter some text to analyze.")
    else:
        pipeline = load_pipeline()

        # Prepare input DataFrame
        input_df = pd.DataFrame([{
            'text': input_text,
            'word_count': len(input_text.split()),
            'char_count': len(input_text),
            'punctuation_ratio': sum([1 for c in input_text if c in string.punctuation]) / (len(input_text) + 1),
            'has_hashtags': 1 if '#' in input_text else 0,
            'emotional_words': sum(1 for word in re.findall(r'\w+', input_text.lower()) if word in emotional_word_set),
            'slang_words': sum(1 for word in re.findall(r'\w+', input_text.lower()) if word in slang_word_set)
        }])

        # Predict and display results
        pred_label = pipeline.predict(input_df)[0]
        pred_proba = pipeline.predict_proba(input_df)[0]
        confidence = round(max(pred_proba) * 100, 2)

        label_map = {0: "Human-written", 1: "AI-generated"}
        label = label_map[int(pred_label)]

        st.subheader("🔍 Classification Result")
        st.success(f"*Prediction:* {label}  \n*Confidence:* {confidence}%")

        # Get LLM analysis
        with st.spinner("💬 Getting detailed analysis from LLM..."):
            llm_output = generate_llm_analysis_groq(input_text, label, confidence)
            # st.markdown("### 📊 LLM Analysis")
            st.markdown(llm_output)

        # Save to chat history
        st.session_state.chat_history.append({
            'input_text': input_text[:500] + "..." if len(input_text) > 500 else input_text,  # Truncate for display
            'label': label,
            'confidence': confidence,
            'llm_analysis': llm_output
        })

# Display chat history
st.subheader("📜 Chat History")
if st.session_state.chat_history:
    for idx, history in enumerate(reversed(st.session_state.chat_history)):  # Most recent first
        with st.expander(f"Analysis {len(st.session_state.chat_history) - idx}"):
            st.markdown(f"**Input Text**: {history['input_text']}")
            st.markdown(f"**Prediction**: {history['label']} ({history['confidence']}% confidence)")
            st.markdown(f"**LLM Analysis**:")
            st.markdown(history['llm_analysis'])
else:
    st.write("No analysis history yet.")