from groq import Groq
import os

os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

def generate_llm_analysis_groq(text, label, confidence):
    prompt = f"""
You are an expert at analyzing the origin of a given text. A machine learning classifier has labeled the text as \"{label}\" with {confidence}% confidence.

Analyze and respond with the following:

1. *Text Origin Classification*:
   - Origin: {label}, {confidence}% confidence.
   - Rationale: Explain why it is classified this way.

2. *Feature Analysis*:
   - Repetition (0–100): Score + explanation.
   - Errors (0–100): Score + explanation.
   - Creativity (0–100): Score + explanation.
   - Example: Quote 1 sentence from the text to justify.

3. **Output (Recommendation & Alert)**:
- Suggestions:
  - Rewrite:  the most AI-looking sentence with more natural or original language.

Text:
{text}

Respond in clean markdown format.
"""
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content