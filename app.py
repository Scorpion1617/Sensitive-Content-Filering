import streamlit as st
import joblib
import re

# Load trained model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# List of known offensive terms
OFFENSIVE_TERMS = ['bitch', 'shit', 'hoe', 'fuck', 'damn', 'tranny', 'slut', 'nigga']

# Function to censor offensive words
def censor_text(text, terms):
    pattern = r'\b(' + '|'.join(re.escape(term) for term in terms) + r')\b'
    return re.sub(pattern, lambda m: '*' * len(m.group()), text, flags=re.IGNORECASE)

# Hybrid classifier: ML + offensive word check
def classify_and_censor(text):
    vec = vectorizer.transform([text])
    label = int(model.predict(vec)[0])

    censored = censor_text(text, OFFENSIVE_TERMS)
    keyword_hit = censored != text

    if keyword_hit:
        if label in [0, 1]:
            final_label = label
            final_text = censored
        else:
            final_label = 1
            final_text = censored
    else:
        final_label = 2
        final_text = text

    label_map = {
        0: "🚫 Hate Speech",
        1: "⚠️ Offensive Language",
        2: "✅ Clean"
    }
    return final_label, label_map[final_label], final_text

# Streamlit UI
st.set_page_config("Tweet Classifier", layout="centered")
st.title("🛡️ Hate & Offensive Speech Detector")

tweet = st.text_area("📩 Enter any text:")

if st.button("Analyze"):
    if tweet.strip():
        label, label_text, censored = classify_and_censor(tweet)
        st.subheader(f"Prediction: {label_text}")

        if label == 0 or label == 1:
            st.error("⚠️ Sensitive content found. Here is the censored version:")
        elif label == 2:
            st.success("✅ Your content is clean!")
        else:
            st.warning("🤔 Unexpected label. Please check the model.")

        st.markdown("### Output:")
        st.write(censored)
    else:
        st.warning("Please enter some text.")
