
import streamlit as st
import pickle
import os
import numpy as np
from preprocess import TextPreprocessor

st.set_page_config(page_title="Fake News Core Portal", page_icon="📰", layout="centered")

@st.cache_resource
def load_assets():
    with open('saved_models/model.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('saved_models/vectorizer.pkl', 'rb') as v_file:
        vectorizer = pickle.load(v_file)
    return model, vectorizer

st.title("📰 Fake News Verification Dashboard")
st.markdown("Analyze textual linguistic metrics with advanced probability confidence output vectors.")

try:
    model, vectorizer = load_assets()
    preprocessor = TextPreprocessor()
    
    user_input = st.text_area(
        "Paste the complete article or news copy sequence below:", 
        placeholder="Input content context body text here...",
        height=250
    )
    
    if st.button("Analyze News Integrity", type="primary"):
        if not user_input.strip():
            st.warning("Analysis triggered with blank parameters. Provide target text.")
        else:
            with st.spinner("Calculating probability dimensions..."):
                cleaned = preprocessor.clean(user_input)
                vectorized = vectorizer.transform([cleaned])
                
                # Fetch class targets and probability arrays
                prediction = model.predict(vectorized)[0]
                probabilities = model.predict_proba(vectorized)[0]
                class_labels = model.classes_
                
                # Map accurate tracking indices
                target_idx = np.where(class_labels == prediction)[0][0]
                confidence_score = probabilities[target_idx] * 100
                
                st.subheader("System Verdict:")
                if prediction.lower() == 'real':
                    st.success(f"✅ **Linguistic markers match AUTHENTIC source news.** (Confidence: {confidence_score:.2f}%)")
                    st.info("💡 Language profiling displays low sensationalism, structured punctuation, and standard vocabulary context distribution patterns.")
                else:
                    st.error(f"🚨 **Warning: High probability of FAKE / FABRICATED text patterns.** (Confidence: {confidence_score:.2f}%)")
                    st.warning("⚠️ Anomalies spotted: Context contains unverified bias distributions, sensational terminology clusters, or missing structural background variables.")
                    
except FileNotFoundError:
    st.error("⚠️ System Weights Array Missing! Execute `python train.py` inside your root project terminal panel first.")
