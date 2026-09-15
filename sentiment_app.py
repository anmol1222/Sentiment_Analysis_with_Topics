import streamlit as st
import numpy as np
import pandas as pd
import joblib
import re
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# 1. Page Configuration
st.set_page_config(
    page_title="Sentiment & Topic Intelligence Engine",
    page_icon="🔍",
    layout="wide"
)

# 2. Custom CSS for Animations & UI Styling
st.markdown("""
<style>
    @keyframes slideIn {
        from { opacity: 0; transform: translateY(15px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-box {
        animation: slideIn 0.6s ease-out;
        padding: 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .positive-card {
        background-color: #d4edda;
        border-left: 6px solid #28a745;
        color: #155724;
    }
    .negative-card {
        background-color: #f8d7da;
        border-left: 6px solid #dc3545;
        color: #721c24;
    }
    .keyword-pill {
        display: inline-block;
        background-color: #f0f2f6;
        color: #31333f;
        padding: 4px 12px;
        border-radius: 16px;
        margin: 3px;
        font-weight: 500;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# 3. Cached Model & Vectorizer Loader
@st.cache_resource
def load_artifacts():
    svm = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return svm, vectorizer

try:
    model, tfidf = load_artifacts()
except Exception:
    st.error("Error: 'sentiment_model.pkl' ya 'tfidf_vectorizer.pkl' directory me nahi mile. Pehle models save karein.")
    st.stop()

# 4. Text Preprocessing Helper
def clean_input(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()

# 5. Application Header
st.title("Sentiment Analyzer & Topic Extraction Engine")
st.caption("Linear SVM inference combined with PCA-projected K-Means topic clusters.")

# 6. Main Interface Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("Input Review")
    user_review = st.text_area(
        "Enter customer review text below:",
        placeholder="e.g., The battery backup is absolutely fantastic and screen quality is crisp...",
        height=180
    )
    analyze_btn = st.button("Run Prediction & Cluster Analysis", type="primary", use_container_width=True)

    if analyze_btn:
        if not user_review.strip():
            st.warning("Analysis run karne ke liye text enter karein.")
        else:
            with st.spinner("Processing text and mapping semantic clusters..."):
                # Inference
                cleaned = clean_input(user_review)
                vec = tfidf.transform([cleaned])
                pred = model.predict(vec)[0]
                
                # Decision distance as confidence proxy
                decision_val = model.decision_function(vec)[0] if hasattr(model, "decision_function") else 1.0
                confidence = min(99.0, max(52.0, 50 + abs(decision_val) * 20))

                # Extract top non-zero TF-IDF keywords
                feature_names = tfidf.get_feature_names_out()
                cx = vec.tocoo()
                word_weights = sorted(zip(cx.col, cx.data), key=lambda x: x[1], reverse=True)
                top_words = [feature_names[idx] for idx, _ in word_weights[:6]]

            # Animated Result Card
            is_pos = (pred == 1 or str(pred).lower() == 'positive')
            card_class = "positive-card" if is_pos else "negative-card"
            label = "Positive Sentiment" if is_pos else "Negative Sentiment"

            st.markdown(f"""
            <div class="animate-box {card_class}">
                <h3 style="margin:0; padding-bottom: 6px;">{label}</h3>
                <p style="margin:0;">Model Confidence: <b>{confidence:.2f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            if is_pos:
                st.balloons()

            st.write("**Top Salient Keywords:**")
            if top_words:
                pills = " ".join([f'<span class="keyword-pill">{w}</span>' for w in top_words])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.caption("No vocabulary terms matched.")

with col2:
    st.subheader("Topic Cluster Space")
    
    # Generate PCA-projected 2D scatter space with K-Means
    @st.cache_data
    def generate_cluster_visualization():
        # Top 120 vocabulary features projection
        vocab_terms = list(tfidf.vocabulary_.keys())[:120]
        sub_vecs = tfidf.transform(vocab_terms).toarray()

        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(sub_vecs)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        clusters = kmeans.fit_predict(sub_vecs)

        df_vis = pd.DataFrame({
            'PC1': coords[:, 0],
            'PC2': coords[:, 1],
            'Term': vocab_terms,
            'Cluster': [f"Topic {c + 1}" for c in clusters]
        })
        return df_vis

    df_clusters = generate_cluster_visualization()

    fig = px.scatter(
        df_clusters,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_name='Term',
        title="Interactive 2D Semantic Space (PCA + K-Means)",
        color_discrete_sequence=['#4C78A8', '#F58518', '#E45756'],
        template="plotly_white",
        height=420
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

    st.plotly_chart(fig, use_container_width=True)