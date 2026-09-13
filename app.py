import re

import streamlit as st

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def preprocess_text(text: str) -> str:
    
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def compute_similarity(doc1: str, doc2: str, ngram_range = (1, 2)):

    vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = ngram_range)

    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])

    similarity_matrix = cosine_similarity(tfidf_matrix)
    score = similarity_matrix[0, 1]

    return score, vectorizer, tfidf_matrix



def get_top_similar_words(doc1: str, doc2: str, top_n: int = 10):

    word_vectorizer = TfidfVectorizer(stop_words = 'english', ngram_range = (1, 1))
    word_matrix = word_vectorizer.fit_transform([doc1, doc2])
    vocab = word_vectorizer.get_feature_names_out()

    row1 = word_matrix[0].toarray().flatten()
    row2 = word_matrix[1].toarray().flatten()

    shared_words = [
        (vocab[i], row1[i] * row2[i])
        for i in range(len(vocab))
        if row1[i] > 0 and row2[i] > 0
    ]

    shared_words.sort(key = lambda x: x[1], reverse = True)

    return shared_words[:top_n]



def classify_similarity(score: float) -> tuple[str, str]:

    if score > 0.85:
        return "Likely Duplicate", "red"
    elif score >= 0.5:
        return "Similar Content", "orange"
    else:
        return "Different Documents", "green"



def plot_similarity_gauge(score: float, color: str):

    fig, ax = plt.subplots(figsize = (6, 1))

    ax.barh([0], [1], color = "#e0e0e0")
    ax.barh([0], [score], color = color)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Cosine Similarity Score")
    ax.text(score, 0, f" {score:.2f}", va = "center",
            fontweight = "bold", color = "black")

    fig.tight_layout()

    return fig




st.set_page_config(page_title = "Document Similarity & Duplicate Detector", layout = "centered")
st.title("📄 Document Similarity & Duplicate Detector")
st.write("Upload two '.txt' files to check how similar they are and whether they might be duplicates.")

col1, col2 = st.columns(2)

with col1:
    file1 = st.file_uploader("Upload Document 1", type = ["txt"], key = "doc1")
with col2:
    file2 = st.file_uploader("Upload Document 2", type = ["txt"], key = "doc2")

use_bigrams = st.checkbox("Include bigrams (word pairs) in comparison?", value = True)

if file1 and file2:
    raw_text1 = file1.read().decode("utf-8", errors = "ignore")
    raw_text2 = file2.read().decode("utf-8", errors = "ignore")

    clean_text1 = preprocess_text(raw_text1)
    clean_text2 = preprocess_text(raw_text2)

    ngram_range = (1, 2) if use_bigrams else (1, 1)
    score, vectorizer, tfidf_matrix = compute_similarity(clean_text1, clean_text2, ngram_range)
    verdict, color = classify_similarity(score)

    st.subheader("Result")
    st.markdown(f"### Verdict: :{color}[{verdict}]")
    st.pyplot(plot_similarity_gauge(score, color))

    if verdict == "Likely Duplicate":
        st.subheader("🔎 Top 10 Most Similar Words")
        top_words = get_top_similar_words(clean_text1, clean_text2, top_n = 10)

        if top_words:
            words, weights = zip(*top_words)
            fig, ax = plt.subplots(figsize = (6, 3))
            ax.barh(words[::-1], weights[::-1], color = "crimson")
            ax.set_xlabel("Shared TF-IDF Weight")
            fig.tight_layout()
            st.pyplot(fig)
        else:
            st.write("No overlapping words found (unexpected for a flagged duplicate).")


    with st.expander("View preprocessed text"):
        st.write("**Document 1 (cleaned):**")
        st.text(clean_text1[:1000] + ("..." if len(clean_text1) > 1000 else ""))
        st.write("**Document 2 (cleaned):**")
        st.text(clean_text2[:1000] + ("..." if len(clean_text2) > 1000 else ""))

    with st.expander("View top shared terms"):
        feature_names = vectorizer.get_feature_names_out()
        row1 = tfidf_matrix[0].toarray().flatten()
        row2 = tfidf_matrix[1].toarray().flatten()

        shared_terms = [
            (feature_names[i], row1[i] * row2[i])
            for i in range(len(feature_names))
            if row1[i] > 0 and row2[i] > 0
        ]

        shared_terms.sort(key = lambda x: x[1], reverse = True)

        if shared_terms:
            st.write("Terms contributing mose to the similarity score:")
            for term, weight in shared_terms[:15]:
                st.write(f"- **{term}** (weight: {weight:.4f})")
        else:
            st.write("No shared terms found between the two documents.")
else:
    st.info("Please upload both files for comparison.")
