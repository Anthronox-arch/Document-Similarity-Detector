# 📄 Document Similarity & Duplicate Detector

A lightweight **Streamlit** app that compares two `.txt` documents and tells you how similar they are — and whether one might be a duplicate of the other — using **TF-IDF vectorization** and **cosine similarity**.

---

## ✨ Features

- 📤 **Upload two `.txt` files** side by side for instant comparison
- 🧮 **TF-IDF + Cosine Similarity** scoring engine
- 🔤 **Optional bigrams** — toggle whether word pairs (not just single words) are factored into the score
- 🚦 **Automatic verdict** — classifies the pair as:
  - 🟢 **Different Documents** (score `< 0.5`)
  - 🟠 **Similar Content** (score `0.5 – 0.85`)
  - 🔴 **Likely Duplicate** (score `> 0.85`)
- 📊 **Visual similarity gauge** showing the cosine score on a 0–1 scale
- 🔎 **Top shared words** chart when a duplicate is flagged, ranked by contribution to the score
- 🧹 **Text preprocessing preview** — see the cleaned/normalized version of each document
- 📋 **Top shared terms breakdown** with individual TF-IDF weights

---

## 🛠️ Tech Stack

| Component        | Library                         |
|-------------------|----------------------------------|
| Web App Framework  | [Streamlit](https://streamlit.io/) |
| Text Vectorization | `scikit-learn` (`TfidfVectorizer`) |
| Similarity Metric  | `scikit-learn` (`cosine_similarity`) |
| Visualization      | `matplotlib` |
| Text Cleaning      | Python `re` (regex) |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Anthronox-arch/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install streamlit matplotlib scikit-learn
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📁 How It Works

1. **Upload** two `.txt` files using the side-by-side uploaders.
2. Each document is **preprocessed** — lowercased, stripped of punctuation, and whitespace-normalized.
3. The cleaned texts are converted into **TF-IDF vectors** (unigrams, or unigrams + bigrams if enabled).
4. **Cosine similarity** is computed between the two vectors, producing a score between `0` and `1`.
5. The score is mapped to a **verdict** and displayed with a color-coded gauge.
6. If the documents are flagged as a **Likely Duplicate**, the app surfaces the words/terms that contributed most to the similarity score.

---

## 📸 Example Output

| Verdict | Meaning |
|----------|----------|
| 🟢 Different Documents | Little to no meaningful overlap |
| 🟠 Similar Content | Notable overlap — may share topic or structure |
| 🔴 Likely Duplicate | Near-identical content |

---

## 🧩 Possible Improvements

- Support for `.pdf` and `.docx` uploads
- Batch comparison across multiple files
- Adjustable similarity thresholds via the UI
- Word-cloud visualization of shared vocabulary

---

## 👤 Author

**Muhammad Ahmad Awan**
BSCS Student, University of Management and Technology, Lahore
[GitHub: Anthronox-arch](https://github.com/Anthronox-arch)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
