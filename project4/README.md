# Project 4: NLP & Sentiment Analysis

**Notebook:** `sentiment_analysis.ipynb`

## Goal

Program a machine to read and mathematically categorize unstructured human text, predicting whether a movie review is Positive or Negative.

## Dataset

`IMDB_Dataset.csv` — [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle), 50,000 reviews, perfectly balanced (25,000 positive / 25,000 negative).

> **Note:** An initial candidate dataset (`Customer_Sentiment.csv`, product reviews) was evaluated and rejected — its `review_text` column had only 15 unique template phrases repeated across 25,000 rows, which wouldn't exercise real text pre-processing or generalization. IMDB was used instead for genuinely diverse, organically-written text.

## Approach & Results

**1. Data quality check**
29,200 of 50,000 reviews (58.4%) contain leftover `<br />` HTML tags from the original scrape — stripped during cleaning to avoid polluting the vocabulary with tags.

**2. Text pre-processing pipeline**
Four stages: clean (strip HTML, lowercase, remove punctuation/digits) → tokenize (NLTK `word_tokenize`) → remove stop words → lemmatize (NLTK `WordNetLemmatizer`).

**Bug found and fixed — negation handling.** NLTK's default English stopword list removes `not`, `no`, `nor`, and every contracted negative auxiliary (`wasn`, `didn`, `isn`, `doesn`, `couldn`, etc.). Combined with an initial cleaning step that stripped apostrophes before tokenizing, `"wasn't good"` was collapsing to just `"good"` — completely inverting the apparent sentiment. Fixed by preserving apostrophes during cleaning (so `"wasn't"` tokenizes correctly into `"was"` + `"n't"`) and excluding negation words from stopword removal.

**3. TF-IDF vectorization**
`TfidfVectorizer(max_features=10000, ngram_range=(1,2))` — bigrams included so phrases like "not good" are captured as a unit. Vocabulary capped at 10,000 terms.

**Bug found and fixed — data leakage.** The vectorizer was initially fit on all 50,000 reviews before the train/test split, leaking test-set document statistics into the IDF weighting — the same class of mistake as applying SMOTE before splitting in Project 2. Fixed by splitting the raw text first, then fitting the vectorizer only on training data (`fit_transform` on train, `transform` only on test).

**4. Two classifiers**
- **Naive Bayes** (`MultinomialNB`) — fast baseline
- **SVM** (`LinearSVC`) — a linear-kernel SVM, chosen over the default RBF-kernel `SVC` because `SVC` scales poorly to 40,000 rows × 10,000 TF-IDF features

**5. Final results (post-fix)**

| Model | Accuracy | Precision (avg) | Recall (avg) |
|-------|----------|------------------|----------------|
| Naive Bayes | 86.93% | 0.87 | 0.87 |
| **SVM (LinearSVC)** | **89.45%** | 0.89 | 0.89 |

Note: fixing both bugs barely moved these numbers (86.90%→86.93%, 89.25%→89.45%) — expected on a dataset this large, since global IDF statistics and stopword effects average out similarly with or without the leakage. The fixes matter for individual predictions on negation-heavy reviews and for methodological correctness, not for this aggregate metric.

Both models correctly classified hand-written test reviews, including a deliberately ambiguous one ("okay, not great but not terrible either") — both called it negative.

**Known minor limitation:** 133 reviews (0.27%) appear as exact duplicates across the train/test split. Checked and considered negligible at this scale — not addressed.

## Tech Stack

pandas, NumPy, NLTK, scikit-learn, matplotlib

## Files

| File | Description |
|------|-------------|
| `IMDB_Dataset.csv` | Input: raw movie reviews with sentiment labels |
| `sentiment_analysis.ipynb` | Full pipeline: load → pre-process → TF-IDF → train → evaluate |

## How to Run

```bash
pip install pandas numpy nltk scikit-learn matplotlib
jupyter notebook sentiment_analysis.ipynb
```

The notebook downloads required NLTK corpora (`punkt`, `stopwords`, `wordnet`, `omw-1.4`) on first run.
