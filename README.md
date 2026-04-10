# AG News Text Classification

This project develops and evaluates multiple NLP models to classify news articles into four categories: **World, Sports, Business, and Sci/Tech** using the AG News dataset.

The objective is not just to maximize accuracy, but to determine the **best tradeoff between performance, interpretability, and computational cost** for real-world deployment.

---

## Results

| Model                          | Test Accuracy | Notes |
|-------------------------------|-------------|------|
| Naïve Bayes (TF-IDF)          | ~91%        | Fast, strong baseline |
| Logistic Regression (TF-IDF)  | ~91–92%     | Best balance of performance + interpretability |
| LSTM (RNN)                    | ~91–93%     | Slight improvement, higher complexity |
| RoBERTa Transformer           | ~94%        | Best performance, highest computational cost |

> Key takeaway: **Logistic Regression with TF-IDF provides the best practical tradeoff**, while Transformers offer marginal gains at significantly higher cost :contentReference[oaicite:0]{index=0}

---

## Key Insights

- Combining **title + description** consistently produced the best results across all models
- Traditional linear models remain highly competitive for structured text classification
- Deep learning models (LSTM, Transformer) provide incremental improvements, not breakthroughs, for this task
- Model selection should be treated as a **business decision**, not just a technical one

---

## Project Structure

```
ag-news-text-classification/
├── data/
│   └── raw/
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   ├── 01_tfidf_models_and_ngram_rankings.ipynb
│   ├── 02_rnn_model.ipynb
│   └── 03_transformer_model.ipynb
├── reports/
│   └── final_report.pdf
├── README.md
└── requirements.txt
```


---

## Reproducibility

### Dataset

This project uses the **AG News dataset** from Kaggle.

Expected structure:

```
data/raw/
├── train.csv
└── test.csv
```


If not included, download from:
https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

---

### Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### Running the Project

Run notebooks in order:

1. `01_tfidf_models_and_ngram_rankings.ipynb`
    - Naïve Bayes
    - Logistic Regression
    - TF-IDF + n-gram analysis
2. `02_rnn_models.ipynb`
    - LSTM-based sequence model
3. `03_transformer_model.ipynb`
    - RoBERTa Transformer (transfer learning)

---

### Modeling Approach

The project follows a structured progression:

1. **Baseline models**
    - TF-IDF vectorization
    - Naïve Bayes
    - Logistic Regression
2. **Sequential modeling**
    - Tokenization
    - LSTM (RNN)
3. **Transformer-based modeling**
    - Subword tokenization
    - Pretrained RoBERTa encoder
    - Fine-tuned classification head

---

### Data Overview
- **120,000 training samples**
- **7,600 test samples**
- Balanced across 4 classes (25% each)
- Features:
    - Title
    - Description

---

### Business Recommendation

For most real-world applications:
- Use **Logistic Regression + TF-IDF**
    - Fast
    - Interpretable
    - Strong performance

- Use **Transformers only when**:
    - marginal accuracy gains justify the increased compute + infrastructure cost

---

### Report

Full project report available:

`reports/final_report.pdf`

---

### Tech Stack
- Python
- Pandas / NumPy
- Scikit-learn
- TensforFlow / Keras
- Keras NLP / Transformer models
- Matplotlib / Seaborn

