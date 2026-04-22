# Sentiment Analysis of Financial News Using Machine Learning

**Module:** 7058EFA — Artificial Intelligence in FinTech  
**Institution:** Coventry University  
**Submitted:** April 2026

---

## Overview

This project benchmarks three classical ML classifiers against FinBERT on the Financial PhraseBank (Malo et al., 2014) for three-class financial news sentiment classification — **positive**, **negative**, and **neutral**. The trained SVM is deployed in a FinTech trading signal simulator and validated against 53,650 unseen headlines from Reuters, CNBC, and the Guardian.

### Key Results

| Model | Accuracy | F1-Score (Wt.) | Negative Recall |
|---|---|---|---|
| Logistic Regression | 73.20% | 0.71 | 0.43 |
| Multinomial Naive Bayes | 69.28% | 0.64 | 0.05 |
| SVM (linear kernel) | 73.20% | 0.72 | 0.52 |
| **FinBERT (transformer)** | **87.22%** | **0.87** | **0.97** |

FinBERT achieves a 14 percentage point accuracy gain over the best classical model and near-perfect Negative class recall (0.97 vs 0.52 for SVM), driven by its bidirectional attention resolving compound financial phrases as unified semantic units. The SVM is selected for the deployed trading simulator due to real-time speed, interpretable decision function, and regulatory auditability requirements.

---

## Pipeline

```
Financial PhraseBank (4,846 sentences)
        ↓
Preprocessing: lowercase → noise removal → tokenisation → stopword removal → lemmatisation
        ↓
TF-IDF Vectorisation (5,000 features)  ←──── Classical ML path
        ↓
┌─────────────────────────────────────────────────────┐
│  Cell 3: Logistic Regression  │  Naive Bayes  │ SVM │
└─────────────────────────────────────────────────────┘
        ↓
Raw headlines (uncleaned) ←───────────────────────────── FinBERT path
        ↓
┌──────────────────────────────┐
│  Cell 4: FinBERT (ProsusAI)  │
└──────────────────────────────┘
        ↓
Cell 5: Four-model performance comparison
Cell 6: SVM feature importance (coefficient weights)
Cell 7: FinTech trading signal simulator (SVM vs FinBERT)
Cell 8: Cross-dataset validation — Reuters, CNBC, Guardian (53,650 headlines)
Cell 9: Reuters temporal split — pre-2019 vs 2019+ (SVM + FinBERT)
Appendix Cell: TextBlob polarity analysis (supplementary)
```

---

## Repository Structure

```
├── sentiment_analysis_finbert.ipynb   # Full pipeline notebook (9 cells + appendix)
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── data/
    └── README_data.md                 # Dataset download instructions
```

> **Note:** The datasets are not included in this repository due to licensing and size constraints. See `data/README_data.md` for download instructions.

---

## Datasets

| Dataset | Source | Records | Use |
|---|---|---|---|
| Financial PhraseBank (75% agreement) | [Kaggle — ankurzing](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) | 4,846 labelled sentences | Training and test set |
| Reuters Financial News Headlines | [Kaggle — notlucasp](https://www.kaggle.com/datasets/notlucasp/financial-news-headlines) | 32,770 headlines | Cross-dataset validation, temporal split |
| CNBC Headlines | [Kaggle — notlucasp](https://www.kaggle.com/datasets/notlucasp/financial-news-headlines) | 3,080 headlines | Cross-dataset validation |
| Guardian Headlines | [Kaggle — notlucasp](https://www.kaggle.com/datasets/notlucasp/financial-news-headlines) | 17,800 headlines | Cross-dataset validation |

---

## Requirements

Python 3.8+ on Google Colab (recommended) or any local environment with GPU for FinBERT.

```bash
pip install -r requirements.txt
```

**To run Cell 4 (FinBERT) efficiently:**  
In Google Colab: `Runtime → Change runtime type → T4 GPU`  
CPU inference on 970 sentences takes 10–20 minutes. GPU takes ~3 minutes.

---

## Running the Notebook

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `sentiment_analysis_finbert.ipynb`
3. Download the datasets from Kaggle (see `data/README_data.md`)
4. Upload the following CSV files to `/content/` in Colab:
   - `all-data.csv` (Financial PhraseBank)
   - `reuters_headlines.csv`
   - `cnbc_headlines.csv`
   - `guardian_headlines.csv`
5. Run cells in order (Cell 1 → Cell 9, then Appendix Cell)

> **Cell 4 (FinBERT)** installs `transformers` and `torch` automatically on first run. The model (~700MB) is downloaded from HuggingFace Hub and cached for subsequent runs.

---

## Notebook Cell Summary

| Cell | Description | Output |
|---|---|---|
| 1 | Imports, data loading, TF-IDF preprocessing | `Preprocessing complete` |
| 2 | Exploratory data analysis | Class distribution charts |
| 3 | Classical model training + confusion matrices | 3 confusion matrix heatmaps |
| 4 | FinBERT evaluation (4th model) | Classification report + confusion matrix |
| 5 | Four-model performance comparison | Grouped bar chart + Table 1 |
| 6 | SVM feature importance | 3-panel coefficient chart |
| 7 | FinTech trading signal simulator (SVM vs FinBERT) | Table 2 — signal output |
| 8 | Cross-dataset validation (Reuters, CNBC, Guardian) | Distribution chart |
| 9 | Reuters temporal split (SVM vs FinBERT, pre/post 2019) | 2×2 distribution chart |
| Appendix | TextBlob polarity analysis (supplementary) | Polarity histograms + stats |

---

## References

- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *JASIST*, 65(4), 782–796.
- Yang, Y., Uy, M. C. S., & Huang, A. (2020). FinBERT: A pretrained language model for financial communications. *arXiv:2006.08097*.
- Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35–65.
- Karanikola, A., et al. (2023). Financial sentiment analysis: Classic methods vs. deep learning models. *Intelligent Decision Technologies*, 17(3).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The Financial PhraseBank dataset is licensed under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) and must not be used for commercial purposes.
