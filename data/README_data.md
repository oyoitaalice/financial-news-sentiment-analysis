# Dataset Download Instructions

The datasets are **included** in this repository in the root folder at /content. They can also be downloaded from their source on Kaggle and placed in the `/content/` directory when running on Google Colab (or adjust paths for local use).

---

## Dataset 1 — Financial PhraseBank

**Source:** https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news  
**Licence:** CC BY-NC-SA 3.0 (non-commercial academic use only)  
**File needed:** `all-data.csv`

This is the primary training and evaluation dataset. It contains 4,846 English-language financial sentences labelled by 16 domain-expert annotators as positive, negative, or neutral. The 75% agreement configuration (all 4,846 rows of `all-data.csv`) is used in this project.

**Citation:**  
Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.

---

## Dataset 2 — Reuters / CNBC / Guardian Financial Headlines

**Source:** https://www.kaggle.com/datasets/notlucasp/financial-news-headlines  
**Licence:** Kaggle open access  
**Files needed:**
- `reuters_headlines.csv` — 32,770 headlines (used in Cells 8 and 9)
- `cnbc_headlines.csv` — 3,080 headlines (used in Cell 8)
- `guardian_headlines.csv` — 17,800 headlines (used in Cell 8)

These datasets are used for out-of-sample cross-dataset validation (Cell 8) and the Reuters temporal split experiment (Cell 9). They are unlabelled — the trained SVM predicts sentiment distributions across them.

The `reuters_headlines.csv` file contains a `Time` column with publication dates spanning March 2018 to July 2020, which is used for the chronological split at January 2019 in Cell 9.

---

## How to Upload in Google Colab

After downloading from Kaggle, upload all four CSV files using the Colab file browser (folder icon in the left sidebar):

```
/content/all-data.csv
/content/reuters_headlines.csv
/content/cnbc_headlines.csv
/content/guardian_headlines.csv
```

Or upload programmatically:

```python
from google.colab import files
uploaded = files.upload()
```

---

## Ethical and Reproducibility Notes

- Neither dataset contains personally identifiable information.
- Both datasets are open-access and satisfy the reproducibility requirements of the 7058EFA assignment brief.
- Direct links to the datasets (not just the hosting sites) are provided above, as required.
- The Financial PhraseBank must not be used for commercial purposes under its CC BY-NC-SA 3.0 licence.
