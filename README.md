# Medical Tweet Classification with NLP
Binary classification model for Spanish-language COVID-19 tweets, trained to distinguish between mentions of **healthcare professionals** and mentions of the **general population**. Project developed as part of the Master's in Data Science, Big Data & Business Analytics at Universidad Complutense de Madrid.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1utmnP9W_7jXRHNeMqPyu-dI8gY0oAobL?usp=sharing)

---

## Results

| Model | F1-Score |
|--------|----------|
| Baseline TF-IDF + Logistic Regression | 0.526 |
| **BETO fine-tuned (3 epochs)** | **0.804** |

**+53%** improvement over the baseline.

---

## Problem description
The dataset used is **ProfNER** — a corpus of manually annotated Spanish-language tweets from the COVID-19 pandemic context. The task is to classify each tweet into one of two categories:

- `1` — The tweet mentions a healthcare professional
- `0` — The tweet does not mention a healthcare professional

The difficulty of the problem lies in the informal nature of language on Twitter: abbreviations, spelling errors, language mixing, and semantic ambiguity.

---

## Methodology

### 1. Preprocessing
- Text cleaning: removal of URLs, mentions, and special characters
- Tokenization with BETO's native tokenizer
- Truncation and padding to a maximum length of 128 tokens

### 2. Baseline
- TF-IDF vectorization
- Classifier: Logistic Regression
- F1-Score: **0.526**

### 3. Fine-tuned model
- Base model: [`dccuchile/bert-base-spanish-wwm-uncased`](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased) (BETO)
- Supervised fine-tuning on the ProfNER dataset
- 3 training epochs
- Final F1-Score: **0.804**

---

## Technologies
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

- **Transformers:** HuggingFace `transformers` + `datasets`
- **Base model:** BETO (`dccuchile/bert-base-spanish-wwm-uncased`)
- **Baseline:** Scikit-learn (TF-IDF + Logistic Regression)
- **Environment:** Google Colab (GPU T4)

---

## Repository structure
```
tweet-classification-beto/
│
├── README.md
├── notebook.ipynb        # Full notebook with code and results
└── requirements.txt      # Project dependencies
```

---

## How to run
1. Open the notebook in Google Colab using the button above
2. Run the cells in order
3. Training requires a GPU — enable it under `Runtime → Change runtime type → T4 GPU`

---

## References
- Dataset: [ProfNER — NLP4COVID shared task](https://temu.bsc.es/profner/)
- Base model: [BETO — dccuchile/bert-base-spanish-wwm-uncased](https://huggingface.co/dccuchile/bert-base-spanish-wwm-uncased)

---

## Author
**Daniel Molina Novoa** · Data Analyst
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/daniel-molina-novoa/)
[![GitHub](https://img.shields.io/badge/GitHub-danielmolinan-181717?style=flat&logo=github&logoColor=white)](https://github.com/danielmolinan)
