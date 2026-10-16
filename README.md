# Sentiment Analysis of Customer Reviews Using Machine Learning

This project builds and evaluates machine-learning models that classify customer product reviews as **positive** or **negative**.

It compares a Logistic Regression model and a Linear Support Vector Machine (SVM) using TF-IDF text features.

## Project Overview

The workflow includes:

- Loading and exploring a customer-review dataset
- Converting numeric star ratings into sentiment labels
- Cleaning and preparing review text
- Splitting the data into training and test sets
- Converting text into TF-IDF features
- Training Logistic Regression and Linear SVM classifiers
- Evaluating models using precision, recall, F1-score, and confusion matrices
- Saving trained models for prediction
- Testing predictions on new reviews

## Dataset

The project uses the Kaggle **Product Review Rating Prediction** dataset.

The dataset contains 1,000 synthetic product reviews with:

- `review_text`: the customer review
- `rating`: a rating from 1 to 5

Ratings were converted as follows:

| Rating | Sentiment |
|---|---|
| 4–5 | Positive |
| 1–2 | Negative |
| 3 | Neutral |

Neutral reviews were excluded from the initial binary classification model.

## Models

| Model | Text Features | Result |
|---|---|---|
| Logistic Regression | TF-IDF unigrams and bigrams | 100% test accuracy |
| Linear SVM | TF-IDF unigrams and bigrams | 100% test accuracy |

## Important Limitation

The dataset contains repeated synthetic review templates. Because of this, the test set contains reviews very similar to those seen during training.

Therefore, the 100% accuracy demonstrates that the machine-learning pipeline works, but it should **not** be interpreted as real-world generalization performance.

A more realistic next step would be to use a larger, more diverse review dataset with fewer duplicated texts.

## Project Structure

```text
sentiment-analysis-reviews/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_explore_data.ipynb
│   └── 02_evaluate_model.ipynb
│
├── src/
│   ├── train.py
│   └── predict.py
│
├── models/
├── reports/
├── requirements.txt
└── README.md
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train the Models

```bash
python src/train.py
```

## Make a Prediction

```bash
python src/predict.py "The product is excellent and delivery was very fast."
```

Example output:

```text
Review: The product is excellent and delivery was very fast.
Predicted sentiment: positive
```

## Outputs

The training process saves:

- `models/logistic_regression.joblib`
- `models/linear_svm.joblib`
- `models/tfidf_vectorizer.joblib`

The evaluation notebook produces:

- `reports/confusion_matrix.png`
- `reports/important_words.png`

## Future Improvements

- Use a larger and more diverse real-world review dataset
- Add neutral-class classification
- Perform aspect-based sentiment analysis
- Fine-tune DistilBERT or RoBERTa
- Deploy the model with Streamlit or FastAPI
- Add explainability using SHAP or LIME
