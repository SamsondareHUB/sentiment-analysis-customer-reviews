import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

project_dir = Path.home() / "Documents" / "sentiment-analysis-reviews"
data_path = project_dir / "data" / "processed" / "clean_reviews.csv"
model_dir = project_dir / "models"
report_dir = project_dir / "reports"

model_dir.mkdir(parents=True, exist_ok=True)
report_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_path)

print("Dataset shape:", df.shape)
print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    df["review_text"],
    df["sentiment"],
    test_size=0.2,
    stratify=df["sentiment"],
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=20000,
    min_df=2
)

X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

print("\nTF-IDF feature matrix shape:")
print(X_train_features.shape)

models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    "linear_svm": LinearSVC(
        class_weight="balanced"
    )
}

for name, model in models.items():
    print(f"\n{'=' * 50}")
    print(f"Training {name}")
    print(f"{'=' * 50}")

    model.fit(X_train_features, y_train)
    predictions = model.predict(X_test_features)

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    model_path = model_dir / f"{name}.joblib"
    joblib.dump(model, model_path)

    print("\nModel saved to:")
    print(model_path)

vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
joblib.dump(vectorizer, vectorizer_path)

print(f"\n{'=' * 50}")
print("Training complete.")
print("Vectorizer saved to:")
print(vectorizer_path)