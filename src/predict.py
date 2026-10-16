import joblib
import sys
from pathlib import Path

project_dir = Path.home() / "Documents" / "sentiment-analysis-reviews"
model_dir = project_dir / "models"

vectorizer_path = model_dir / "tfidf_vectorizer.joblib"
model_path = model_dir / "linear_svm.joblib"

vectorizer = joblib.load(vectorizer_path)
model = joblib.load(model_path)

if len(sys.argv) < 2:
    print("Usage:")
    print('python src/predict.py "Your review text here"')
    sys.exit(1)

review = " ".join(sys.argv[1:])

features = vectorizer.transform([review])
prediction = model.predict(features)[0]

print(f"Review: {review}")
print(f"Predicted sentiment: {prediction}")