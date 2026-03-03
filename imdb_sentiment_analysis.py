# ==============================
# 1. Import Libraries
# ==============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

# ==============================
# 2. Load Dataset
# ==============================

df = pd.read_csv("IMDB Dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==============================
# 3. Text Cleaning Function
# ==============================

def clean_text(text):
    text = re.sub('<.*?>', '', text)      # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove numbers and punctuation
    text = text.lower()                  # Convert to lowercase
    text = re.sub('\s+', ' ', text)      # Remove extra spaces
    return text

df['review'] = df['review'].apply(clean_text)

# ==============================
# 4. Convert Target to Numeric
# ==============================

df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# ==============================
# 5. Split Features and Labels
# ==============================

X = df['review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# 6. TF-IDF Vectorization
# ==============================

vectorizer = TfidfVectorizer(
    max_features=15000,
    stop_words='english'
)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

print("TF-IDF Feature Shape:", X_train.shape)

# ==============================
# 7. Logistic Regression Model
# ==============================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:,1]

# ==============================
# 8. Naïve Bayes Model
# ==============================

nb_model = MultinomialNB(alpha=1.0)
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)
y_prob_nb = nb_model.predict_proba(X_test)[:,1]

# ==============================
# 9. Evaluation Function
# ==============================

def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Performance:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-Score:", f1_score(y_true, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

evaluate_model("Logistic Regression", y_test, y_pred_log)
evaluate_model("Naïve Bayes", y_test, y_pred_nb)

# ==============================
# 10. Confusion Matrix (Logistic)
# ==============================

cm = confusion_matrix(y_test, y_pred_log)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==============================
# 11. ROC Curve
# ==============================

fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
roc_auc_log = auc(fpr_log, tpr_log)

plt.figure()
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {roc_auc_log:.2f})")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ==============================
# 12. Hyperparameter Tuning (Optional but Strong for Report)
# ==============================

param_grid = {
    'C': [0.1, 1, 10]
}

grid = GridSearchCV(LogisticRegression(max_iter=1000),
                    param_grid,
                    cv=5,
                    scoring='f1')

grid.fit(X_train, y_train)

print("\nBest Parameters (Logistic Regression):", grid.best_params_)
print("Best Cross-validation F1 Score:", grid.best_score_)

# ==============================
# 13. Cross Validation Scores
# ==============================

cv_scores = cross_val_score(log_model, X_train, y_train, cv=5, scoring='accuracy')
print("\nCross-validation Accuracy:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())

# ==============================
# 14. Model Comparison Plot
# ==============================

models = ['Logistic Regression', 'Naïve Bayes']
accuracies = [
    accuracy_score(y_test, y_pred_log),
    accuracy_score(y_test, y_pred_nb)
]

plt.figure()
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()