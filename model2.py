# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("support_tickets.csv")

# ==========================================
# CREATE TEXT FEATURE
# ==========================================

df['text'] = (
    df['Ticket Subject'].fillna('') + ' ' +
    df['Ticket Description'].fillna('')
)

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df['text'])

# ==========================================
# CATEGORY CLASSIFICATION
# Ticket Type = Category
# ==========================================

print("\n==============================")
print("CATEGORY CLASSIFICATION")
print("==============================")

y_category = df['Ticket Type']

X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
    X,
    y_category,
    test_size=0.2,
    random_state=42,
    stratify=y_category
)

category_model = LinearSVC()

category_model.fit(X_train_cat, y_train_cat)

category_pred = category_model.predict(X_test_cat)

print("\nCategory Accuracy:")
print(accuracy_score(y_test_cat, category_pred))

print("\nCategory Classification Report:")
print(classification_report(y_test_cat, category_pred))

# ==========================================
# CATEGORY CONFUSION MATRIX
# ==========================================

cm_cat = confusion_matrix(y_test_cat, category_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_cat,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=category_model.classes_,
    yticklabels=category_model.classes_
)

plt.title("Category Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================================
# PRIORITY CLASSIFICATION
# Ticket Priority = Priority
# ==========================================

print("\n==============================")
print("PRIORITY CLASSIFICATION")
print("==============================")

y_priority = df['Ticket Priority']

X_train_pri, X_test_pri, y_train_pri, y_test_pri = train_test_split(
    X,
    y_priority,
    test_size=0.2,
    random_state=42,
    stratify=y_priority
)

priority_model = LinearSVC()

priority_model.fit(X_train_pri, y_train_pri)

priority_pred = priority_model.predict(X_test_pri)

print("\nPriority Accuracy:")
print(accuracy_score(y_test_pri, priority_pred))

print("\nPriority Classification Report:")
print(classification_report(y_test_pri, priority_pred))

# ==========================================
# PRIORITY CONFUSION MATRIX
# ==========================================

cm_pri = confusion_matrix(y_test_pri, priority_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_pri,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=priority_model.classes_,
    yticklabels=priority_model.classes_
)

plt.title("Priority Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()