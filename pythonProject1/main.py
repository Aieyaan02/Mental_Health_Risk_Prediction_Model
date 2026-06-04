import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("mental_health_risk_prediction_1000.csv")

# Drop non-informative column
df = df.drop(columns=['user_id'])

# Encode target variable
df['mental_risk'] = LabelEncoder().fit_transform(df['mental_risk'])

# Features and target
X = df.drop(columns=['mental_risk'])
y = df['mental_risk']

# Normalize features
X_scaled = StandardScaler().fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

#  Train KNN with tuned parameters
knn = KNeighborsClassifier(
    n_neighbors=7,
    weights='distance',
    metric='manhattan'
)

knn.fit(X_train, y_train)

# Predict
y_pred = knn.predict(X_test)

# Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print metrics
print("Confusion Matrix:\n", conf_matrix)
print("\nAccuracy:", round(accuracy, 2))
print("Precision:", round(precision, 2))
print("Recall:", round(recall, 2))
print("F1 Score:", round(f1, 2))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
plt.title("Confusion Matrix - Tuned KNN Classifier")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
