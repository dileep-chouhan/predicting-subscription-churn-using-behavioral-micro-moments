import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
# Simulate user behavior data
num_users = 500
data = {
    'login_count': np.random.randint(1, 30, num_users),
    'avg_session_duration': np.random.randint(5, 60, num_users),
    'feature_usage_count': np.random.randint(0, 20, num_users),
    'support_tickets': np.random.randint(0, 5, num_users),
    'churned': np.random.choice([0, 1], num_users, p=[0.8, 0.2]) # 20% churn rate
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Feature Engineering (Minimal in this synthetic example) ---
# In a real-world scenario, this section would involve handling missing values, 
# outlier detection, and potentially creating more sophisticated features.
# --- 3. Analysis and Modeling ---
X = df.drop('churned', axis=1)
y = df['churned']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
# --- 4. Visualization ---
# Feature Importance (Illustrative -  LogisticRegression doesn't directly provide feature importance like tree-based models)
feature_importance = pd.Series(model.coef_[0], index=X.columns)
plt.figure(figsize=(8, 6))
feature_importance.plot(kind='bar')
plt.title('Feature Importance (Illustrative)')
plt.ylabel('Coefficient Magnitude')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Plot saved to feature_importance.png")
#Churn rate visualization
plt.figure(figsize=(6,4))
sns.countplot(x='churned', data=df)
plt.title('Churn Rate')
plt.xlabel('Churned (0=No, 1=Yes)')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('churn_rate.png')
print("Plot saved to churn_rate.png")