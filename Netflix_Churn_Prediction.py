#!/usr/bin/env python
# coding: utf-8

# #                                                           Netflix Churn Prediction

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# For better visuals
plt.style.use('seaborn-v0_8')


# ## Loading the Dataset

# In[2]:


df = pd.read_csv("Project/streaming_service_churn_data.csv")

df.head()


# ## Explore Dataset

# In[3]:


print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())

df.info()
print("\nMissing values:\n", df.isnull().sum())

df.describe(include='all')


# ## Handling Missing Values

# In[4]:


# Handle missing numerical values
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["WatchHoursPerWeek"].fillna(df["WatchHoursPerWeek"].mean(), inplace=True)

# Handle missing categorical values (mode)
df["PaymentMethod"].fillna(df["PaymentMethod"].mode()[0], inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum())


# ## Balancing the dataset

# In[5]:


from sklearn.utils import resample

# Separate churned and stayed users
churned = df[df["Churn"] == 1]
stayed = df[df["Churn"] == 0]

# Balance dataset
min_size = min(len(churned), len(stayed))
churned_bal = resample(churned, replace=False, n_samples=min_size, random_state=42)
stayed_bal = resample(stayed, replace=False, n_samples=min_size, random_state=42)

df_balanced = pd.concat([churned_bal, stayed_bal]).sample(frac=1, random_state=42).reset_index(drop=True)

print("New Churn Ratio:")
print(df_balanced["Churn"].value_counts(normalize=True))

# Continue using balanced data
df = df_balanced


# ## Visualize Churn and Features

# In[6]:


plt.figure(figsize=(5,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution (0 = Stay, 1 = Cancel)")
plt.show()

print(f"Churn Rate: {df['Churn'].mean() * 100:.2f}%")


# In[7]:


plt.figure(figsize=(6,4))
sns.countplot(x='Country', hue='Churn', data=df)
plt.title("Country-wise Churn Comparison")
plt.show()


# In[8]:


plt.figure(figsize=(6,4))
sns.countplot(x='Plan', data=df, order=df['Plan'].value_counts().index)
plt.title("Plan Distribution")
plt.show()


# In[9]:


plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='WatchHoursPerWeek', data=df)
plt.title("Watch Hours per Week vs Churn")
plt.show()


# In[10]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# ## Label Encoding for Categorical Columns

# In[11]:


label_cols = ['Country', 'Gender', 'Plan', 'PaymentMethod']
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

df.head()


# ## Feature Scaling

# In[12]:


scaler = StandardScaler()
num_cols = ['Age', 'MonthlyFee', 'TenureMonths', 'WatchHoursPerWeek', 'DevicesUsed']
df[num_cols] = scaler.fit_transform(df[num_cols])


# ## Split Data

# In[13]:


X = df.drop(columns=['UserID', 'Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Class distribution in Train:", np.bincount(y_train))
print("Class distribution in Test:", np.bincount(y_test))


# ## Tuning Random Forest with GridSearchCV

# In[14]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\nBest Parameters Found:", grid_search.best_params_)
best_rf = grid_search.best_estimator_


# ## Training and Evaluating the model

# In[15]:


best_rf.fit(X_train, y_train)
y_pred = best_rf.predict(X_test)

print("\n=== Random Forest Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))


# ## Confusion Matrix

# In[16]:


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ## Save Trained Model and Scaler

# In[17]:


import pickle

with open("netflix_churn_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully!")


# ## Feature Importance

# In[19]:


feat_importance = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,4))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title("Feature Importance in Predicting Churn")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

print("\nFeature Importance (%):\\")
for feature, importance in feat_importance.items():
    print(f"{feature}: {importance*100:.2f}%")


# In[ ]:




