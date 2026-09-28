#!/usr/bin/env python
# coding: utf-8

# In[6]:


# 📊 Netflix Churn Prediction - Data Cleaning, Preprocessing & Visualization

# --- Step 1: Import Libraries ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')

# For better visuals
plt.style.use('seaborn-v0_8')


# In[7]:


df = pd.read_csv("Project\streaming_service_churn_data.csv")
df.head()


# In[8]:


# Check shape and columns
print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())

# Info summary
df.info()

# Check missing values
print("\nMissing values:\n", df.isnull().sum())

# Summary stats
df.describe(include='all')


# In[9]:


# Handle missing numerical values
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["WatchHoursPerWeek"].fillna(df["WatchHoursPerWeek"].mean(), inplace=True)

# Handle missing categorical values (mode)
df["PaymentMethod"].fillna(df["PaymentMethod"].mode()[0], inplace=True)

# Verify no missing data
print("\nMissing values after cleaning:\n", df.isnull().sum())


# In[38]:


# --- Step: Balance dataset to 50-50 churn ratio ---
from sklearn.utils import resample

# Separate churned and non-churned users
churned = df[df["Churn"] == 1]
stayed = df[df["Churn"] == 0]

# Make both classes the same size (use the smaller group size)
min_size = min(len(churned), len(stayed))

churned_bal = resample(churned, replace=False, n_samples=min_size, random_state=42)
stayed_bal = resample(stayed, replace=False, n_samples=min_size, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([churned_bal, stayed_bal]).sample(frac=1, random_state=42).reset_index(drop=True)

print("New Churn Ratio:")
print(df_balanced["Churn"].value_counts(normalize=True))

# Continue using the balanced dataset
df = df_balanced


# In[ ]:





# In[39]:


# Split data
X = df.drop(columns=['UserID', 'Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[40]:


plt.figure(figsize=(5,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution (0 = Active, 1 = Cancelled)")
plt.show()

churn_rate = df["Churn"].mean() * 100
print(f"Overall Churn Rate: {churn_rate:.2f}%")


# In[41]:


plt.figure(figsize=(6,4))
sns.countplot(x='Country', hue='Churn', data=df)
plt.title("Country-wise Churn Comparison")
plt.show()


# In[42]:


plt.figure(figsize=(6,4))
sns.countplot(x='Plan', data=df, order=df['Plan'].value_counts().index)
plt.title("Distribution of Netflix Plans")
plt.show()


# In[43]:


plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='WatchHoursPerWeek', data=df)
plt.title("Average Watch Hours per Week vs Churn")
plt.show()


# In[44]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# In[45]:


label_cols = ['Country', 'Gender', 'Plan', 'PaymentMethod']
le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

df.head()


# In[46]:


scaler = StandardScaler()
num_cols = ['Age', 'MonthlyFee', 'TenureMonths', 'WatchHoursPerWeek', 'DevicesUsed']
df[num_cols] = scaler.fit_transform(df[num_cols])


# In[47]:


X = df.drop(columns=['UserID', 'Churn'])
y = df['Churn']


# In[48]:


# --- Step 1: Split the dataset into training and testing sets ---
from sklearn.model_selection import train_test_split

# Stratify ensures same churn ratio in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Class distribution in Train:", np.bincount(y_train))
print("Class distribution in Test:", np.bincount(y_test))


# In[49]:


# --- Step 2: Balance the training data using SMOTE ---
from imblearn.over_sampling import SMOTE
import numpy as np

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_bal))


# In[50]:


# --- Step 3: Tune Random Forest using GridSearchCV ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
grid_search.fit(X_train_bal, y_train_bal)

print("\nBest Parameters Found:", grid_search.best_params_)
best_rf = grid_search.best_estimator_


# In[51]:


# --- Step 4: Train and Evaluate the Best Model ---
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

best_rf.fit(X_train_bal, y_train_bal)
y_pred = best_rf.predict(X_test)

print("\n=== Improved Random Forest Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))


# In[52]:


# --- Step 5: Confusion Matrix Heatmap ---
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[ ]:





# In[53]:


# --- Step 6: Save the Trained Model and Scaler ---
import pickle

with open("netflix_churn_model.pkl", "wb") as f:
    pickle.dump(best_rf, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("💾 Model and scaler saved successfully!")


# In[ ]:





# In[54]:


# --- Step 7: Feature Importance Visualization ---
feat_importance = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10,4))
sns.barplot(x=feat_importance, y=feat_importance.index)
plt.title("Feature Importance in Predicting Churn")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

print("\nFeature Importance (%):")
for feature, importance in feat_importance.items():
    print(f"{feature}: {importance*100:.2f}%")


# In[55]:


df['Churn'].value_counts(normalize=True)


# In[ ]:





# In[ ]:





# In[ ]:




