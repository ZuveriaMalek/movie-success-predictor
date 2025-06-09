#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# In[53]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")
print(movies.head())
print(credits.head())


# In[54]:


print(movies.shape)
print(credits.shape)

print(movies.isnull().sum())
print(credits.isnull().sum())


# In[55]:


# Drop homepage column from movies
movies = movies.drop(columns=['homepage'])

# Fill missing overview and tagline with empty strings
movies['overview'] = movies['overview'].fillna('')
movies['tagline'] = movies['tagline'].fillna('')

# Drop rows with missing release_date
movies = movies.dropna(subset=['release_date'])

# Fill missing runtime with median
median_runtime = movies['runtime'].median()
movies['runtime'] = movies['runtime'].fillna(median_runtime)


# In[45]:


# Ensure id columns are integer type for merge
movies['id'] = movies['id'].astype(int)
credits['movie_id'] = credits['movie_id'].astype(int)

#Rename the col name in credits to match with movies
credits = credits.rename(columns={'movie_id': 'id'})

# Merge on'id'
df = movies.merge(credits, on='id')

print(df.shape)
print(df.columns)


# In[47]:


#Feature Engineering step
def extract_genre(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        if genres:
            return genres[0]['name']
        else:
            return 'Unknown'
    except:
        return 'Unknown'

df['main_genre'] = df['genres'].apply(extract_genre)

# Extract release month from release_date
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month.fillna(0).astype(int)

# Create Success label (target)
df['Success'] = df['vote_average'].apply(lambda x: 'Hit' if x >= 7 else ('Average' if x >= 5 else 'Flop'))

# Encode Categorical Features
le_genre = LabelEncoder()
df['genre_encoded'] = le_genre.fit_transform(df['main_genre'])


# In[48]:


# Define Features and Target
features = ['budget', 'runtime', 'release_month', 'genre_encoded']
target = 'Success'

X = df[features]
y = df[target]

le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)


# In[49]:


#Train-Test Split and Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[50]:


# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le_target.classes_))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le_target.classes_, yticklabels=le_target.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[51]:


# Save Model
joblib.dump(model, 'movie_success_model.pkl')
joblib.dump(le_genre, 'genre_encoder.pkl')
joblib.dump(le_target, 'target_encoder.pkl')


# In[52]:


import streamlit as st
import pandas as pd
import joblib

# Load saved models
model = joblib.load('movie_success_model.pkl')
le_genre = joblib.load('genre_encoder.pkl')
le_target = joblib.load('target_encoder.pkl')

st.title("🎬 Movie Success Predictor")
st.markdown("Predict whether your movie will be a **Hit**, **Average**, or **Flop** based on budget, runtime, genre, and release month.")

# User input
budget = st.number_input("Enter Budget (in $)", min_value=10000, max_value=500000000, step=10000)
runtime = st.slider("Runtime (minutes)", 60, 240, 120)
release_month = st.selectbox("Release Month", list(range(1, 13)))
genre = st.selectbox("Genre", sorted(le_genre.classes_))

# Encode genre
genre_encoded = le_genre.transform([genre])[0]

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([[budget, runtime, release_month, genre_encoded]],
                            columns=['budget', 'runtime', 'release_month', 'genre_encoded'])
    prediction = model.predict(input_df)[0]
    result = le_target.inverse_transform([prediction])[0]
    st.success(f"🎉 Prediction: **{result}**")

