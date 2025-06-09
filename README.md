# 🎬 Movie Success Predictor

This is a **Streamlit web app** that predicts whether a movie will be a **Hit**, **Average**, or **Flop** based on:

- Budget
- Runtime
- Genre
- Release Month

It uses a trained machine learning classification model and encoders saved using `joblib`.

---

## 🚀 How to Run the App

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-success-predictor.git
cd movie-success-predictor
```

### 2. Install the required libraries

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app

```bash
streamlit run Movies.py
```

After a few seconds, it will open a local server in your browser.

---

## 📁 Project Files

- `Movies.py` - Main Streamlit app
- `movie_success_model.pkl` - Trained ML model
- `genre_encoder.pkl` - Encoder for genre feature
- `target_encoder.pkl` - Encoder for prediction labels (Hit, Average, Flop)
- `requirements.txt` - Required Python libraries
- `README.md` - Project overview (this file)

---

## 📊 Features

- Simple and interactive user interface
- Predicts movie success in real-time
- Encodes genre and prediction categories
- Can be extended with visualization and more features

---

## 📦 Built With

- Python
- Streamlit
- scikit-learn
- pandas
- joblib

---

## ✨ Future Ideas

- Add feature importance charts
- Add probability-based confidence chart
- Accept batch prediction via CSV upload
- Deploy to Streamlit Cloud

