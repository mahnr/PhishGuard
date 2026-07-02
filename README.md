# 🛡️ PhishGuard — AI-Powered Phishing Website Detection System

A university final semester project that detects phishing URLs using
machine learning and cybersecurity-based feature engineering.

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python model/train_model.py
streamlit run app.py
```

## 📁 Project Structure

```
PhishGuard/
├── app.py                  → Streamlit dashboard (main entry)
├── requirements.txt        → Python dependencies
├── saved_model.pkl         → Trained ML model (auto-generated)
├── README.md               → Project documentation
│
├── dataset/
│   └── phishing_data.csv   → Labeled URL dataset
│
├── model/
│   ├── train_model.py      → Model training script
│   └── evaluate_model.py   → Evaluation & metrics
│
├── utils/
│   ├── feature_extractor.py → URL feature engineering
│   └── predictor.py         → Prediction helper
│
└── assets/
    └── style.css           → Custom UI styles
```

## 🧠 ML Models Used
- Random Forest Classifier (primary)
- Logistic Regression (baseline comparison)

## 👨‍🎓 Academic Project — Final Semester
