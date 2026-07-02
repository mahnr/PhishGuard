# ============================================================
# PhishGuard — Improved Model Training Script
# model/train.py
#
# Run: python model/train.py
# ============================================================

import os, sys, pickle, warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.feature_extraction import extract_features, FEATURE_NAMES

DATASET_PATH    = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'urls.csv')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')


def load_dataset(path):
    print("📂  Loading dataset...")
    df = pd.read_csv(path)
    df.dropna(subset=['url', 'label'], inplace=True)
    df['url']   = df['url'].astype(str).str.strip()
    df['label'] = df['label'].astype(int)
    print(f"    Rows: {len(df)} | Safe: {(df.label==0).sum()} | Phishing: {(df.label==1).sum()}")

    print("⚙️   Extracting features...")
    rows = []
    for i, url in enumerate(df['url']):
        try:
            rows.append(extract_features(url))
        except:
            rows.append({k: 0 for k in FEATURE_NAMES})

    X = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = df['label'].reset_index(drop=True)
    print(f"    Feature matrix: {X.shape}")
    return X, y


def train(X_train, y_train):
    print("\n🌲  Training improved ensemble model...")

    # Model 1: Random Forest (stronger settings)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Model 2: Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=2,
        random_state=42
    )

    # Model 3: Logistic Regression (inside pipeline with scaler)
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42, C=1.0))
    ])

    # Voting Ensemble: combines all 3 models
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb), ('lr', lr_pipeline)],
        voting='soft',   # soft = use probability scores
        weights=[3, 2, 1]  # RF trusted most
    )

    ensemble.fit(X_train, y_train)
    print("    ✅ Ensemble trained (RF + GradientBoosting + LogisticRegression)")
    return ensemble, rf  # return RF separately for feature importances


def evaluate(model, X_test, y_test, name="Model"):
    y_pred = model.predict(X_test)
    print(f"\n{'='*55}")
    print(f"  📊 {name} — Evaluation")
    print(f"{'='*55}")
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
    print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted Safe   Predicted Phishing")
    print(f"  Actual Safe       {cm[0][0]:<5}            {cm[0][1]:<5}")
    print(f"  Actual Phish      {cm[1][0]:<5}            {cm[1][1]:<5}")
    report = classification_report(y_test, y_pred, target_names=['Safe','Phishing'])
    for line in report.splitlines():
        print(f"  {line}")
    return accuracy_score(y_test, y_pred)


def feature_importances(rf_model, feature_names, top_n=10):
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1]
    print(f"\n🔍  Top {top_n} Features (Random Forest component):")
    print(f"  {'Rank':<5} {'Feature':<30} {'Importance':>12}")
    print(f"  {'-'*50}")
    for rank, idx in enumerate(indices[:top_n], 1):
        print(f"  {rank:<5} {feature_names[idx]:<30} {importances[idx]:>12.4f}")


if __name__ == '__main__':
    print("="*55)
    print("  🛡️  PhishGuard — Improved Training Pipeline")
    print("="*55)

    X, y = load_dataset(DATASET_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\n✂️   Train: {len(X_train)} | Test: {len(X_test)}")

    ensemble, _ = train(X_train, y_train)
    rf_component = ensemble.estimators_[0]  # fitted RF from VotingClassifier

    acc = evaluate(ensemble, X_test, y_test, "Ensemble (RF + GB + LR)")

    print("\n🔁  5-Fold Cross Validation:")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv  = cross_val_score(ensemble, X, y, cv=skf, scoring='f1')
    print(f"    Scores: {[round(s,4) for s in cv]}")
    print(f"    Mean  : {cv.mean():.4f} ± {cv.std():.4f}")

    feature_importances(rf_component, FEATURE_NAMES)

    bundle = {
        'model'        : ensemble,
        'rf_component' : rf_component,
        'feature_names': FEATURE_NAMES,
        'accuracy'     : acc,
        'labels'       : {0: 'Safe', 1: 'Phishing'},
    }
    with open(MODEL_SAVE_PATH, 'wb') as f:
        pickle.dump(bundle, f)

    print(f"\n💾  Model saved → {MODEL_SAVE_PATH}")
    print(f"🎉  Done! Accuracy: {acc*100:.1f}%\n")
