import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import TextPreprocessor

def run_training_pipeline():
    print("📦 Extracting unified structural text arrays...")
    if not os.path.exists('data/fake_or_real_news.csv'):
        print("❌ Error: Target file missing. Run merge_data.py first!")
        return

    df = pd.read_csv('data/fake_or_real_news.csv')
    df = df.dropna(subset=['title', 'text', 'label'])
    df['total_text'] = df['title'] + " " + df['text']
    
    print("🧹 Cleaning tokens (Removing dataset-specific publisher bias)...")
    preprocessor = TextPreprocessor()
    df['total_text'] = df['total_text'].apply(preprocessor.clean)
    
    # Stratified 80-20 partition configuration
    X_train, X_test, y_train, y_test = train_test_split(
        df['total_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    print("📊 Generating analytical TF-IDF numerical space matrix...")
    vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, stop_words='english', ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("🏋️ Training Logistic Regression Classifier (With Probability Vector Mapping)...")
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Model evaluation metrics derivation
    y_pred = model.predict(X_test_tfidf)
    print("\n" + "="*20 + " FINAL MODEL METRICS " + "="*20)
    print(f"System Accuracy Score: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("\nLinguistic Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("="*61 + "\n")
    
    # Serialize assets
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('saved_models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("💾 Production weights successfully serialized inside saved_models/")

if __name__ == "__main__":
    run_training_pipeline()
