import os
import pickle
from preprocess import TextPreprocessor

def test_article():
    # 1. Check if model files exist
    if not os.path.exists('saved_models/model.pkl') or not os.path.exists('saved_models/vectorizer.pkl'):
        print("❌ Error: Saved model files not found. Please run 'python train.py' first!")
        return

    # 2. Load the trained model and vectorizer
    with open('saved_models/model.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('saved_models/vectorizer.pkl', 'rb') as v_file:
        vectorizer = pickle.load(v_file)

    preprocessor = TextPreprocessor()

    print("\n--- 📰 TERMINAL FAKE NEWS TESTER ---")
    print("Type or paste your news text below. Press Enter when done.")
    print("------------------------------------\n")
    
    # 3. Take text input from the user in terminal
    user_input = input("Enter News Text: ")

    if not user_input.strip():
        print("⚠️ Input cannot be empty!")
        return

    # 4. Process and Predict
    cleaned_text = preprocessor.clean(user_input)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized_text)[0]

    # 5. Display Output
    print("\n" + "="*30)
    if prediction.lower() == 'real':
        print("✅ VERDICT: REAL / RELIABLE NEWS")
    else:
        print("🚨 VERDICT: FAKE / UNRELIABLE NEWS")
    print("="*30 + "\n")

if __name__ == "__main__":
    test_article()
