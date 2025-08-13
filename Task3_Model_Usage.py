# Fake News Detector - Interactive Version
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and prepare text identical to training"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+|\@\w+|\#', '', text)  # Remove URLs/social tags
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def load_model(model_path='saved_models/fake_news_pipeline.pkl'):
    """Load the pre-trained pipeline"""
    try:
        pipeline = joblib.load(model_path)
        print("\nModel loaded successfully!")
        return pipeline
    except Exception as e:
        print(f"\nError loading model: {e}")
        return None

def predict_news(text, pipeline):
    """Make prediction on new text"""
    cleaned_text = preprocess_text(text)
    prediction = pipeline.predict([cleaned_text])
    proba = pipeline.predict_proba([cleaned_text])[0]
    
    return {
        'prediction': 'REAL NEWS' if prediction[0] == 1 else 'FAKE NEWS',
        'confidence': float(max(proba)),
        'fake_prob': float(proba[0]),
        'real_prob': float(proba[1])
    }

def display_result(result, text):
    """Display prediction results with visual formatting"""
    print("\n" + "="*50)
    print("PREDICTION RESULTS:".center(50))
    print("="*50)
    print(f"\nAnalyzed Text:\n{text[:500]}...\n")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print("\nProbability Breakdown:")
    print(f"- Fake News Probability: {result['fake_prob']:.1%}")
    print(f"- Real News Probability: {result['real_prob']:.1%}")
    print("="*50 + "\n")

def main():
    """Main interactive function"""
    print("\n" + "="*50)
    print("FAKE NEWS DETECTOR".center(50))
    print("="*50)
    
    # Load model
    model = load_model()
    if not model:
        return
    
    while True:
        print("\nOptions:")
        print("1. Enter text to analyze")
        print("2. Exit")
        choice = input("\nEnter your choice (1/2): ").strip()
        
        if choice == '1':
            text = input("\nEnter the news text you want to analyze:\n> ")
            if not text.strip():
                print("\nPlease enter some text to analyze!")
                continue
                
            result = predict_news(text, model)
            display_result(result, text)
            
        elif choice == '2':
            print("\nThank you for using the Fake News Detector!")
            break
            
        else:
            print("\nInvalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()