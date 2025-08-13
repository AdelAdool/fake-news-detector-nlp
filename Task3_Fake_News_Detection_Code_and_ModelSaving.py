# Fake News Detection 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
import joblib
from sklearn.pipeline import Pipeline

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set random seed for reproducibility
np.random.seed(42)

# ========================
# 1. Data Loading
# ========================
print("Loading datasets...")
fake_path = r"C:\Users\LEGION\Desktop\Interships\Elevvo Internship\Tasks\Task3\DataSet\Fake.csv"
real_path = r"C:\Users\LEGION\Desktop\Interships\Elevvo Internship\Tasks\Task3\DataSet\True.csv"

# Load and combine datasets
fake_df = pd.read_csv(fake_path)
real_df = pd.read_csv(real_path)

# Add labels and combine
fake_df['label'] = 0  # 0 for fake
real_df['label'] = 1   # 1 for real
df = pd.concat([fake_df, real_df]).sample(frac=1).reset_index(drop=True)

print(f"Dataset loaded with {len(df)} samples")
print(f"Fake news count: {len(df[df['label']==0])}")
print(f"Real news count: {len(df[df['label']==1])}")

# ========================
# 2. Text Preprocessing
# ========================
print("\nPreprocessing text...")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs, mentions, and hashtags
    text = re.sub(r'http\S+|www\S+|https\S+|\@\w+|\#', '', text)
    
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    
    # Tokenize and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    
    return ' '.join(tokens)

# Apply preprocessing
df['cleaned_text'] = df['text'].apply(preprocess_text)
df['cleaned_title'] = df['title'].apply(preprocess_text)
df['combined'] = df['cleaned_title'] + ' ' + df['cleaned_text']

# ========================
# 3. Feature Extraction
# ========================
print("\nCreating features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = tfidf.fit_transform(df['combined'])
y = df['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ========================
# 4. Model Training
# ========================
print("\nTraining models...")

# Logistic Regression
lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# SVM
svm = SVC(kernel='linear', C=1.0, probability=True)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# ========================
# 5. Evaluation
# ========================
def evaluate_model(y_true, y_pred, model_name):
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.show()

evaluate_model(y_test, lr_pred, "Logistic Regression")
evaluate_model(y_test, svm_pred, "SVM")

# ========================
# 6. Visualization
# ========================
print("\nGenerating word clouds...")

def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=500, 
                         background_color='white').generate(text)
    plt.figure(figsize=(10,6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Generate word clouds
fake_text = " ".join(df[df['label']==0]['combined'])
real_text = " ".join(df[df['label']==1]['combined'])

generate_wordcloud(fake_text, "Most Common Words in Fake News")
generate_wordcloud(real_text, "Most Common Words in Real News")

# ========================
# 7. Save Model
# ========================
print("\nSaving model pipeline...")
model_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('model', lr)  # Saving logistic regression as it usually performs well
])

# Create directory if it doesn't exist
import os
os.makedirs('saved_models', exist_ok=True)

# Save the pipeline
joblib.dump(model_pipeline, 'saved_models/fake_news_pipeline.pkl')
print("Model pipeline saved to 'saved_models/fake_news_pipeline.pkl'")

print("\nTraining completed successfully!")