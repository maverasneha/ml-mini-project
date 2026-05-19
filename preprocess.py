import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Silently prepare NLP dependencies
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        if not isinstance(text, str):
            return ""
        
        # Lowercase normalization
        text = text.lower()
        
        # Strip Reuters or location dateline anchors to remove source bias
        text = re.sub(r'^.*?\(reuters\)\s*-\s*', '', text)
        text = re.sub(r'^[a-z\s\.,]+–\s*', '', text) 
        
        # Clean URLs, HTML tags, punctuation, and structural digits
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<[^>]*>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Token clean and morphological reduction
        words = text.split()
        cleaned_words = [
            self.lemmatizer.lemmatize(word) 
            for word in words 
            if word not in self.stop_words
        ]
        
        return ' '.join(cleaned_words)
