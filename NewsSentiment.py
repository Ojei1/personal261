# NewsSentiment.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class SentimentAnalysisModel:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.pipeline = make_pipeline(TfidfVectorizer(), LogisticRegression())

    def load_data(self, filepath):
        df = pd.read_csv(filepath, header=None, encoding='ISO-8859-1', names=['label', 'text'])
        labels = df['label'].values
        texts = df['text'].values
        return texts, labels

    def encode_labels(self, labels):
        return self.label_encoder.fit_transform(labels)
    
    def train(self, X_train, y_train): 
        self.pipeline.fit(X_train, y_train)

    def predict(self, X_test):
        return self.pipeline.predict(X_test)

    def predict_probs(self, X_test):
        return self.pipeline.predict_proba(X_test)

    def get_sentiment_score(self, probs):
        scores = []
        for prob in probs:
            if prob.argmax() == 0:
                score = 1 + prob[0] * 2
            elif prob.argmax() == 1:
                score = 4 + prob[1] * 2
            else:
                score = 8 + prob[2] * 3
            scores.append(int(score))
        return scores
