# sentiment_analysis.py
from NewsSentiment import SentimentAnalysisModel

def analyze_sentiment(summaries):
    model = SentimentAnalysisModel()
    filepath = 'all-data.csv'  
    texts, labels = model.load_data(filepath)
    encoded_labels = model.encode_labels(labels)
    
    # Assuming the model is pre-trained or you train it here
    model.train(texts, encoded_labels)

    # Predict sentiment scores for the summaries
    scores = model.get_sentiment_score(model.predict_probs(summaries))
    
    return scores

summaries = ["I love this phone", "This movie is boring"]
sentiment_scores = analyze_sentiment(["Shares of Standard Chartered ( STAN ) rose 1.2 % in the FTSE 100 , while Royal Bank of Scotland ( RBS ) shares rose 2 % and Barclays shares ( BARC ) ( BCS ) were up 1.7 % .", "Tesla is the best car in the world"])
print("Sentiment Scores:", sentiment_scores)
