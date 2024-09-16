"""
Setting up the Environment

python -m venv nlp_env
source nlp_env/bin/activate
pip install numpy pandas matplotlib nltk torch
"""

import re

import matplotlib.pyplot as plt
# Imports
import pandas as pd
import seaborn as sb
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud

#Reading the dataset downloaded from the Crawler
csvFile = pd.read_csv("a.csv")

#Text preprocessing

text = "Call me Ishmael. Some years ago - never mind how long precisely - having little or no money in my purse, and nothing particular..."
punctuation_free_text = re.sub(r'[^\w\s]', '', text)

text = "2:43PM: It is announced that flight EK-712 is delayed due to the..."
text = text.lower()
non_numeric_text = re.sub(r'\d+', '', text)

tokens = word_tokenize("Out Stealing Horses has been embraced across the world as a classic, a novel of universal relevance and power.")
filtered_tokens = list(filter(lambda word: word.lower() not in stopwords, tokens))

stemmer = PorterStemmer()
stem = stemmer.stem("playing")
stem = stemmer.stem("happiness")

lemmatizer = WordNetLemmatizer()
lemma = lemmatizer.lemmatize("happiness", pos=wordnet.NOUN)

analyzer = SentimentIntensityAnalyzer()
score = analyzer.polarity_scores("We regret to inform you that the request product is unavailable.")

score = analyzer.polarity_scores("It was a sunny morning of March with flowers blossoming everywhere.")

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return 'positive'
    elif score['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

df['Sentiment'] = df['full_text'].apply(get_sentiment)

randomState = 5

X_train, X_test, y_train, y_test = train_test_split(df['full_text'],df['sentiment'], test_size=0.2, random_state=randomState)

train_df = pd.DataFrame({'full_text': X_train, 'sentiment': y_train})
test_df = pd.DataFrame({'full_text': X_test, 'sentiment': y_test})

tfidf_vectorizer = TfidfVectorizer()

tfidf_features_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_features_test = tfidf_vectorizer.transform(X_test)

count_vectorizer = CountVectorizer()

cv_features_train = count_vectorizer.fit_transform(X_train)
cv_features_test = count_vectorizer.transform(X_test)

svm_model = SVC(kernel='rbf', random_state=randomState)

svm_model.fit(tfidf_features_train, y_train)

y_pred = svm_model.predict(tfidf_features_test)
accuracy = accuracy_score(y_test, y_pred)

df['compound_sentiment_score'] = df['full_text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

plt.figure(figsize=(8, 5))
sb.histplot(df['compound_sentiment_score'], bins=30, kde=True, color='yellow')
plt.title('Sentiment Scores Distribution')
plt.xlabel('Compound Score')
plt.ylabel('Frequency')
plt.show()

excluded_words = {'https', 't', 'co', 'S'}
stopwords = set(WordCloud().stopwords)
stopwords.update(excluded_words)

text_for_sentiment = df[df['sentiment'] == 'positive']['full_text'].str.cat(sep=' ')
wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text_for_sentiment)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Positive Sentiments')
plt.show()