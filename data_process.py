import pandas as pd
import numpy as np
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *

# nltk.download('stopwords')


def getdata(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()
    # data = data['new_text']
    return data

data = getdata(r'C:\Users\ASHAROX\Downloads\cleaned_text.csv')
# tweets['clean_tweet'] = tweets.apply(preprocess_tweet,axis=1)

#print (data)
# nltk.download('vader_lexicon')

#Sentiment Analysis
nltk.download('vader_lexicon')
SIA = SentimentIntensityAnalyzer()
data["new_text"]= data["new_text"].astype(str)
# Applying Model, Variable Creation
data['Polarity Score']=data['new_text'].apply(lambda x:SIA.polarity_scores(x)['compound'])
data['Neutral Score']=data['new_text'].apply(lambda x:SIA.polarity_scores(x)['neu'])
data['Negative Score']=data['new_text'].apply(lambda x:SIA.polarity_scores(x)['neg'])
data['Positive Score']=data['new_text'].apply(lambda x:SIA.polarity_scores(x)['pos'])
# Converting 0 to 1 Decimal Score to a Categorical Variable
data['Sentiment']=''
data.loc[data['Polarity Score']>0,'Sentiment']='Positive'
data.loc[data['Polarity Score']==0,'Sentiment']='Neutral'
data.loc[data['Polarity Score']<0,'Sentiment']='Negative'

print(data)
data.to_csv('data.csv')
# print(tweets.head())



