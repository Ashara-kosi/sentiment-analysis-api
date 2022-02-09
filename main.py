
from fastapi import FastAPI
from markupsafe import string
import uvicorn
import pickle
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# Creating FastAPI instance
app = FastAPI()

# Creating class to define the request body
# and the type hints of each attribute
class request_body(BaseModel):
    feedback : str


with open("clf.pkl", "rb") as f:

    model = pickle.load(f)

@app.get('/')

def index():

    return {'message': 'Omo I just dey try test if e go work first '}

@app.get('/prediction')
async def query_sentiment_analysis(text: str):
    return predict(text)

def predict(text:str):
    # Making the data in a form suitable for prediction
    #convert text to matrix for prediction
    vectorizer = CountVectorizer(lowercase=False)
    x_vec = vectorizer.fit_transform(text)
    #converting the sparse matrix to dense
    X_vec = x_vec.todense()

    tfidf = TfidfTransformer() # by default applies "l2" normalization
    X_tfidf = tfidf.fit_transform(X_vec)
    X_tfidf = X_tfidf.todense()
    
    prediction = model.predict(X_tfidf)
    
    output = int(prediction[0])
    
    sentiments = {0: "Negative", 1: "Positive"}
    return {'prediction': sentiments[output]}

if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
     