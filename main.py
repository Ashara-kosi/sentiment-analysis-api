
from fastapi import FastAPI
import uvicorn
import pickle
from pydantic import BaseModel



# Creating FastAPI instance
app = FastAPI()

# Creating class to define the request body
# and the type hints of each attribute

with open("clf.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pickle", "rb") as f:
    vec = pickle.load(f)
class request_body(BaseModel):
    text: str
    
@app.get('/')
def index():
    return {'message': 'Omo I just dey try test if e go work first '}

@app.post('/prediction')
async def query_sentiment_analysis(text: str):
    return predict([text])

def predict(text: str):
    # # Making the data in a form suitable for prediction
    # #convert text to matrix for prediction
    # vectorizer = CountVectorizer(lowercase=False)
    # x_vec = vectorizer.fit_transform(text)
    # #converting the sparse matrix to dense
    # X_vec = x_vec.todense()

    # tfidf = TfidfTransformer() # by default applies "l2" normalization
    # X_tfidf = tfidf.fit_transform(X_vec)
    # X_tfidf = X_tfidf.todense()
    prediction = model.predict(vec.transform(text))

    
    
    output = int(prediction[0])
    
    sentiments = {0: "Positive text", 1: "Negative text"}

    return {'prediction': sentiments[output]}

    if __name__ == '__main__':
        uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)

testing = predict(["I love you"])
print(testing)