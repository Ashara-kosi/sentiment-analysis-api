
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import confusion_matrix,accuracy_score
from data_process import getdata
import pickle







tweets = getdata(r'./data.csv')

tweets = tweets[['Sentiment','new_text']]
data_drop = tweets.loc[tweets["Sentiment"] == "Neutral" ]
tweets.drop(tweets[tweets['Sentiment'] == 'Neutral'].index, inplace = True)
tweets['Sentiment'].replace(to_replace={'Positive':0,'Negative':1},inplace = True)
y = tweets.iloc[:5000,0]
x = tweets.iloc[:5000,1]


#using count vectorizer to convert the text to matrix
vectorizer = CountVectorizer(lowercase=False)
x_vec = vectorizer.fit_transform(x)

#converting the sparse matrix to dense
X_vec = x_vec.todense()

tfidf = TfidfTransformer() # by default applies "l2" normalization
X_tfidf = tfidf.fit_transform(X_vec)
X_tfidf = X_tfidf.todense()
X_tfidf

#splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

    

#testing the complementnb model
clf = ComplementNB()
clf.fit(X_train, y_train)



# Save the vectorizer
vec_file = 'vectorizer.pickle'
pickle.dump(vectorizer, open(vec_file, 'wb'))



y_pred = clf.predict(X_test)

confusion_matrix(y_test, y_pred)

print("Number of mislabeled points out of a total %d points : %d"
       % (X_test.shape[0], (y_test != y_pred).sum()))
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

#saving the model
pickle_out = open("clf.pkl", "wb") 
pickle.dump(clf, pickle_out) 
pickle_out.close()









