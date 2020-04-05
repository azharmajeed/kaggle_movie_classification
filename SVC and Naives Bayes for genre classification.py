#In this script I have used two ML models for movie genre classification. One is Linear SVC and the other is Multinomial Navies Bayes. For the SVC model
#I preprocess the text by a statistical approach using tfidf to get the importance of a word in the corpus. For the MNB model I use a count vectorizer.


import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt 
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import json
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 20000000)


# In[151]:
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
stop_words = set(stopwords.words('english'))

movies_data=pd.read_csv('movies_metadata.csv')
df=movies_data[['original_title','overview','genres','original_language']].loc[movies_data['original_language']=='en']


#extract genres from the dictionary
def strip_genres(x):
    genres=[]
    for i in eval(x):
        genres.append(i['name'])
    return genres

df['genre']=df['genres'].apply(lambda x : strip_genres(x))
df.drop(['genres','original_language'],axis=1,inplace=True)
df.head()

# get all genre tags in a list
all_genres = sum(df.genre,[])
print(len(set(all_genres)))

def plot_genres_dist(all_genres):
    all_genres = nltk.FreqDist(all_genres)

    # create dataframe
    all_genres_df = pd.DataFrame({'Genre': list(all_genres.keys()),
                                  'Count': list(all_genres.values())})

    g = all_genres_df.nlargest(columns="Count", n = 50)
    plt.figure(figsize=(12,15))
    ax = sns.barplot(data=g, x= "Count", y = "Genre")
    ax.set(ylabel = 'Count')
    plt.show()


# function for text cleaning 
def clean_text(text):
    # remove everything except alphabets 
    text = re.sub("[^a-zA-Z]"," ",text) 
    # remove whitespaces 
    text = ' '.join(text.split()) 
    # convert text to lowercase 
    text = text.lower() 
    
    return text

df['overview'] = df['overview'].apply(lambda x: clean_text(str(x)))

def freq_words(x, terms = 30): 
  all_words = ' '.join([text for text in x]) 
  all_words = all_words.split() 
  fdist = nltk.FreqDist(all_words) 
  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
  
  # selecting top 20 most frequent words 
  d = words_df.nlargest(columns="count", n = terms) 
  
  # visualize words and frequencies
  plt.figure(figsize=(12,15)) 
  ax = sns.barplot(data=d, x= "count", y = "word") 
  ax.set(ylabel = 'Word') 
  plt.show()
  
# print 100 most frequent words 
freq_words(df['overview'], 100)
plot_genres_dist(all_genres)


# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

df['overview'] = df['overview'].apply(lambda x: remove_stopwords(x))

freq_words(df['overview'], 100)


from sklearn.preprocessing import MultiLabelBinarizer

multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(df['genre'])

# transform target variable
y = multilabel_binarizer.transform(df['genre'])

#initiate a tfidf vectorizer object
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)

# split dataset into training and validation set
xtrain, xtest, ytrain, ytest = train_test_split(df['overview'], y, test_size=0.2, random_state=42)


# create TF-IDF features
xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
xval_tfidf = tfidf_vectorizer.transform(xtest)

#build a linear one vs rest svc model
svc = LinearSVC()
clf = OneVsRestClassifier(svc)
# fit model on train data
clf.fit(xtrain_tfidf, ytrain)

# make predictions for validation set
y_pred_svc = clf.predict(xval_tfidf)

# evaluate performance
print(f1_score(ytest, y_pred_svc, average="micro"))

def infer_tags(model,prep,q):
    q = clean_text(q)
    q = remove_stopwords(q)
    q_vec = prep.transform([q])
    q_pred = model.predict(q_vec)
    return multilabel_binarizer.inverse_transform(q_pred)

#print examples
for i in range(10):
  k = xtest.sample(10).index[0]
  print('predictions for Linear SVC model')
  print("Movie: ", df['original_title'][k], "\nPredicted genre: ", infer_tags(clf,tfidf_vectorizer,xtest[k])),
  print("Actual genre: ",df['genre'][k], "\nDescription:",df['overview'][k])


# build a naives bayes model by passing the input as count of occurence of the word
count_vec=CountVectorizer()
xtrain_cv = count_vec.fit_transform(xtrain)
xval_cv = count_vec.transform(xtest)


mnb_clas=MultinomialNB()
nb_ovr=OneVsRestClassifier(mnb_clas)
nb_ovr.fit(xtrain_cv,ytrain)


# make predictions for validation set
y_pred_nb = nb_ovr.predict(xval_cv)
print(f1_score(ytest, y_pred_nb, average="micro"))


for i in range(10): 
  k = xtest.sample(10).index[0]
  print('Predictions for Nultinomial Naives Bayes model')
  print("Movie: ", df['original_title'][k], "\nPredicted genre: ", infer_tags(nb_ovr,count_vec,xtest[k])),
  print("Actual genre: ",df['genre'][k], "\nDescription:",df['overview'][k])





