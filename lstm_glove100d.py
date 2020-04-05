
import numpy as np
import keras
import keras.preprocessing.text
import nltk
from keras.preprocessing import sequence
from keras.models import Sequential, model_from_json, load_model
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.utils.np_utils import to_categorical
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras import optimizers

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

movies_data=pd.read_csv('movies_metadata.csv')
df=movies_data[['original_title','overview','genres','original_language']].loc[movies_data['original_language']=='en']

def strip_genres(x):
    genres=[]
    for i in eval(x):
        genres.append(i['name'])
    return genres

df['genre']=df['genres'].apply(lambda x : strip_genres(x))
df.drop(['genres','original_language'],axis=1,inplace=True)

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


# function to remove stopwords
def remove_stopwords(text):
    no_stopword_text = [w for w in text.split() if not w in stop_words]
    return ' '.join(no_stopword_text)

df['overview'] = df['overview'].apply(lambda x: remove_stopwords(x))

#tokenize the words with their indices to pass to the model
tk = keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
tk.fit_on_texts(df['overview'])
x_train = tk.texts_to_sequences(df['overview'])


word_index = tk.word_index  # index of unique words
print('Found %s unique tokens.' % len(word_index))

mlb = MultiLabelBinarizer()
y_train  = mlb.fit_transform(df['genre'])



max_len = 64     #length of sequence
batch_size = 32
epochs = 64
max_features = len(word_index) + 1   # (number of words in the vocabulary) + 1

x_train = sequence.pad_sequences(x_train, maxlen=max_len, padding='pre')

xtrain, xtest, ytrain, ytest = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print('x_train shape:', xtrain.shape)
print('x_test shape:', xtest.shape)

print('y_train shape:', ytrain.shape)
print('y_test shape:', ytest.shape)
label_num = len(y_train[0])

#getting the glove embeddingsm
embeddings_index = {}
f = open('glove.6B.100d.txt',encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    emb_weights = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = emb_weights
f.close()
print('Found %s word vectors.' % len(embeddings_index))

#defining the learnt matrix to pass to the embedding the layer weights
embedding_matrix = np.zeros((max_features, 100))

#the words not in the embedding matrix will be zeros
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

max_len = 64       #length of sequence
batch_size = 32
epochs = 64

#initating the weights of the embedding layer
embedding_layer = Embedding(input_dim = max_features,
                            output_dim = 100,
                            weights=[embedding_matrix],
                            mask_zero = True,
                            input_length = max_len,
                            trainable = False)

# building the model and compiling it

model = Sequential()
model.add(embedding_layer)
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64,return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.3))
model.add(Dense(label_num, activation = 'sigmoid'))

print(model.summary())

#rmsprop = optimizers.RMSprop(lr = 0.01, decay = 0.0001)
model.compile(optimizer ='adam', loss = 'binary_crossentropy', metrics=['accuracy'])


# fitting the data on the model and saving the weights obtained

model.fit(xtrain, ytrain, epochs=epochs, batch_size= batch_size,validation_split=0.2,verbose=1,validation_data=[xtest, ytest])

#model.save_weights('my_model_weights.h5')
model.save('bi-lstm-glove.h5')


output = model.predict(xtest)
output = np.array(out)

y_pred = np.zeros(out.shape)
#setting a threshold of 0.5
y_pred[output>0.5]=1
y_pred = np.array(y_pred)

print(mlb.inverse_transform(y_pred)[0])

f1 = metrics.f1_score(ytest,y_pred, average = 'micro')
print(f1)



'''# Creating a reverse dictionary
reverse_word_map = dict(map(reversed, tk.word_index.items()))

# Function takes a tokenized sentence and returns the words
def sequence_to_text(list_of_indices):
    # Looking up words in dictionary
    words = [reverse_word_map.get(letter) for letter in list_of_indices]
    return(words)

# Creating texts 
my_texts = list(map(sequence_to_text, xtest))'''


#printing the first ten test results
df_xtest1=df_xtest.copy().reset_index()
for i in range(10):
    print('Title:', df_xtest1.loc[i,'original_title'])
    print('Actual Genres:',mlb.inverse_transform(ytest)[i])
    print('Predicted Genres:',mlb.inverse_transform(y_pred)[i])
    print('-----------')





