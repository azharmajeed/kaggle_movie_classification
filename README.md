# kaggle_movie_classification
Movie Genre Classification
 
This project's goal is to build and compare results of multiple machine learning models to classify movie's genres based on their plot overviews. The dataset can be found in the following link.(https://www.kaggle.com/rounakbanik/the-movies-dataset). The scripts contain all the steps from data visualization, text preprocessing, model building, prediction and evaluation. 

The first two model can be imported from the sci-kit library(https://scikit-learn.org/stable/supervised_learning.html#supervised-learning). Since this is a multi-label classification I used a OneVsRest approach where one model will be build per label and their combining their resuls to get the final output. In the first script 'SVC and Naives Bayes for genre classification.py'-
1. The first model used was a Linear SVC model where I preporcessed the text by converting a sentence to it's corresponing TF-IDF vector( which represents the importance of a word in the whole corpus of text).
2. The other model is a Naives Bayes model, here I preprocessed the text using a CountVectorizer.

In the second script 'lstm_glove100d.py'-
I used a deep learning approach, where I used Keras Bidirectional LSTM(https://keras.io/examples/imdb_bidirectional_lstm/) with the help pretrained word embeddings from GLOVE (https://nlp.stanford.edu/projects/glove/). I used their most basic word embeddings with a dimension of 100.


Results-
I used F1-score as a way to evalute my model perfomances. The deep learning approach had the best model with a score of 54, next was the Naives Bayes modeel with a score of 51 and the SVC model scored at 49. 
Note- I have not performed any type of hyper-parameter tuning yet, so these baseline model scores can be improved. Another thing to take in to consideration is that training the LSTM took nearly 1-2 hours to train(using a CPU), whereas the other two approaches were much more faster(like they were build instantaneously) but the LSTM would perform better if we feed it more data as Deep learning models are know to perform better on larger datasets.

Future Scope-
Another approach to look out for would be to use transform based models that use attention like Bert (https://github.com/google-research/bert) since it has recenlty been one of the best performers on the GLUE benchmark.



