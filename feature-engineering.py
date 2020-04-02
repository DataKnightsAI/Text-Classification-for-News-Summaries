# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Import necessary dependencies

# %%
import pandas
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy
from sklearn.feature_selection import chi2
from PIL import Image
from collections import Counter
import re
import sqlite3

# %% [markdown]
# ## Load in the data from the database

# %%
dbconn = sqlite3.connect('./data/newsclassifier.db')
train_data_df = pandas.read_sql_query('SELECT * FROM train_data_sample', dbconn)
headline_bagofwords_df = pandas.read_sql_query('SELECT * FROM headline_bagofwords', dbconn)
dbconn.commit()
dbconn.close()

# %% [markdown]
# ### Check the if the data was loaded correctly

# %%
train_data_df.head()


# %%
headline_bagofwords_df.head()


# %%
train_data_df.drop('index', axis=1, inplace=True)
train_data_df.head()


# %%
headline_bagofwords_df.drop('index', axis=1, inplace=True)
headline_bagofwords_df.head()


# %% [markdown]
# ### We have bag of words already, let's make a Bag of N-Grams
# %%
# Use countvectorizer to get a word vector
cv = CountVectorizer(min_df = 2, lowercase = True, token_pattern=r'(?u)\b[A-Za-z]{2,}\b', 
                        strip_accents = 'ascii', ngram_range = (2, 3), 
                        stop_words = 'english')
cv_matrix = cv.fit_transform(train_data_df.headline_cleaned).toarray()

# below is if wanted to define a specific category for the data.
# cv_matrix = cv.fit_transform(train_data_df[train_data_df.category == 1].headline_cleaned).toarray()

# get all unique words in the corpus
vocab = cv.get_feature_names()

# produce a dataframe including the feature names
headline_bagofngrams_df = pandas.DataFrame(cv_matrix, columns=vocab)

# %% [markdown]
# ### Make sure we got the dataframe output for the Bag of N-Grams

# %%
headline_bagofngrams_df.head()

# %% [markdown]
# ### Let's explore the data we got through plots and tables

# %%
word_count_dict = {}
for word in vocab:
    word_count_dict[word] = int(sum(headline_bagofngrams_df.loc[:, word]))

counter = Counter(word_count_dict)

freq_df = pandas.DataFrame.from_records(counter.most_common(20),
                                        columns=['Top 20 words', 'Frequency'])
freq_df.plot(kind='bar', x='Top 20 words');


# %% [markdown]
# ## TF/IDF

# %% [markdown]
# ### Unigram TF/IDF

# %%
tfidf_vect = TfidfVectorizer(sublinear_tf = True, min_df = 2, lowercase = True, 
                             strip_accents = 'ascii', ngram_range = (1, 1), 
                             stop_words = 'english', use_idf = True, token_pattern=r'(?u)\b[A-Za-z]{2,}\b')
tfidf_unigram = tfidf_vect.fit_transform(train_data_df.headline_cleaned).toarray()

# get all unique words in the corpus
vocab = tfidf_vect.get_feature_names()

tfidf_unigram = pandas.DataFrame(numpy.round(tfidf_unigram, 2), columns = vocab)
tfidf_unigram.head()

# %% [markdown]
# ### N-Gram TF/IDF

# %%
tfidf_vect = TfidfVectorizer(sublinear_tf = True, min_df = 2, lowercase = True, 
                             strip_accents = 'ascii', ngram_range = (2, 3), 
                             stop_words = 'english', use_idf = True, token_pattern=r'(?u)\b[A-Za-z]{2,}\b')
tfidf_ngram = tfidf_vect.fit_transform(train_data_df.headline_cleaned).toarray()

# get all unique words in the corpus
vocab = tfidf_vect.get_feature_names()

tfidf_ngram = pandas.DataFrame(numpy.round(tfidf_ngram, 2), columns = vocab)
tfidf_ngram.head()

# %% [markdown]
# ### Character TF/IDF

# %%
tfidf_vect = TfidfVectorizer(analyzer = 'char', sublinear_tf = True, min_df = 2, 
                             lowercase = True, strip_accents = 'ascii', ngram_range = (2, 3), 
                             stop_words = 'english', use_idf = True, token_pattern=r'\w{1,}')
tfidf_char = tfidf_vect.fit_transform(train_data_df.headline_cleaned).toarray()

# get all unique words in the corpus
vocab = tfidf_vect.get_feature_names()

tfidf_char = pandas.DataFrame(numpy.round(tfidf_char, 2), columns = vocab)
tfidf_char.head()

# %%
word_count_dict = {}
for word in vocab:
    word_count_dict[word] = int(sum(tfidf_char.loc[:, word]))

counter = Counter(word_count_dict)

freq_df = pandas.DataFrame.from_records(counter.most_common(50),
                                        columns=['Top 50 words', 'Frequency'])
freq_df.plot(kind='bar', x='Top 50 words');

# %%
