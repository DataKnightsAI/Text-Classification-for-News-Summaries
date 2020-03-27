# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
### Import necessary dependencies

# %%
import pandas
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy
from sklearn.feature_selection import chi2

"""====================================================================="""
# %% [markdown]
### Read in the data

# %%
train_data = pandas.read_csv("./data/train.csv")
train_data.head()

"""====================================================================="""
# %% [markdown]
### Data Cleaning

# %%
train_data.columns = ['category', 'headline', 'content']
train_data.head()

"""====================================================================="""
# %% [markdown]
#### Sample 1000 rows

# %%
train_data_sample = train_data.sample(n = 1000, replace = False, random_state = 123)
print(train_data_sample.head())

# %%
cv = CountVectorizer(min_df = 2, lowercase = True, token_pattern=r'(?u)\b[A-Za-z]+\b', 
                        strip_accents = 'ascii', ngram_range = (1, 1), 
                        stop_words = 'english')
cv_matrix = cv.fit_transform(train_data_sample.headline).toarray()

# get all unique words in the corpus
vocab = cv.get_feature_names()

# produce a dataframe including the feature names
cv_matrix_df = pandas.DataFrame(cv_matrix, columns=vocab)

"""====================================================================="""
# %% [markdown]
### Data Exploration

# %%
train_data.groupby('category').headline.count().plot.bar(ylim = 0)
plt.show()

# %%
print(pandas.DataFrame(train_data_sample.groupby(['category']).count()))

"""====================================================================="""
# %% [markdown]
### TF/IDF

# %%
tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 0, norm = 'l2', lowercase = True, 
                        strip_accents = 'ascii', ngram_range = (1, 2), 
                        stop_words = 'english', use_idf = True)
# tfidf = TfidfVectorizer(min_df=0, use_idf=True, lowercase=True, stop_words='english')
features = tfidf.fit_transform(train_data_sample.headline).toarray()
vocab = tfidf.get_feature_names()
pandas.DataFrame(numpy.round(features, 2), columns = vocab)
features.shape()