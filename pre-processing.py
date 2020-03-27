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

# %% [markdown]
### Read in the data

# %%
train_data = pandas.read_csv("./data/train.csv")
train_data.head()

# %% [markdown]
### Data cleaning

# %%
train_data.columns = ['category', 'headline', 'content']
train_data.head()

# %% [markdown]
### Explore by graphing stuff

# %%
#fig = pyplot.figure()
train_data.groupby('category').headline.count().plot.bar(ylim = 0)
plt.show()

# %%
""" def normalize_document(doc):
    # lower case and remove special characters\whitespaces
    doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = wpt.tokenize(doc)
    # filter stopwords out of document
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # re-create document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc """

# %%
# Sample 1000 rows
train_data_sample = train_data.sample(n = 1000, replace = False, random_state = 123)
print(train_data_sample)

# %%
# TF/IDF
tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 0, norm = 'l2', lowercase = True, 
                        strip_accents = 'ascii', ngram_range = (1, 2), 
                        stop_words = 'english', use_idf = True)
# tfidf = TfidfVectorizer(min_df=0, use_idf=True, lowercase=True, stop_words='english')
features = tfidf.fit_transform(train_data_sample.headline).toarray()
vocab = tfidf.get_feature_names()
pandas.DataFrame(numpy.round(features, 2), columns = vocab)
features.shape


# %%
print(pandas.DataFrame(train_data_sample.groupby(['category']).count()))

# %%
cv = CountVectorizer(min_df = 2, lowercase = True, token_pattern=r'(?u)\b[A-Za-z]+\b', 
                        strip_accents = 'ascii', ngram_range = (1, 1), 
                        stop_words = 'english')
cv_matrix = cv.fit_transform(train_data_sample.headline)
cv_matrix = cv_matrix.toarray()
cv_matrix
# get all unique words in the corpus
vocab = cv.get_feature_names()
# show document feature vectors
cv_matrix_named = pandas.DataFrame(cv_matrix, columns=vocab)

# %%
