# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
### Import necessary dependencies

# %%
import pandas
from matplotlib import pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.feature_selection
import numpy

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
pyplot.show()

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
print(features)

# %%
