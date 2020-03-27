# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %% [markdown]
### Import necessary dependencies

# %%
import pandas
from matplotlib import pyplot
import sklearn.feature_extraction.text
from sklearn.feature_selection import chi2
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
tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 1, norm = 'l2', lowercase = True, 
                        strip_accents = 'ascii', ngram_range = (1, 2), stop_words = 'english')
features = tfidf.fit_transform(train_data_sample).toarray()
labels = train_data_sample.category
features.shape


# %%
print(features)

# %%
