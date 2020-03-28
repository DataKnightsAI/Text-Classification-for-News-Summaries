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
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

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
train_data_sample.head()

# %%
cv = CountVectorizer(min_df = 2, lowercase = True, token_pattern=r'(?u)\b[A-Za-z]{2,}\b', 
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
# bar plot of the count of unique things in each category
train_data.groupby('category').headline.count().plot.bar(ylim = 0)
plt.show()

# %%
# print out the number of unique things in each category
print(pandas.DataFrame(train_data_sample.groupby(['category']).count()))

# %%
print("There are {} observations and {} features in this dataset. \n".\
    format(cv_matrix_df.shape[0],cv_matrix_df.shape[1]))

# %%
# print a description of the categories
categories = train_data_sample.groupby("category")
categories.describe().head()

# %%
# prepare the dictionary to be used in
# word_count = []
text = {}
for word in vocab:
    # word_count.append(sum(cv_matrix_df.loc[:, word]))
    text[word] = int(sum(cv_matrix_df.loc[:, word]))

# %%
# generate a word cloud image with top 50 words and 80% horizontal:
wordcloud = WordCloud(max_words=50, prefer_horizontal=0.8).\
            generate_from_frequencies(text)

# display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# %%
# table of the top 10 words
#vocab_df = pandas.DataFrame(vocab)
#vocab_df.groupby()

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