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
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

"""====================================================================="""
# %% [markdown]
# ## Read in the data

# %%
train_data = pandas.read_csv("./data/train.csv")
train_data.head()

"""====================================================================="""
# %% [markdown]
# ## Data Cleaning

# %%
train_data.columns = ['category', 'headline', 'content']
train_data.head()

"""====================================================================="""
# %% [markdown]
# ### Sample 1000 rows

# %%
train_data_sample = train_data.sample(n = 1000, replace = False, random_state = 123)
train_data_sample.head()

# %%
# Clean HTML code & news sources from headline
import re

def clean(x):
    x = re.sub(r'(&[A-Za-z]+)|\(.*\)', '', x)
    return str(x)

for i, row in train_data_sample.iterrows():
    train_data_sample.at[i, "headline"] = clean(row.headline)

# %%
# clean news sources from content
sources_data = pandas.read_csv("./data/news_sources_clean_v1.csv")

def remove_sources(x):
    x = str(x)
    # print('X OUTSIDE OF LOOP:' + x)
    for i, source in sources_data.iterrows():
        source_list_string = str(sources_data.at[i, 'list'])
        #print('source_list_string:' + source_list_string)
        source_list_stripped = source_list_string.strip()
        #print('source_list_stripped:' + source_list_stripped)
        
        if source_list_stripped in x:
            # print('x at this point:' + x)
            # print('source_list_stripped:' + source_list_stripped)
            # print('row number: ' + str(i))
            
            #this doesn't work
            x = x.replace(source_list_stripped, '')
            #regex_expression = re.compile(source.list)
            #x = re.sub(regex_expression, '', x)
    return x

for i, row in train_data_sample.iterrows():
    train_data_sample.at[i, "content_cleaned"] = remove_sources(row.content)


# %%
# create a CountVectorizer from raw data, with options to clean it
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
# ## Data Exploration

# %%
# bar plot of the count of unique things in each category
train_data.groupby('category').headline.count().plot.bar(ylim = 0)
plt.title("Category count raw data")
plt.show()
train_data_sample.groupby('category').headline.count().plot.bar(ylim = 0)
plt.title("Category count sample data")
plt.show()

# %%
# print out the number of unique things in each category
print(pandas.DataFrame(train_data_sample.groupby(['category']).count()))

# %%
# print a count of observations and features
print("There are {} observations and {} features in this dataset. \n".\
    format(cv_matrix_df.shape[0],cv_matrix_df.shape[1]))

# %%
# print a description of the categories
categories = train_data_sample.groupby("category")
categories.describe().head()

# %% [markdown]
# ### WordCloud/TagCloud of the top words in the headlines

# %%
# prepare the dictionary to be used in wordcloud
word_count_dict = {}
for word in vocab:
    word_count_dict[word] = int(sum(cv_matrix_df.loc[:, word]))

# %%
# generate a word cloud image with top 100 words and 80% horizontal:
wordcloud = WordCloud(max_words=100, prefer_horizontal=0.8, background_color='white').\
            generate_from_frequencies(word_count_dict)

# display the generated image:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# %% [markdown]
# ### Plots of the data

# bar plot of the top word counts

from collections import Counter

counter = Counter(word_count_dict)

freq_df = pandas.DataFrame.from_records(counter.most_common(20),
                                        columns=['Top 20 words', 'Frequency'])
freq_df.plot(kind='bar', x='Top 20 words');


# %%
# table of the top 10 words
#vocab_df = pandas.DataFrame(vocab)
#vocab_df.groupby()

"""====================================================================="""
# %% [markdown]
### TF/IDF

# # %%
# tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 0, norm = 'l2', lowercase = True, 
#                         strip_accents = 'ascii', ngram_range = (1, 2), 
#                         stop_words = 'english', use_idf = True, token_pattern=r'(?u)\b[A-Za-z]{2,}\b')
# # tfidf = TfidfVectorizer(min_df=0, use_idf=True, lowercase=True, stop_words='english')
# features = tfidf.fit_transform(train_data_sample.headline).toarray()
# pandas.DataFrame(numpy.round(features, 2), columns = vocab)
# features.shape()