# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Data Cleaning and Data Exploration 

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
from collections import Counter
import re
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


# %% [markdown]
# ## Read in the data

# %%
train_data = pandas.read_csv('./data/train.csv', header=None)
train_data.head()

# %%
test_data = pandas.read_csv("./data/test.csv", header=None)
test_data.head()

# %% [markdown]
# ## Data Cleaning

# %%
# ### Add headers to the data
train_data.columns = ['category', 'headline', 'content']
train_data.head()

# %%
test_data.columns = ['category', 'headline', 'content']
test_data.head()

# %% [markdown]
# ### Clean HTML code & news sources from headline

# %%
# Define a clean function: lowercase, strip HTML, punctuations, non-alpha, stopwords
def clean(x):
    # strip HTML and sources of the format eg. "&lt and (Reuters)"
    x = re.sub(r'(&[A-Za-z]+)|\(.*\)', '', x)
    # split into words
    tokens = word_tokenize(x)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # # remove punctuation from each word
    # table = str.maketrans(string.punctuation, ' ')
    # stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in tokens if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    # re-create document from words
    doc = ' '.join(words)
    return str(doc)

# %%
for i, row in train_data.iterrows():
    train_data.at[i, "headline_cleaned"] = clean(row.headline)
train_data.head()

# %%
for i, row in test_data.iterrows():
    test_data.at[i, "headline_cleaned"] = clean(row.headline)
test_data.head()

# %% [markdown]
# ### Clean news sources from content

# %%
# Function to clean out the dates
def clean_dates(x):
    x = re.sub(r'[0-9 ]*(January|February|March|April|May|June|July|August|September|October|November|December|JANUARY|FEBRUARY|MARCH|APRIL|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEPT|SEP|OCT|NOV|DEC|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)[., ]*[0-9]*[., ]*[0-9]*', ' ', x)
    return x

# Clean out the dates for training data
for i, row in train_data.iterrows():
    train_data.at[i, "content_cleaned"] = clean_dates(row.content)

# Clean out the dates for testing data
for i, row in test_data.iterrows():
    test_data.at[i, "content_cleaned"] = clean_dates(row.content)

# %%
# Function to extract the sources
def extract_sources(x):
    sources = []
    for sentence in x:
        trimmed = sentence[:35]
        temp = re.search(r'^[A-Za-z0-9\/,. ]*\(*[A-Za-z.]+\)* -', trimmed)
        if temp is not None:
            sources.append(temp.group())
    sources = numpy.array(sources)
    sources = numpy.unique(sources)
    return sources

# Call function to generate the sources list and save it
sources = extract_sources(train_data.content_cleaned)
sources = numpy.append(sources, extract_sources(test_data.content_cleaned))
sources = numpy.unique(sources)

# %%
# Save the sources to a file
numpy.savetxt("./data/news_sources_from_both.csv", sources, \
             header='list', delimiter=',', fmt='%s')

# %% [markdown]
# #### STOP! Make sure to check and clean manually any invalid sources first
# Manually go into the csv file to clean it out before running below!

# %%
# Use the sources list to clean out the news sources
sources_df = pandas.read_csv("./data/news_sources_from_both_v1.csv", sep='\n')

# %%
def remove_sources(x, sources):
    x = str(x)
    for i, source in sources.iterrows():
        if source.list in x:
            x = x.replace(source.list, ' ')
    return x

for i, row in train_data.iterrows():
    train_data.at[i, "content_nosources"] = remove_sources(row.content_cleaned, sources_df)

for i, row in test_data.iterrows():
    test_data.at[i, "content_nosources"] = remove_sources(row.content_cleaned, sources_df)

# %%
for i, row in train_data.iterrows():
    train_data.at[i, "content_cleaned"] = clean(row.content_nosources)
train_data.head()

# %%
for i, row in test_data.iterrows():
    test_data.at[i, "content_cleaned"] = clean(row.content_nosources)
test_data.head()

# %% [markdown]
# ### Sample 4000 rows

# %%
train_data_sample = train_data.sample(n = 4000, replace = False, random_state = 123)
train_data_sample.head()

# %%
test_data_sample = test_data.sample(n = 4000, replace = False, random_state = 123)
test_data_sample.head()

# %%
# create a CountVectorizer from raw data, with options to clean it
cv = CountVectorizer(min_df = 2, token_pattern=r'(?u)\b[A-Za-z]{2,}\b', 
                    ngram_range = (1, 1), stop_words = 'english')
cv_matrix = cv.fit_transform(train_data_sample.headline).toarray()

# get all unique words in the corpus
vocab = cv.get_feature_names()

# produce a dataframe including the feature names
cv_matrix_df = pandas.DataFrame(cv_matrix, columns=vocab)

#<<<<<<<<<<<<< compared with this (less stopwords removed)>>>>>>>>>>>>>>>>>
# cv2 = CountVectorizer(min_df = 2, token_pattern=r'(?u)\b[A-Za-z]{2,}\b', ngram_range = (1, 1))
# cv_matrix2 = cv2.fit_transform(train_data_sample.headline).toarray()

# # get all unique words in the corpus
# vocab2 = cv2.get_feature_names()

# # produce a dataframe including the feature names
# cv_matrix_df2 = pandas.DataFrame(cv_matrix2, columns=vocab2)

# %% [markdown]
# ### Use a SQLite3 database to save necessary data

# %%
db = sqlite3.connect('./data/newsclassifier.db')
cat_list = pandas.read_csv('./data/classes.txt', header=None)
cat_list.head()
cat_list.to_sql("category_list", db, if_exists='replace')
train_data.to_sql('train_data', db, if_exists='replace')
test_data.to_sql('test_data', db, if_exists='replace')
train_data_sample.to_sql('train_data_sample', db, if_exists='replace')
test_data_sample.to_sql('test_data_sample', db, if_exists='replace')
#cv_matrix_df.to_sql('headline_bagofwords', db, if_exists='replace') <- sqlite columns max 2000
db.commit()
db.close()

# %%
train_data.to_csv('./data/cleanedtrain.csv')
test_data.to_csv('./data/cleanedtest.csv')

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
# print out the number of unique documents in each category
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

counter = Counter(word_count_dict)

freq_df = pandas.DataFrame.from_records(counter.most_common(20),
                                        columns=['Top 20 words', 'Frequency'])
freq_df.plot(kind='bar', x='Top 20 words')