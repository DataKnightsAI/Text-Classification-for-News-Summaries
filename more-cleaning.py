# %% [markdown]
# ## Import necessary dependencies

# %%
import sqlite3
import pandas
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string

# %% [markdown]
# ## Load in the data from the database

# %%
dbconn = sqlite3.connect('./data/newsclassifier.db')
train_data = pandas.read_sql_query('SELECT * FROM train_data', dbconn)
test_data = pandas.read_sql_query('SELECT * FROM test_data', dbconn)
dbconn.commit()
dbconn.close()

# %%
train_data.drop('index', axis=1, inplace=True)
test_data.drop('index', axis=1, inplace=True)

# %% [markdown]
# ### Clean v2: More HTML code & news sources

# %%
# Define a clean function: lowercase, strip HTML, punctuations, non-alpha, stopwords
def clean(x):
    # strip HTML and sources of the format eg. "&lt and (Reuters)"
    x = re.sub(r'(&[A-Za-z]+)|\(.*\)', '', x)
    x = re.sub(r'(FONT|font).*\/(FONT|font)', '', x)
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

# %% Clean train content from no sources set
for i, row in train_data.iterrows():
    train_data.at[i, "content_cleaned"] = clean(row.content_nosources)
train_data.head()

# %% Clean test content from no sources set
for i, row in test_data.iterrows():
    test_data.at[i, "content_cleaned"] = clean(row.content_nosources)
test_data.head()

# %% Save to database as a new file
dbconn = sqlite3.connect('./data/cleanedtraintest_v2.db')
train_data.to_sql('train_data', dbconn, if_exists='replace')
test_data.to_sql('test_data', dbconn, if_exists='replace')
dbconn.commit()
dbconn.close()

# %% Save to csv
train_data.to_csv('./data/cleanedtrain_v2.csv')
test_data.to_csv('./data/cleanedtest_v2.csv')


# %% Verify that everything is saved.
dbconn = sqlite3.connect('./data/cleanedtraintest_v2.db')
train_data = pandas.read_sql_query('SELECT * FROM train_data', dbconn)
test_data = pandas.read_sql_query('SELECT * FROM test_data', dbconn)
dbconn.commit()
dbconn.close()

test_data.head(20)
train_data.head(20)