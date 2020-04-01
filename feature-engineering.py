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


# %%


