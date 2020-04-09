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
from sklearn import decomposition, ensemble
import nltk
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
import pydot

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
tfidf_vect = TfidfVectorizer(sublinear_tf = True, min_df = 1, lowercase = True, 
                             strip_accents = 'ascii', ngram_range = (1, 1), 
                             stop_words = 'english', use_idf = True, token_pattern=r'(?u)\b[A-Za-z]{2,}\b')
tfidf_unigram = tfidf_vect.fit_transform(train_data_df.headline_cleaned).toarray()
tfidf_fit = tfidf_vect.fit_transform(train_data_df.headline_cleaned)
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
                             stop_words = 'english', use_idf = True, token_pattern=r'(?u)\b[A-Za-z]{2,}\b')
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

# %% [markdown]
# ## Word Embedding

# Build the Corpus Vocabulary

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(train_data_df.headline_cleaned)
word2id = tokenizer.word_index

# build vocabulary of unique words
word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in train_data_df.headline_cleaned]

vocab_size = len(word2id)
embed_size = 100
window_size = 2 # context window size

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:100])

# %% 
# Build a CBOW (context, target) generator
def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size*2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word   = []            
            start = index - window_size
            end = index + window_size + 1
            
            context_words.append([words[i] 
                                 for i in range(start, end) 
                                 if 0 <= i < sentence_length 
                                 and i != index])
            label_word.append(word)

            x = sequence.pad_sequences(context_words, maxlen=context_length)
            y = np_utils.to_categorical(label_word, vocab_size)
            yield (x, y)

# Test this out for some samples
i = 0
for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
    if 0 not in x[0]:
        print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[numpy.argwhere(y[0])[0][0]])
    
        if i == 20:
            break
        i += 1

# %%

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda

# Build CBOW architecture
cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# view model summary
print(cbow.summary())

# visualize model structure
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(cbow, show_shapes=True, show_layer_names=False, rankdir='TB').create(prog='dot', format='svg'))


# %% [markdown]
# ## Document Similarity

# %%
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(tfidf_fit)
similarity_df = pandas.DataFrame(similarity_matrix)
similarity_df

# %%
# ### Cluster using similarity features
from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(similarity_matrix, 'ward')
print(pandas.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2', 
                         'Distance', 'Cluster Size'], dtype='object'))

plt.figure(figsize=(8, 3))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Data point')
plt.ylabel('Distance')
dendrogram(Z)
plt.axhline(y=1.0, c='k', ls='--', lw=0.5)

# %%
from scipy.cluster.hierarchy import fcluster
max_dist = 1.0

cluster_labels = fcluster(Z, max_dist, criterion='distance')
cluster_labels = pandas.DataFrame(cluster_labels, columns=['ClusterLabel'])
pandas.concat([train_data_df, cluster_labels], axis=1)

# %% [markdown]
# ## LDA Model for features

# %%
# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=85, max_iter=20, learning_method='online')
topics = lda_model.fit_transform(cv_matrix)
topic_word = lda_model.components_ 
vocab = cv.get_feature_names()
features = pandas.DataFrame(topics, columns=vocab)
print(features)

# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

# %%
print('Topic Words: ' + topic_words[1])
print('\n')
print('Topic Summaries: ' + topic_summaries[1])

# %%_
for topic_weights in topic_word:
    topic = [(token, weight) for token, weight in zip(vocab, topic_weights)]
    topic = sorted(topic, key=lambda x: -x[1])
    topic = [item for item in topic if item[1] > 0.6]
    print(topic)
    print()

# %%
from sklearn.cluster import KMeans

km = KMeans(n_clusters=4, random_state=0)
km.fit_transform(features)
cluster_labels = km.labels_
cluster_labels = pandas.DataFrame(cluster_labels, columns=['Cluster Label'])
pandas.concat([train_data_df, cluster_labels], axis=1)


# %%
