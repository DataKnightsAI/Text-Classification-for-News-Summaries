# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Feature Engineering, Baseline Model and Feature Selection

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
import seaborn as sns
from sklearn.metrics import precision_recall_curve # The average precision score in multi-label settings
from sklearn.metrics import average_precision_score
from sklearn import svm # Support Vector Machine
from sklearn.preprocessing import label_binarize # Split category encoding eg. y=[1,2,3] into y1=[0,1], y2=[0,1], y3=[0,1]
from sklearn.model_selection import train_test_split # Built-in train test splitter
from sklearn.multiclass import OneVsRestClassifier # We use OneVsRestClassifier for multi-label prediction
from itertools import cycle
from sklearn.feature_selection import SelectPercentile, f_classif

# %% [markdown]
# ## Load in the data from the database

# %%
dbconn = sqlite3.connect('./data/cleanedtraintest_v2.db')
train_data_df = pandas.read_sql_query('SELECT * FROM train_data', dbconn)
test_data_df = pandas.read_sql_query('SELECT * FROM test_data', dbconn)
dbconn.commit()
dbconn.close()

# %% [markdown]
# ### Check the if the data was loaded correctly

# %%
train_data_df.head()

# %%
train_data_df.drop('index', axis=1, inplace=True)
train_data_df.head()

# %%
test_data_df.head()

# %%
test_data_df.drop('index', axis=1, inplace=True)
test_data_df.head()

# %% [markdown]
# ### Sample 4000 rows

# %%
train_data_sample = train_data_df.sample(n = 4000, replace = False, random_state = 123)
train_data_sample.head()

# %% 
test_data_sample = test_data_df.sample(n = 4000, replace = False, random_state = 123)
test_data_sample.head()

# %% [markdown]
# ## Train & Test data where x is the predictor features, y is the predicted feature

n_classes = 4

x_train = train_data_sample.content_cleaned
y_train = label_binarize(train_data_sample.category, classes=[1, 2, 3, 4])

x_test = test_data_sample.content_cleaned
y_test = label_binarize(test_data_sample.category, classes=[1, 2, 3, 4])

# %% [markdown]
# ### Let's make a Bag of Words
# %%
# Use countvectorizer to get a vector of words
cv = CountVectorizer(min_df = 2, lowercase = True,
                     token_pattern=r'\b[A-Za-z]{2,}\b', ngram_range = (1, 1))
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(x_train_cv, train_data_sample.category)
x_train_cv_10p = selector.transform(x_train_cv).toarray()
x_test_cv_10p = selector.transform(x_test_cv).toarray()

# get all unique words in the corpus
bow_vocab = cv.get_feature_names()

columns = numpy.asarray(bow_vocab)
support = numpy.asarray(selector.get_support())
bow_vocab_10p = columns[support]

x_train_cv = x_train_cv.toarray()
x_test_cv = x_test_cv.toarray()

# produce a dataframe including the feature names
x_train_bagofwords = pandas.DataFrame(x_train_cv, columns=bow_vocab)
x_test_bagofwords = pandas.DataFrame(x_test_cv, columns=bow_vocab)
x_train_bagofwords_10p = pandas.DataFrame(x_train_cv_10p, columns=bow_vocab_10p)
x_test_bagofwords_10p = pandas.DataFrame(x_test_cv_10p, columns=bow_vocab_10p)
x_train_bagofwords.head()

#%%
x_test_bagofwords_10p.head()

# %% [markdown]
# ### We have bag of words already, let's make a Bag of N-Grams
# %%
# Use countvectorizer to get a vector of ngrams
cv = CountVectorizer(min_df = 2, lowercase = True,
                     token_pattern=r'\b[A-Za-z]{2,}\b', ngram_range = (2, 3))
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

# get all unique words in the corpus
ngram_vocab = cv.get_feature_names()

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(x_train_cv, train_data_sample.category)
x_train_cv_10p = selector.transform(x_train_cv).toarray()
x_test_cv_10p = selector.transform(x_test_cv).toarray()

columns = numpy.asarray(ngram_vocab)
support = numpy.asarray(selector.get_support())
ngram_vocab_10p = columns[support]

x_train_cv = x_train_cv.toarray()
x_test_cv = x_test_cv.toarray()

# produce a dataframe including the feature names
x_train_bagofngrams = pandas.DataFrame(x_train_cv, columns=ngram_vocab)
x_test_bagofngrams = pandas.DataFrame(x_test_cv, columns=ngram_vocab)
x_train_bagofngrams_10p = pandas.DataFrame(x_train_cv_10p, columns=ngram_vocab_10p)
x_test_bagofngrams_10p = pandas.DataFrame(x_test_cv_10p, columns=ngram_vocab_10p)
x_train_bagofngrams.head()

# %% 
# Use countvectorizer to get a vector of chars
cv = CountVectorizer(analyzer='char', min_df = 2, ngram_range = (2, 3), 
                     token_pattern=r'\b[A-Za-z]{2,}\b')
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

# get all unique words in the corpus
cv_char_vocab = cv.get_feature_names()

selector = SelectPercentile(f_classif, percentile=10)
selector.fit(x_train_cv, train_data_sample.category)
x_train_cv_10p = selector.transform(x_train_cv).toarray()
x_test_cv_10p = selector.transform(x_test_cv).toarray()

columns = numpy.asarray(cv_char_vocab)
support = numpy.asarray(selector.get_support())
cv_char_vocab_10p = columns[support]

x_train_cv = x_train_cv.toarray()
x_test_cv = x_test_cv.toarray()

# produce a dataframe including the feature names
x_train_cv_char = pandas.DataFrame(x_train_cv, columns = cv_char_vocab)
x_test_cv_char = pandas.DataFrame(x_test_cv, columns=cv_char_vocab)
x_train_cv_char_10p = pandas.DataFrame(x_train_cv_10p, columns = cv_char_vocab_10p)
x_test_cv_char_10p = pandas.DataFrame(x_test_cv_10p, columns=cv_char_vocab_10p)
x_train_cv_char.head()

# %% [markdown]
# ### Let's explore the data we got through plots and tables

# %% Function to show a barchart of the top 40 words
def words_barchart(df, df_label):
    word_count_dict = {}
    for word in df_label:
        word_count_dict[word] = int(sum(df.loc[:, word]))

    counter = Counter(word_count_dict)

    freq_df = pandas.DataFrame.from_records(counter.most_common(40),
                                            columns=['Top 40 words', 'Frequency'])

    plt.figure(figsize=(10,5))
    chart = sns.barplot(
        data=freq_df,
        x='Top 40 words',
        y='Frequency'
    )

    chart.set_xticklabels(
        chart.get_xticklabels(), 
        rotation=45, 
        horizontalalignment='right',
        fontweight='light'
    )

#%% Bar chart of Bag of Words frequency
words_barchart(x_train_bagofwords, bow_vocab)

# %% Bar chart of N-Grams frequency
words_barchart(x_train_bagofngrams, ngram_vocab)

# %% Bar chart of Character grams frequency
words_barchart(x_train_cv_char, cv_char_vocab)

#%% Bar chart of Bag of Words frequency
words_barchart(x_train_bagofwords_10p, bow_vocab_10p)

# %% Bar chart of N-Grams frequency
words_barchart(x_train_bagofngrams_10p, ngram_vocab_10p)

# %% Bar chart of Character grams frequency
words_barchart(x_train_cv_char_10p, cv_char_vocab_10p)

# %% [markdown]
# ## TF/IDF

# %% [markdown]
# ### Unigram TF/IDF

# %%

# Use TF/IDF vectorizer to get a vector of unigrams
tfidf_vect = TfidfVectorizer(sublinear_tf = True, min_df = 2, ngram_range = (1, 1), 
                             use_idf = True, token_pattern=r'\b[A-Za-z]{2,}\b')
x_train_tfidf_unigram = tfidf_vect.fit_transform(x_train).toarray()
x_test_tfidf_unigram = tfidf_vect.transform(x_test).toarray()

# get all unique words in the corpus
vocab = tfidf_vect.get_feature_names()

# produce a dataframe including the feature names
x_train_tfidf_unigram = pandas.DataFrame(numpy.round(x_train_tfidf_unigram, 2), columns = vocab)
x_test_tfidf_unigram = pandas.DataFrame(numpy.round(x_test_tfidf_unigram, 2), columns = vocab)
x_train_tfidf_unigram.head()

# %% [markdown]
# ### N-Gram TF/IDF

# Use TF/IDF vectorizer to get a vector of n-grams
tfidf_vect = TfidfVectorizer(sublinear_tf = True, min_df = 2, ngram_range = (2, 3), 
                             use_idf = True, token_pattern=r'\b[A-Za-z]{2,}\b')
x_train_tfidf_ngram = tfidf_vect.fit_transform(x_train).toarray()
x_test_tfidf_ngram = tfidf_vect.transform(x_test).toarray()
# get all unique words in the corpus
vocab = tfidf_vect.get_feature_names()

# produce a dataframe including the feature names
x_train_tfidf_ngram = pandas.DataFrame(numpy.round(x_train_tfidf_ngram, 2), columns = vocab)
x_test_tfidf_ngram = pandas.DataFrame(numpy.round(x_test_tfidf_ngram, 2), columns = vocab)
x_train_tfidf_ngram.head()

# %% [markdown]
# ### Character TF/IDF

# Use TF/IDF vectorizer to get a vector of chars 
tfidf_vect = TfidfVectorizer(analyzer = 'char', sublinear_tf = True, min_df = 2, 
                             ngram_range = (2, 3), use_idf = True, 
                             token_pattern=r'\b[A-Za-z]{2,}\b')
x_train_tfidf_char = tfidf_vect.fit_transform(x_train).toarray()
x_test_tfidf_char = tfidf_vect.transform(x_test).toarray()

# get all unique words in the corpus
char_vocab = tfidf_vect.get_feature_names()

# produce a dataframe including the feature names
x_train_tfidf_char = pandas.DataFrame(numpy.round(x_train_tfidf_char, 2), columns = char_vocab)
x_test_tfidf_char = pandas.DataFrame(numpy.round(x_test_tfidf_char, 2), columns = char_vocab)
x_train_tfidf_char.head()


# %% [markdown]
# ## Document Similarity

# %%
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(x_train_tfidf_ngram)
similarity_df = pandas.DataFrame(similarity_matrix)
similarity_df

# %%
from sklearn.cluster import KMeans

km = KMeans(
    n_clusters=2, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(similarity_df)

#Convert to array for clustering to work
similarity_df_array = numpy.array(similarity_df)

# plot the 3 clusters
plt.scatter(
    similarity_df_array[y_km == 0, 0], similarity_df_array[y_km == 0, 1],
    s=50, c='lightgreen',
    marker='s', edgecolor='black',
    label='cluster 1'
)

plt.scatter(
    similarity_df_array[y_km == 1, 0], similarity_df_array[y_km == 1, 1],
    s=50, c='red',
    marker='o', edgecolor='black',
    label='cluster 2'
)

# plt.scatter(
#     similarity_df_array[y_km == 2, 0], similarity_df_array[y_km == 2, 1],
#     s=50, c='lightblue',
#     marker='v', edgecolor='black',
#     label='cluster 3'
# )

#plt.scatter(
#    w2v_feature_array[y_km == 2, 0], w2v_feature_array[y_km == 2, 1],
#    s=50, c='red',
#    marker='h', edgecolor='black',
#    label='cluster 4'
#)

# plot the centroids
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    s=250, marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.legend(scatterpoints=1)
plt.grid()
plt.show()

# %% [markdown]
# ## LDA Model for features

# %%
# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=5929, max_iter=20, learning_method='online')
topics = lda_model.fit_transform(x_train_tfidf_ngram)
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
pandas.concat([x_train, cluster_labels], axis=1)

# %% [markdown]
# ## Using gensim to build Word2Vec

# %%
from gensim.models import word2vec

# tokenize sentences in corpus
wpt = nltk.WordPunctTokenizer()
tokenized_corpus_train = [wpt.tokenize(document) for document in x_train]
tokenized_corpus_test = [wpt.tokenize(document) for document in x_test]

# Set values for various parameters
feature_size = 4000    # Word vector dimensionality  
window_context = 20          # Context window size      
workers = 12                                                                              
min_word_count = 5   # Minimum word count                        
sample = 1e-3   # Downsample setting for frequent words

w2v_model_train = word2vec.Word2Vec(tokenized_corpus_train, size=feature_size, 
                          window=window_context, min_count=min_word_count,
                          sample=sample, iter=50)
w2v_model_test = word2vec.Word2Vec(tokenized_corpus_test, size=feature_size, 
                          window=window_context, min_count=min_word_count,
                          sample=sample, iter=50)

# %% [markdown]
# ### Visualize Word Embedding

# %%
# from sklearn.manifold import TSNE
# words = w2v_model.wv.index2word
# wvs = w2v_model.wv[words]
# tsne = TSNE(n_components=2, random_state=0, n_iter=500, perplexity=2)
# numpy.set_printoptions(suppress=True)
# T = tsne.fit_transform(wvs)
# labels = words
# plt.figure(figsize=(12, 6))
# plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
# for label, x, y in zip(labels, T[:, 0], T[:, 1]):
#  plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')

# %% [markdown]
# ### Functions to get document level embeddings
# ### The idea is to distill a word vector of 'n' features into a single point and use that at a document level

# %%
def average_word_vectors(words, model, vocabulary, num_features):

 feature_vector = numpy.zeros((num_features,),dtype="float64")
 nwords = 0.

 for word in words:
    if word in vocabulary:
      nwords = nwords + 1.
      feature_vector = numpy.add(feature_vector, model[word])

 if nwords:
    feature_vector = numpy.divide(feature_vector, nwords)

 return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
 vocabulary = set(model.wv.index2word)
 features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
            for tokenized_sentence in corpus]
 return numpy.array(features)

# %% [markdown]
# ### Obtain document level embeddings

# %%
w2v_feature_array_train = averaged_word_vectorizer(corpus=tokenized_corpus_train, model=w2v_model_train,
                                            num_features=feature_size)
w2v_feature_array_test = averaged_word_vectorizer(corpus=tokenized_corpus_test, model=w2v_model_test,
                                            num_features=feature_size)
x_train_w2v = pandas.DataFrame(w2v_feature_array_train)
x_test_w2v = pandas.DataFrame(w2v_feature_array_test)

#%%
x_train_w2v.head()

#%%
x_test_w2v.head()


# %% [markdown]
# ## Perform SVM as a baseline model and evaluate it.

# %%
# Use label_binarize to be multi-label like settings
# X = x_train['content_cleaned']
# y = x_train['category']
# Y = label_binarize(y, classes=[1, 2, 3, 4])
# n_classes = Y.shape[1]

# %%
word_freq_df = pandas.DataFrame(x_train_bagofwords.toarray(), columns=cv.get_feature_names())
top_words_df = pandas.DataFrame(word_freq_df.sum()).sort_values(0, ascending=False)
word_freq_df.head(20)

# %%
top_words_df.head(20)

#%% [markdown]
# ## Run SVM and Plot the results

# %%
# Run classifier
def run_svm(x_train, y_train, x_test, emb):
    str(emb)
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=1))
    classifier.fit(x_train, y_train)
    y_score = classifier.decision_function(x_test)


    # The average precision score in multi-label settings
    # For each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                        average="micro")
    print(emb)
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'
        .format(average_precision["micro"]))

    # Plot the micro-averaged Precision-Recall curve
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score for, micro-averaged over all classes: AP={0:0.2f}'
        .format(average_precision["micro"]))
    
    # Plot Precision-Recall curve for each class and iso-f1 curves
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = numpy.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = numpy.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                    ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.5), prop=dict(size=14))

    plt.show()

#%%
run_svm(x_train_bagofwords, y_train, x_test_bagofwords, 'Bag of Words')

#%%
run_svm(x_train_bagofngrams, y_train, x_test_bagofngrams, 'Bag of N-Grams')

#%%
run_svm(x_train_cv_char, y_train, x_test_cv_char, 'Bag of Chars')

#%%
run_svm(x_train_tfidf_unigram, y_train, x_test_tfidf_unigram, 'TF/IDF Unigram')

#%%
run_svm(x_train_tfidf_ngram, y_train, x_test_tfidf_ngram, 'TF/IDF N-Grams')

#%%
run_svm(x_train_tfidf_char, y_train, x_test_cv_char, 'TF/IDF Chars')

#%%
run_svm(x_train_w2v, y_train, x_test_w2v, 'Word2Vec')

#%%
run_svm(x_train_bagofwords_10p, y_train, x_test_bagofwords_10p, 'Bag of Words - 90th percentile')

#%%
run_svm(x_train_bagofngrams_10p, y_train, x_test_bagofngrams_10p, 'Bag of N-Grams - 90th percentile')

#%%
run_svm(x_train_cv_char_10p, y_train, x_test_cv_char_10p, 'Bag of Chars - 90th percentile')

#%%
#===================================<experimental code below>===================================

# %%
# from sklearn.naive_bayes import MultinomialNB
# naive_bayes = MultinomialNB()
# naive_bayes.fit(X_train_cv, y_train)
# predictions = naive_bayes.predict(X_test_cv)

# %%
# from sklearn.metrics import accuracy_score, precision_score, recall_score
# print('Accuracy score: ', accuracy_score(y_test, y_score))
#print('Precision score: ', precision_score(y_test, predictions))
#print('Recall score: ', recall_score(y_test, predictions))

# %%
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, predictions)
# sns.heatmap(cm, square=True, annot=True, cmap='RdBu', cbar=False,
# xticklabels=['Business', 'Sports'], yticklabels=['Business', 'Sports'])
# plt.xlabel('true label')
# plt.ylabel('predicted label')

# %% [markdown]
# ## Word Embedding

# # Build the Corpus Vocabulary

# tokenizer = text.Tokenizer()
# tokenizer.fit_on_texts(x_train.headline_cleaned)
# word2id = tokenizer.word_index

# # build vocabulary of unique words
# word2id['PAD'] = 0
# id2word = {v:k for k, v in word2id.items()}
# wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in train_data_sample.headline_cleaned]

# vocab_size = len(word2id)
# embed_size = 100
# window_size = 2 # context window size

# print('Vocabulary Size:', vocab_size)
# print('Vocabulary Sample:', list(word2id.items())[:100])

# # %% 
# # Build a CBOW (context, target) generator
# def generate_context_word_pairs(corpus, window_size, vocab_size):
#     context_length = window_size*2
#     for words in corpus:
#         sentence_length = len(words)
#         for index, word in enumerate(words):
#             context_words = []
#             label_word   = []            
#             start = index - window_size
#             end = index + window_size + 1
            
#             context_words.append([words[i] 
#                                  for i in range(start, end) 
#                                  if 0 <= i < sentence_length 
#                                  and i != index])
#             label_word.append(word)

#             x = sequence.pad_sequences(context_words, maxlen=context_length)
#             y = np_utils.to_categorical(label_word, vocab_size)
#             yield (x, y)

# # Test this out for some samples
# i = 0
# for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
#     if 0 not in x[0]:
#         print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[numpy.argwhere(y[0])[0][0]])
    
#         if i == 20:
#             break
#         i += 1

# # %%
# import keras.backend as K
# from keras.models import Sequential
# from keras.layers import Dense, Embedding, Lambda

# # Build CBOW architecture
# cbow = Sequential()
# cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
# cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
# cbow.add(Dense(vocab_size, activation='softmax'))
# cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# # view model summary
# print(cbow.summary())

# # visualize model structure
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot

# SVG(model_to_dot(cbow, show_shapes=True, show_layer_names=False, rankdir='TB').create(prog='dot', format='svg'))

# %%
# # ### Cluster using similarity features
# from scipy.cluster.hierarchy import dendrogram, linkage

# Z = linkage(similarity_matrix, 'ward')
# print(pandas.DataFrame(Z, columns=['Document\Cluster 1', 'Document\Cluster 2', 
#                          'Distance', 'Cluster Size'], dtype='object'))

# plt.figure(figsize=(8, 3))
# plt.title('Hierarchical Clustering Dendrogram')
# plt.xlabel('Data point')
# plt.ylabel('Distance')
# dendrogram(Z)
# plt.axhline(y=1.0, c='k', ls='--', lw=0.5)

# # %%
# from scipy.cluster.hierarchy import fcluster
# max_dist = 1.0

# cluster_labels = fcluster(Z, max_dist, criterion='distance')
# cluster_labels = pandas.DataFrame(cluster_labels, columns=['ClusterLabel'])
# pandas.concat([train_data_sample, cluster_labels], axis=1)