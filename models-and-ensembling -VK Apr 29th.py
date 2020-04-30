# %% [markdown]
# # Models and Ensembling Methods

# %% [markdown]
# ## Import dependencies
import numpy
from gensim.models import word2vec
from gensim.models import KeyedVectors
import pandas
from nltk import WordPunctTokenizer
from sklearn.preprocessing import label_binarize
import sqlite3
from sklearn.multiclass import OneVsRestClassifier
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve # The average precision score in multi-label settings
from sklearn.metrics import average_precision_score
from sklearn import svm # Support Vector Machine
from itertools import cycle
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# %% [markdown]
# ## Read in the data

# %% [markdown]
# ### Load raw train and test data

# %% [markdown]
# #### Load in the data from the database

# %%
dbconn = sqlite3.connect('./data/cleanedtraintest_v2.db')
train_data_df = pandas.read_sql_query('SELECT category, content_cleaned FROM train_data', dbconn)
test_data_df = pandas.read_sql_query('SELECT category, content_cleaned FROM test_data', dbconn)
dbconn.commit()
dbconn.close()

# %% [markdown]
# #### Check the if the data was loaded correctly

# %%
train_data_df

# %%
test_data_df

# # %% [markdown]
# # #### Sample 4000 rows

# # %%
# train_data_sample = train_data_df #.sample(n=10000, replace=False, random_state=123)
# train_data_sample.head()

# # %% 
# test_data_sample = test_data_df #.sample(n=4000, replace=False, random_state=123)
# test_data_sample.head()

# %% [markdown]
# #### Train & Test data where x is the predictor features, y is the predicted feature

N_CLASSES = 4

x_train = train_data_df.content_cleaned
y_train = label_binarize(train_data_df.category, classes=[1, 2, 3, 4])

x_test = test_data_df.content_cleaned
y_test = label_binarize(test_data_df.category, classes=[1, 2, 3, 4])

# %% [markdown]
# ### Load word2vec data

# %% [markdown]
# #### Load word2vec feature arrays from .npz files

# load dict of arrays
w2v_train_features_array_dict = numpy.load(
    './data/word2vec-train-features-120000-min5dim300.npz')
w2v_test_features_array_dict = numpy.load(
    './data/word2vec-test-features-120000-min5dim300.npz')
# extract the first array from train
data = w2v_train_features_array_dict['arr_0']
# print the array
print(data)
# extract the first array from test
data = w2v_test_features_array_dict['arr_0']
# print the array
print(data)

# %% [markdown]
# #### Load word2vec model trained key vectors
w2v_model_train = KeyedVectors.load(
    './data/custom-trained-word2vec-120000-min5dim300.kv')

# %% [markdown]
# #### Get the word2vec data back into usable form

wpt = WordPunctTokenizer()
tokenized_corpus_train = [wpt.tokenize(document) for document in x_train]
tokenized_corpus_test = [wpt.tokenize(document) for document in x_test]

# %%
def average_word_vectors(words, model, vocabulary, num_features):

 feature_vector = numpy.zeros((num_features,), dtype="float32")
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
# #### Obtain document level embeddings

# %%
FEATURE_SIZE = 300

w2v_feature_array_train = averaged_word_vectorizer(corpus=tokenized_corpus_train,
    model=w2v_model_train, num_features=FEATURE_SIZE)
w2v_feature_array_test = averaged_word_vectorizer(corpus=tokenized_corpus_test,
    model=w2v_model_train, num_features=FEATURE_SIZE)

x_train_w2v = pandas.DataFrame(w2v_feature_array_train)
x_test_w2v = pandas.DataFrame(w2v_feature_array_test)

# %% [markdown]
# ## Build Models

# %% [markdown]
# ### SVM Model Building Functions

# SVM classifier function
def run_svm(x_train, y_train):
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=1))
    classifier.fit(x_train, y_train)
    return classifier

# Calculate, then plot the Precision, Recall, Average Precision, F1
def prf1_calc(classifier, algo_name, n_classes, x_test, y_test):
    # Get the decision function from the classifier
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

    # Plot the data
    prf1_plot(precision, recall, average_precision, algo_name, n_classes)

    # Return all metrics
    results = [algo_name, precision, recall, average_precision]
    return results

# Function to Plot Precision, Recall, F1
def prf1_plot(precision, recall, average_precision, algo_name, n_classes):
    print(algo_name)
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

# %% [markdown]
# ## Run the Models

# Get the SVM model fitted
svm_model = run_svm(x_train_w2v, y_train)

# %% [markdown]
# ## Get the scores
scores = []

# For SVM calculate and plot the Precision, Recall, Avg Precision
scores = scores.append(prf1_calc(svm_model, 'SVM', N_CLASSES, x_test_w2v, y_test))

# %%
# Logistic Regression function
def run_logreg(x_train, y_train):
    #classifier = OneVsRestClassifier(svm.LinearSVC(random_state=1))
    classifier = OneVsRestClassifier(LogisticRegression(random_state=1))
    classifier.fit(x_train, y_train)
    return classifier

# %%
# Run Logistic Regression Model
logreg_model = run_logreg(x_train_w2v, y_train)

# %%
