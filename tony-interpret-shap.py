# %% [markdown]
# # SHAP to interpret our dataset on TF/IDF N-grams embedding

# %% [markdown]
# ## Import packages and declare constants
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter
import sqlite3
import seaborn as sns
from sklearn import svm
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from itertools import cycle
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
import shap

RANDOM_STATE = 123
N_CLASSES = 4
CLASSES = ['World', 'Sports', 'Business', 'Sci/Tech']

# %% [markdown]
# ## Read in the data

# ### First read from the database the cleaned data
dbconn = sqlite3.connect('./data/cleanedtraintest_v2.db')
train_data_df = pd.read_sql_query('SELECT * FROM train_data', dbconn)
test_data_df = pd.read_sql_query('SELECT * FROM test_data', dbconn)
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

# %%
train_data_df.shape

# %%
test_data_df.shape

# %% [markdown]
# ### Sample down
train_data_sample = train_data_df.sample(n = 10000, replace = False,
                                         random_state = RANDOM_STATE)
test_data_sample = test_data_df.sample(n = 10000, replace = False,
                                       random_state = RANDOM_STATE)

# ### Train & Test data where x is the predictor features, y is the predicted feature
x_train = train_data_sample.content_cleaned
x_test = test_data_df.content_cleaned
y_train = train_data_sample.category
y_test = test_data_df.category

# %% [markdown]
# ## N-Gram TF/IDF

# Use TF/IDF vectorizer to get a vector of n-grams
tfidf_vect = TfidfVectorizer(sublinear_tf = True, min_df = 2, ngram_range = (2, 3), 
                             use_idf = True, token_pattern=r'\b[A-Za-z]{2,}\b')
x_train_tfidf = tfidf_vect.fit_transform(x_train)
x_test_tfidf = tfidf_vect.transform(x_test)

# get all unique words in the corpus
vocab = tfidf_vect.get_feature_names()

x_train_tfidf = np.round(x_train_tfidf, 2)
x_test_tfidf = np.round(x_test_tfidf, 2)

# produce a dataframe including the feature names
x_train_tfidf_ngram = pd.DataFrame(x_train_tfidf.toarray(), columns = vocab)
x_test_tfidf_ngram = pd.DataFrame(x_test_tfidf.toarray(), columns = vocab)

# %% [markdown]
# ## Build the models by running the data through an algo

# ### SVM Algo Building Function
def run_svm(x, y):
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=RANDOM_STATE))
    classifier.fit(x, y)
    return classifier

# %% [markdown]
# #### Train SVM
svm_model = run_svm(x_train_tfidf_ngram, y_train)

# %% [markdown]
# ### Decision Trees Function
def run_dectree(x_train, y_train):
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    return classifier 

# %% [markdown]
# #### Train Decision Trees Classifier
dectree_model = run_dectree(x_train_tfidf_ngram, y_train)

# %% [markdown]
# ### Logistic Regression Function
def run_logreg(x_train, y_train):
    classifier = OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE))
    classifier.fit(x_train, y_train)
    return classifier

# %% [markdown]
# #### Train Logistic Regression
logreg_model = run_logreg(x_train_tfidf_ngram, y_train)

# %% [markdown]
# ## Get an idea of the results

# ### Confusion Matrix
y_test_pred = logreg_model.predict(x_test_tfidf_ngram)
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index = CLASSES, columns = CLASSES)
cm_df.index.name = 'Actual'
cm_df.columns.name = 'Predicted'
plt.title('Confusion Matrix for ' + "LogReg", fontsize=14)
sns.heatmap(cm_df, annot=True, fmt='.6g', annot_kws={"size": 10}, cmap='Reds')
plt.show()

# %% [markdown]
# ## Interpret using SHAP

# ### Initialize/Learn with a subset of data
attrib_data = shap.sample(x_train_tfidf_ngram, 20)
explainer = shap.KernelExplainer(logreg_model.predict_proba, attrib_data)
shap_vals = explainer.shap_values(x_test_tfidf_ngram)

# %% [markdown]
# ### Plot the different outputs

# #### Summary first
shap.summary_plot(shap_vals, feature_names=vocab, class_names=CLASSES)

# %% [markdown]
# #### Decision plots for different predictions
shap.decision_plot(explainer.expected_value[0], shap_vals[0][40], features=x_test_tfidf_ngram.iloc[40],
    feature_display_range=slice(None, -17, -1))

shap.decision_plot(explainer.expected_value[0], shap_vals[0][39], features=x_test_tfidf_ngram.iloc[39],
    feature_display_range=slice(None, -17, -1))

shap.decision_plot(explainer.expected_value[0], shap_vals[0][38], features=x_test_tfidf_ngram.iloc[38],
    feature_display_range=slice(None, -17, -1))

shap.decision_plot(explainer.expected_value[0], shap_vals[0][23], features=x_test_tfidf_ngram.iloc[23],
    feature_display_range=slice(None, -17, -1))

# %%
shap.multioutput_decision_plot(explainer.expected_value, shap_vals,
    feature_names=vocab, row_index=40,
    highlight=y_test_pred[40])

# %%
shap.force_plot(explainer.expected_value[0], shap_vals[0][0,:],
    x_test_tfidf_ngram.iloc[0,:], link="logit", matplotlib=True)

# %% [markdown]
# # References
# - Explain NLP models with LIME & SHAP by Susan Li
#   https://towardsdatascience.com/explain-nlp-models-with-lime-shap-5c5a9f84d59b
# - Interpretable ML e-Book by Christoph Molnar, 2020-04-27
#   https://christophm.github.io/interpretable-ml-book/
# - Kernel SHAP explanation for SVM models - Seldon Technologies Ltd, 2019
#   https://docs.seldon.io/projects/alibi/en/stable/examples/kernel_shap_wine_intro.html
# - A Unified Approach to Interpreting Model Predictions - Scott M. Lundberg & Su-In Lee
#   http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions
# - 17 January 2020, From local explanations to global understanding with explainable AI for trees
#   https://www.nature.com/articles/s42256-019-0138-9
# - 10 October 2018, Explainable machine-learning predictions for the prevention of hypoxaemia during surgery
#   https://www.nature.com/articles/s41551-018-0304-0 