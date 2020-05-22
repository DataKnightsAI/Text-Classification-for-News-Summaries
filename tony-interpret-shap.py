# %% [markdown]
# # SHAP to interpret our dataset on n-grams embedding

# %% [markdown]
# ## Import packages and declare constants
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
#from sklearn.feature_selection import chi2
#from PIL import Image
from collections import Counter
#import re
import sqlite3
#from sklearn import decomposition, ensemble
#import nltk
#from keras.preprocessing import text
#from keras.utils import np_utils
#from keras.preprocessing import sequence
#import pydot
import seaborn as sns
#from sklearn.metrics import precision_recall_curve
#from sklearn.metrics import average_precision_score
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
train_data_sample = train_data_df.sample(n = 300, replace = False,
                                         random_state = RANDOM_STATE)
test_data_sample = test_data_df.sample(n = 300, replace = False,
                                       random_state = RANDOM_STATE)

# ### Train & Test data where x is the predictor features, y is the predicted feature
x_train = train_data_sample.content_cleaned
# y_train = label_binarize(train_data_sample.category, classes=range(1, N_CLASSES+1))
x_test = test_data_df.content_cleaned
# y_test = label_binarize(test_data_df.category, classes=range(1, N_CLASSES+1))
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
# ## Build the model by running the data through an algo

# ### SVM Algo Building Function
def run_svm(x, y):
    # classifier = svm.SVC(random_state=RANDOM_STATE, decision_function_shape='ovr')
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=RANDOM_STATE))
    classifier.fit(x, y)
    return classifier

# %% [markdown]
# ### Run the data through the algo
svm_model = run_svm(x_train_tfidf_ngram, y_train)

# %% [markdown]
# ### Decision Trees Function
def run_dectree(x_train, y_train):
    classifier = tree.DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    return classifier 

# %%
# Run Decision Trees Classifier
dectree_model = run_dectree(x_train_tfidf_ngram, y_train)

def run_logreg(x_train, y_train):
    classifier = OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE))
    classifier.fit(x_train, y_train)
    return classifier

logreg_model = run_logreg(x_train_tfidf_ngram, y_train)

# %% 
# ### Confusion Matrix
y_test_pred = svm_model.predict(x_test_tfidf_ngram)
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, index = CLASSES, columns = CLASSES)
cm_df.index.name = 'Actual'
cm_df.columns.name = 'Predicted'
plt.title('Confusion Matrix for ' + "SVC", fontsize=14)
sns.heatmap(cm_df, annot=True, fmt='.6g', annot_kws={"size": 10}, cmap='Reds')
plt.show()

# %%
attrib_data = shap.sample(x_train_tfidf_ngram, 20)
explainer = shap.KernelExplainer(logreg_model.predict_proba, attrib_data)
shap_vals = explainer.shap_values(x_test_tfidf_ngram)
# %%
shap.summary_plot(shap_vals, feature_names=vocab, class_names=CLASSES)
# %%
shap.decision_plot(explainer.expected_value[0], shap_vals[0][1], feature_names=vocab,
    feature_display_range=slice(None, -31, -1))

# %% [markdown]
## Use SHAP to interpret our results!
#attrib_data = x_train_tfidf_ngram[:200]
attrib_data = shap.sample(x_train_tfidf_ngram, nsamples=10)
explainer = shap.KernelExplainer(svm_model.decision_function, attrib_data)

# %%
num_explanations = 5
shap_vals = explainer.shap_values(x_test_tfidf_ngram)

# %%
shap.decision_plot(explainer.expected_value[0], shap_vals[0][100], feature_names=vocab,
    feature_display_range=slice(None, -31, -1))

# %%
shap.force_plot(explainer.expected_value[0], shap_vals[0][0,:],
    x_test_tfidf_ngram.iloc[0,:], link="logit", matplotlib=True)
    
# %%
shap.summary_plot(shap_vals, feature_names=vocab, class_names=CLASSES)

# %%
shap.summary_plot(shap_vals, feature_names=vocab, plot_type="bar")

# %%
shap.force_plot(explainer.expected_value[2], shap_vals[2][0,:],
    x_test_tfidf_ngram.iloc[0,:])

# %% THIS WORKS!!!!
shap.decision_plot(explainer.expected_value[0], shap_vals[0][100], feature_names=vocab,
    feature_display_range=slice(None, -31, -1))

# %%
shap.multioutput_decision_plot(explainer.expected_value, shap_vals, row_index=0)

# %%
import warnings
explainer = shap.TreeExplainer(dectree_model, model_output='probability', feature_perturbation='interventional')
expected_value = explainer.expected_value
if isinstance(expected_value, list):
    expected_value = expected_value[0]
print(f"Explainer expected value: {expected_value}")

select = range(20)
features = x_test_tfidf_ngram.iloc[select]
#features_display = X_display.loc[features.index]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    shap_values = explainer.shap_values(features, check_additivity=False)[0]
    shap_interaction_values = explainer.shap_interaction_values(features)
if isinstance(shap_interaction_values, list):
    shap_interaction_values = shap_interaction_values[1]

# %%
shap.decision_plot(expected_value, shap_values, vocab)

# %% TRY ON TREE
explainer2 = shap.TreeExplainer(dectree_model)

# %%
shap_vals2 = explainer2.shap_values(x_test_tfidf_ngram, check_additivity=False)

# %%
shap.force_plot(explainer2.expected_value[0], shap_vals2[0][0,:],
    x_test_tfidf_ngram.iloc[0,:], link="logit", matplotlib=True)
    
# %%
shap.summary_plot(shap_vals2, feature_names=vocab, class_names=CLASSES)

# %%
shap.summary_plot(shap_vals2, feature_names=vocab, plot_type="bar")

# %%
row_index = 0
shap.multioutput_decision_plot(list(explainer2.expected_value), shap_vals2,
                               row_index=row_index, 
                               feature_names=vocab, 
                               highlight=y_test[row_index]-1)

# %%
shap.decision_plot(explainer2.expected_value[0], shap_vals2[0][1], feature_names=vocab,
    feature_display_range=slice(None, -31, -1))

# %%
shap.initjs()  
 
shap.force_plot(explainer2.expected_value[0], shap_vals2[0][10], x_test_tfidf_ngram.iloc[10, :])

# # %% [markdown]
# # ## ALIBI for interpretation
# import alibi
# from sklearn.metrics import accuracy_score
# from alibi.explainers import AnchorText
# from alibi.explainers import KernelShap
# # from alibi.datasets import fetch_movie_sentiment

# # %%
# pred_fcn = svm_model.predict
# svm_explainer = KernelShap(pred_fcn, link='raw', feature_names=vocab)
# svm_explainer.fit(x_train_tfidf_ngram, summarise_background=True)

# #%%
# explainer = KernelShap(predict_fn, link='logit', feature_names=vocab)

# # %%
# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     explainer.fit(x_test_tfidf_ngram, summarise_background=True)

# # %%
# explanation = explainer.explain(np.array(x_test_tfidf_ngram.iloc[0, :]).reshape(1,-1))


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