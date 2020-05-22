# %% [markdown]
# # Interpretability with ELI 5

# %% [markdown]
# ## Import dependencies
import numpy
import pandas
from sklearn.preprocessing import label_binarize
from sklearn.feature_extraction.text import CountVectorizer
import sqlite3
from sklearn.multiclass import OneVsRestClassifier
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import eli5

# %% [markdown]
# ## Define Constants
N_CLASSES = 4
RANDOM_STATE = 123

# %% [markdown]
# #### Load in the data from the database

# %%
dbconn = sqlite3.connect('./data/cleanedtraintest_v2.db')
train_data_df = pandas.read_sql_query(
    'SELECT category, content_cleaned FROM train_data', dbconn)
test_data_df = pandas.read_sql_query(
    'SELECT category, content_cleaned FROM test_data', dbconn)
dbconn.commit()
dbconn.close()

# %% [markdown]
# #### Check the if the data was loaded correctly

# %%
train_data_df

# %%
test_data_df

# %%
# sample data
train_data_sample = train_data_df.sample(n=3000, replace=False, random_state=123)
train_data_sample.head()

# %% 
test_data_sample = test_data_df #.sample(n=4000, replace=False, random_state=123)
test_data_sample.head()

# %% [markdown]
# #### Train & Test data where x is the predictor features, y is the predicted feature

x_train = train_data_sample.content_cleaned
y_train = label_binarize(train_data_sample.category, classes=range(1, N_CLASSES + 1))

x_test = test_data_df.content_cleaned
y_test = label_binarize(test_data_sample.category, classes=range(1, N_CLASSES + 1))

# %% [markdown]
# ## Count Vectorizer
cv = CountVectorizer(min_df = 2, lowercase = True, token_pattern=r'(?u)\b[A-Za-z]{2,}\b', 
                        strip_accents = 'ascii', ngram_range = (2, 3), 
                        stop_words = 'english')
cv_matrix = cv.fit_transform(train_data_sample.content_cleaned).toarray()

#%%
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

#%%
# get all unique words in the corpus
bow_vocab = cv.get_feature_names()

#%%
# produce a dataframe including the feature names
x_train_bagofwords = pandas.DataFrame(x_train_cv.toarray(), columns=bow_vocab)
x_test_bagofwords = pandas.DataFrame(x_test_cv.toarray(), columns=bow_vocab)

# %% [markdown]
# ## Build Classification Model

# %% [markdown]
# ### Logistic Regression Model Building Function
def run_logreg(x_train, y_train):
    classifier = OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE))
    classifier.fit(x_train, y_train)
    return classifier

# %%
# Run Logistic Regression Model
logreg_model = run_logreg(x_train_bagofwords, y_train)

#%%
# ## ELI 5

print('Estimator: ' % (['Logistic Regression']))

# Global Explanation
eli5.show_weights(estimator = logreg_model, 
                top = 10, 
                target_names = ['W','S','B','T'], 
                feature_names = bow_vocab)

# %%
#Local Explanation
eli5.show_prediction(estimator = logreg_model, 
                    doc = x_test_bagofwords.values[764], 
                    target_names = ['W','S','B','T'], 
                    feature_names = bow_vocab)
                                    

    


# %% [markdown]
# ## References - Code sample sources disclaimer:
# Code for this project is either directly from (with some modification), 
# or inspired by, but not limited to the following sources:
# - Respective documentation and examples from each used API's doc/guide website
# - Kelly Epley Naive Bayes: 
#   https://towardsdatascience.com/naive-bayes-document-classification-in-python-e33ff50f937e
# - MLWhiz's excellent blogs about text classification and NLP: 
#   https://mlwhiz.com/blog/2018/12/17/text_classification/
#   https://mlwhiz.com/blog/2019/01/17/deeplearning_nlp_preprocess/
#   https://mlwhiz.com/blog/2019/02/08/deeplearning_nlp_conventional_methods/
#   https://www.kaggle.com/mlwhiz/conventional-methods-for-quora-classification/
# - Christof Henkel preprocessing: 
#   https://www.kaggle.com/christofhenkel/how-to-preprocessing-when-using-embeddings
# - datanizing GmbH:
#   https://medium.com/@datanizing/modern-text-mining-with-python-part-1-of-5-introduction-cleaning-and-linguistics-647f9ec85b6a
# - Datacamp wordcloud:
#   https://www.datacamp.com/community/tutorials/wordcloud-python
# - Seaborn Pydata tutorials:
#   https://seaborn.pydata.org/introduction.html#intro-plot-customization
# - Dipanjan S's tutorials:
#   https://github.com/dipanjanS
# - Analytics Vidhya:
#   https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
# - Jason Brownlee's Feature Selection For Machine Learning in Python
#   https://machinelearningmastery.com/feature-selection-machine-learning-python/
# - Susan Li's Multi-class text classification with Scikit-learn:
#   https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
# - Vadim Smolyakov Ensemble Learning to Improve Machine Learning Results:
#   https://blog.statsbot.co/ensemble-learning-d1dcd548e936
# - Udacity course video on Youtube UD120:
#   https://www.youtube.com/watch?v=GdsLRKjjKLw
# - Hyperparameter Tuning with Hyperopt
#   https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a 
# - Hyperparameter Tuning for Gaussian NB
#   https://www.quora.com/Can-the-prior-in-a-naive-Bayes-be-considered-a-hyperparameter-and-tuned-for-better-accuracy
# - Hyperparameter Tuning for Decision Trees
#   https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
# - Lime tutorial
#   https://marcotcr.github.io/lime/tutorials/Lime%20-%20multiclass.html
# - ELI5
#   https://towardsdatascience.com/3-ways-to-interpretate-your-nlp-model-to-management-and-customer-5428bc07ce15
# - ELI 5 Text
#   https://eli5.readthedocs.io/en/latest/autodocs/sklearn.html
# - ELI 5 BIAS
#   https://stackoverflow.com/questions/49402701/eli5-explaining-prediction-xgboost-model


# %%
