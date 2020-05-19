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
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
from sklearn import svm
from itertools import cycle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix
from sklearn.metrics import make_scorer
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_learning_curves
import lime
import lime.lime_tabular

# %% [markdown]
# ## Define Constants
W2V_FEATURE_SIZE = 300
N_CLASSES = 4
RANDOM_STATE = 123
N_FOLDS = 5

# %% [markdown]
# ## Read in the data

# %% [markdown]
# ### Load raw train and test data

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

# %% [markdown]
# #### Train & Test data where x is the predictor features, y is the predicted feature

x_train = train_data_df.content_cleaned
y_train = label_binarize(train_data_df.category, classes=range(1, N_CLASSES + 1))

x_test = test_data_df.content_cleaned
y_test = label_binarize(test_data_df.category, classes=range(1, N_CLASSES + 1))

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
w2v_feature_array_train = averaged_word_vectorizer(corpus=tokenized_corpus_train,
    model=w2v_model_train, num_features=W2V_FEATURE_SIZE)
w2v_feature_array_test = averaged_word_vectorizer(corpus=tokenized_corpus_test,
    model=w2v_model_train, num_features=W2V_FEATURE_SIZE)

x_train_w2v = pandas.DataFrame(w2v_feature_array_train)
x_test_w2v = pandas.DataFrame(w2v_feature_array_test)

# %% [markdown]
# #### Sample down for speed, for now.
x_train_w2v = x_train_w2v.sample(
    n = 3000, replace = False, random_state = RANDOM_STATE
)
y_train = train_data_df.category.sample(
    n = 3000, replace = False, random_state = RANDOM_STATE
)
y_train = label_binarize(y_train, classes=range(1, N_CLASSES + 1))

# %% [markdown]
# #### Delete variables we don't need anymore to save memory
# del(w2v_feature_array_test)
# del(w2v_feature_array_train)
# del(w2v_test_features_array_dict)
# del(w2v_train_features_array_dict)
# del(tokenized_corpus_test)
# del(tokenized_corpus_train)
# del(wpt)
# del(train_data_df)
# del(test_data_df)
# del(x_train)
# del(x_test)
# del(data)

# %% [markdown]
# ## Build Models

# %% [markdown]
# ### SVM Model Building Function
def run_svm(x_train, y_train):
    classifier = OneVsRestClassifier(svm.LinearSVC(random_state=RANDOM_STATE))
    classifier.fit(x_train, y_train)
    return classifier

# %% [markdown]
# ### Logistic Regression Model Building Function
def run_logreg(x_train, y_train):
    classifier = OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE))
    classifier.fit(x_train, y_train)
    return classifier

# %% [markdown]
# ### Naive Bayes Function
def run_nb(x_train, y_train):
    classifier = OneVsRestClassifier(GaussianNB())
    classifier.fit(x_train, y_train)
    return classifier

# %% [markdown]
# ### Decision Trees Function
def run_dectree(x_train, y_train):
    classifier = OneVsRestClassifier(tree.DecisionTreeClassifier())
    classifier.fit(x_train, y_train)
    return classifier 

# %% [markdown]
# ### Functions to calculate scores and to plot them

# Calculate, then plot the Precision, Recall, Average Precision, F1
def prf1_calc(classifier, algo_name, n_classes, x_test, y_test):
    # Get the decision function from the classifier
    if algo_name == 'SVM':
        y_score = classifier.decision_function(x_test)
    else:
        y_score = classifier.predict_proba(x_test)
    y_pred = classifier.predict(x_test)

    # The average precision score in multi-label settings
    # For each class
    precision = dict()
    recall = dict()
    average_f1 = dict()
    average_precision = dict()
    mcc = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_score[:, i])
        average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
        average_f1[i] = f1_score(y_test[:, i], y_pred[:, i])
        mcc[i] = matthews_corrcoef(y_test[:, i], y_pred[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
        y_score.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_score,
                                                        average="micro")
    average_f1['micro'] = f1_score(y_test, y_pred, average='micro')
    mcc['micro'] = sum(mcc.values())/4

    # Plot the data
    prf1_plot(precision, recall, average_precision, algo_name, n_classes)

    # Return all metrics
    results = pandas.DataFrame()

    for k in average_precision.keys():
        results.at[algo_name, f'P-R {k}'] = numpy.round(average_precision[k], 3)
        results.at[algo_name, f'F1 {k}'] = numpy.round(average_f1[k], 3)
        results.at[algo_name, f'MCC {k}'] = numpy.round(mcc[k], 3)

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
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
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
# ## Run the Base Models

# %%
# Run SVM Model
svm_model = run_svm(x_train_w2v, y_train)

# %%
# Run Logistic Regression Model
logreg_model = run_logreg(x_train_w2v, y_train)

# %% 
# Run Naive Bayes Classifier
nb_model = run_nb(x_train_w2v, y_train)

# %%
# Run Decision Trees Classifier
dectree_model = run_dectree(x_train_w2v, y_train)

# %% [markdown]
# ## Get the scores

# %%
# Initialize the dataframe to keep track of the scores
scores = pandas.DataFrame()

# %% 
# Precision, Recall, Avg. Precision for SVM
scores = scores.append(prf1_calc(svm_model, 'SVM', N_CLASSES, x_test_w2v, y_test))

# %% 
# Precision, Recall, Avg. Precision for LOG REG
scores = scores.append(prf1_calc(logreg_model, 'LOGREG', N_CLASSES, x_test_w2v, y_test))

# %% 
# Precision, Recall, Avg. Precision for Naive Bayes
scores = scores.append(prf1_calc(nb_model, 'NB', N_CLASSES, x_test_w2v, y_test))

# %% 
# Precision, Recall, Avg. Precision for Decision Trees
scores = scores.append(prf1_calc(dectree_model, 'DT', N_CLASSES, x_test_w2v, y_test))

# %% [markdown]
# ## Look at Cross-Validation

# %% Create model list to iterate through for cross validation
gnb = OneVsRestClassifier(GaussianNB())
sv = OneVsRestClassifier(svm.LinearSVC(random_state=RANDOM_STATE))
lreg = OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE))
dtree = OneVsRestClassifier(tree.DecisionTreeClassifier())

model_list = [gnb, sv, lreg, dtree]
model_namelist = ['Gaussian Naive Bayes',
                  'SVM/Linear SVC', 
                  'Logistic Regression', 
                  'Decision Tree']

# %% Make scoring metrics to pass cv function through
scoring = {'precision': make_scorer(precision_score, average='micro'), 
           'recall': make_scorer(recall_score, average='micro'), 
           'f1': make_scorer(f1_score, average='micro'),
           'roc_auc': make_scorer(roc_auc_score, average='micro'),
           # 'mcc': make_scorer(matthews_corrcoef) <- cannot support multi-label
          }

cv_result_entries = []
i = 0

# %% Loop cross validation through various models and generate results
for mod in model_list:
    metrics = cross_validate(
        mod,
        x_train_w2v,
        y_train,
        cv=N_FOLDS,
        scoring = scoring,
        return_train_score=False,
        n_jobs=-1
    )
    for key in metrics.keys():
        for fold_index, score in enumerate(metrics[key]):
            cv_result_entries.append((model_namelist[i], fold_index, key, score))
    i += 1


# %% 
#cv_result_entries = pandas.read_csv('./data/cv-results.csv')
cv_results_df = pandas.DataFrame(cv_result_entries)
#cv_results_df.drop('Unnamed: 0', axis=1, inplace=True)
cv_results_df.columns = ['algo', 'cv fold', 'metric', 'value']
#test_df = pandas.DataFrame((cv_results_df[cv_results_df.metric.eq('fit_time')]))

# %% Plot cv results
for metric_name, metric in zip(['fit_time',
                                'test_precision',
                                'test_recall',
                                'test_f1',
                                'test_roc_auc'],
                                ['Fit Time',
                                'Precision',
                                'Recall',
                                'F1 Score',
                                'ROC AUC']):
    sns.boxplot(x='algo', y='value', #hue='algo',
        data=cv_results_df[cv_results_df.metric.eq(f'{metric_name}')])
    sns.stripplot(x='algo', y = 'value', 
        data = cv_results_df[cv_results_df.metric.eq(f'{metric_name}')],
        size = 5, linewidth = 1)
    plt.title(f'{metric} Algo Comparison', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel(f'{metric}', fontsize=12)
    plt.xticks([0, 1, 2, 3, 4])
    plt.xticks(rotation=45)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

# %% Misclassification Errors
i=0
for model in model_list:
    plt.figure()
    plot_learning_curves(x_train_w2v, y_train, x_test_w2v, y_test, model)
    plt.title('Learning Curve for ' + model_namelist[i], fontsize=14)
    plt.xlabel('Training Set Size (%)', fontsize=12)
    plt.ylabel('Misclassification Error', fontsize=12)
    plt.show()
    i += 1

# %% Get predictions
y_test_pred = []
for model in model_list:
    y_test_pred.append(model.predict(x_test_w2v))

# %% Confusion Matrix
CLASSES = ['World', 'Sports', 'Business', 'Sci/Tech']
i=0
for _ in model_list:
    cm = confusion_matrix(numpy.argmax(y_test, axis=1),
                          numpy.argmax(y_test_pred[i], axis=1))
    cm_df = pandas.DataFrame(cm, index = CLASSES, columns = CLASSES)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    plt.title('Confusion Matrix for ' + model_namelist[i], fontsize=14)
    sns.heatmap(cm_df, annot=True, fmt='.6g', annot_kws={"size": 10}, cmap='Reds')
    plt.show()
    i += 1

# %% HYPER PARAMETER TUNING BY HYPEROPT (not working)
'''from hyperopt import STATUS_OK
N_FOLDS = 5
#%% 
#Objective Function
def objective(params, n_folds = N_FOLDS):
    cv_results = cross_validate(OneVsRestClassifier(GaussianNB()),
        x_train_w2v,
        y_train,
        cv = n_folds,
        fit_params= params,
        scoring = {'f1': make_scorer(f1_score, average='micro')},
        return_train_score=False,
        n_jobs=-1
    )
    # Extract the best score
    best_score = max(cv_results['test_f1'])
    # Loss must be minimized
    loss = 1 - best_score
    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'status': STATUS_OK}
# %%
#Domain Space
from hyperopt import hp
space = {'estimator__var_smoothing': hp.uniform('estimator__var_smoothing', 
                          1.e-09, 1.e+00)}
#%%
# Optimization Algorithm
from hyperopt import tpe
tpe_algo = tpe.suggest
#%% 
# Results History
from hyperopt import Trials
bayes_trials = Trials()
#%%
# Run the optimization
from hyperopt import fmin
from hyperopt import rand
MAX_EVALS = 500
params = space
# Optimize
best = fmin(fn = objective, space = space, algo = tpe.suggest, 
            max_evals = 100, trials = bayes_trials)
print(best)'''

# %% [markdown]
# ## Hyper-parameter tuning with exhaustive Grid Search

# ### Tune hyperparameters for Gaussian Naive-Bayes

params_gnb = {'estimator__var_smoothing': [1.e-09, 1.e-08, 1.e-07, 1.e-06, 1.e-05,
                                           1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00]
}
clf = GridSearchCV(estimator=gnb,
                   param_grid=params_gnb,
                   scoring='f1_micro',
                   n_jobs=-1,
                   cv=N_FOLDS,
                   return_train_score=True
                  )
clf_res = clf.fit(x_train_w2v, y_train)
print('Best Score: ', clf_res.best_score_)
print('Best Params: ', clf_res.best_params_)

# %%
# ### Tune hyperparameters for Logistic Regression

params_lreg = {
    "estimator__penalty": ['l1', 'l2'],
    "estimator__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #"estimator__class_weight":[{1:0.5, 0:0.5}, {1:0.4, 0:0.6},
    #                           {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
    "estimator__solver": ["newton-cg", "sag", "saga", "lbfgs"]
}
clf = GridSearchCV(estimator=lreg,
                   param_grid=params_lreg,
                   scoring='f1_micro',
                   n_jobs=-1,
                   cv=N_FOLDS,
                   return_train_score=True
                  )
clf_res = clf.fit(x_train_w2v, y_train)
print('Best score:', clf_res.best_score_)
print('Best Params:', clf_res.best_params_)

# %%
# ### Tune hyperparameters for SVM (Linear SVC)

params_sv = {
    "estimator__penalty":['l1', 'l2'],
    "estimator__tol": [1.e-08, 1.e-07, 1.e-06, 1.e-05,
                       1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00],
    "estimator__loss":['hinge','squared_hinge'],
    "estimator__C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    #"estimator__class_weight":['None',{1:0.5, 0:0.5}, 
    #                           {1:0.4, 0:0.6}, {1:0.6, 0:0.4}, {1:0.7, 0:0.3}],
}
clf = GridSearchCV(estimator=sv,
                   param_grid=params_sv,
                   scoring='f1_micro',
                   n_jobs=-1,
                   cv=N_FOLDS,
                   return_train_score=False
                  )
clf_res = clf.fit(x_train_w2v, y_train)
print('Best score:', clf_res.best_score_)
print('Best Params:', clf_res.best_params_)

# %%
# ### Tune hyperparameters for Decision Trees

params_dtree = {
    "estimator__splitter":["best", "random"],
    "estimator__min_samples_split":range(1, 20, 1)
}
clf = GridSearchCV(estimator=dtree,
                   param_grid=params_dtree,
                   scoring='f1_micro',
                   n_jobs=-1,
                   cv=N_FOLDS,
                   return_train_score=False
                  )
clf_res = clf.fit(x_train_w2v, y_train)
print('Best score:', clf_res.best_score_)
print('Best Params:', clf_res.best_params_)

# %% [markdown]
# ## Ensemble Methods

# %% [markdown]
# ### Stacking
estimators = [
              ('nb', GaussianNB()),
              ('svm', svm.LinearSVC())
             ]

sclf = OneVsRestClassifier(StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression())
)

metrics = cross_validate(
    sclf,
    x_train_w2v,
    y_train,
    cv=N_FOLDS,
    scoring = scoring,
    return_train_score=False,
    n_jobs=-1
)

# %%
res = []
for key in metrics.keys():
    for fold_index, score in enumerate(metrics[key]):
        res.append(('Stacking', fold_index, key, score))

# %%
res_df = pandas.DataFrame.from_dict(res)
res_df.columns = ['algo', 'cv fold', 'metric', 'value']
cv_results_inc_ens = pandas.concat([cv_results_df, res_df])

# %% [markdown]
# ### Bagging
sclf = OneVsRestClassifier(BaggingClassifier(
    base_estimator=LogisticRegression())
)

metrics = cross_validate(
    sclf,
    x_train_w2v,
    y_train,
    cv=N_FOLDS,
    scoring = scoring,
    return_train_score=False,
    n_jobs=-1
)

# %% 
res = []
for key in metrics.keys():
    for fold_index, score in enumerate(metrics[key]):
        res.append(('Bagging', fold_index, key, score))

# %%
res_df = pandas.DataFrame.from_dict(res)
res_df.columns = ['algo', 'cv fold', 'metric', 'value']
cv_results_inc_ens = pandas.concat([cv_results_inc_ens, res_df])

# %% [markdown]
# ### Boosting
from sklearn.ensemble import AdaBoostClassifier
sclf = OneVsRestClassifier(AdaBoostClassifier(
    random_state=RANDOM_STATE)
)

metrics = cross_validate(
    sclf,
    x_train_w2v,
    y_train,
    cv=N_FOLDS,
    scoring = scoring,
    return_train_score=False,
    n_jobs=-1
)

# %% 
res = []
for key in metrics.keys():
    for fold_index, score in enumerate(metrics[key]):
        res.append(('AdaBoost', fold_index, key, score))

# %%
res_df = pandas.DataFrame.from_dict(res)
res_df.columns = ['algo', 'cv fold', 'metric', 'value']
cv_results_inc_ens = pandas.concat([cv_results_inc_ens, res_df])

# %%
#cv_results_inc_ens.to_csv('./data/cv-results-inc-ens.csv')

# %% [markdown]
# ### Plot the results including ensembling
for metric_name, metric in zip(['fit_time',
                                'test_precision',
                                'test_recall',
                                'test_f1',
                                'test_roc_auc'],
                                ['Fit Time',
                                'Precision',
                                'Recall',
                                'F1 Score',
                                'ROC AUC']):
    sns.lineplot(x='cv fold', y='value', hue='algo',
        data=cv_results_inc_ens[cv_results_inc_ens.metric.eq(f'{metric_name}')])
    plt.title(f'{metric} Algo Comparison', fontsize=12)
    plt.xlabel('CV Fold', fontsize=12)
    plt.ylabel(f'{metric}', fontsize=12)
    plt.xticks([0, 1, 2, 3, 4])
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

# %% [markdown]
# ## LIME for model interpretation

# predict_fn = model_namelist
# i = 0
# for model in model_list:
#     predict_fn[i] = lambda x: model.predict_proba(x).astype(float)
#     i += 1

# %%
feature_names = list(w2v_model_train.vocab)

# %%
import lime.lime_text
class_names=['W','S','B','T']
explainer = lime.lime_tabular.LimeTabularExplainer(x_test_w2v.values,
 feature_names=feature_names, mode='classification', class_names=class_names)

# %%
# use gnb (predict_fn[0]) 
idx = 34
exp = explainer.explain_instance(x_test_w2v.values[idx], lreg.predict_proba, num_features=20, top_labels=4)
print('Document id: %d' % idx)
# print('Predicted class =', class_names[lreg.predict(numpy.argmax(y_test, axis=1)[idx])])#.reshape(1,-1)[0,0]])
# print('True class: %s' % class_names[y_test[idx]])

# %%
print ('\n'.join(map(str, exp.as_list(label=0))))

# %%
exp.show_in_notebook()

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