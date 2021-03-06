#!/usr/bin/env python
# coding: utf-8

# # Partial Dependence Plots:# ## Importing the libraries needed:
# If you can't import shap consider installing the visual package it suggests from the microsoft website.

# edit for other courses
week_type = 'eq_week'
feature_types = [ "boroujeni_et_al", "chen_cui", "marras_et_al", "lalle_conati"]
course = 'dsp_001'
remove_obvious = True
run_ice_plots = False
# In[ ]:


# uncomment to install shap if not installed
# get_ipython().system('pip install shap')

import json
import shap
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as pyplot
import seaborn as sns
import math
import time


# In[ ]:
# This module fills the NaNs in our data as LIME library can't handle data with NaNs in it.
def fillNaN(feature):
    shape = feature.shape
    feature_min = np.nanmin(
        feature.reshape(-1, shape[2]), axis=0
    )  # min of that feature over all weeks
    feature = feature.reshape(-1, shape[2])
    inds = np.where(np.isnan(feature))
    feature[inds] = np.take(feature_min.reshape(-1), inds[1])
    feature = feature.reshape(shape)
    return feature

# Loading feature names
feature_names= []
for feature_type in feature_types:
    filepath = './scripts/feature_names/' + feature_type + '.csv'
    if feature_type == "lalle_connati":
        filepath = './scripts/feature_names/lalle_conati.csv'
    feature_type_name = pd.read_csv(filepath,header=None)
    feature_type_name = feature_type_name.values.reshape(-1)
    feature_names.append(feature_type_name)

if remove_obvious:
  new_marras = feature_names[2][[2,3,4,5]]
  feature_names[2] = new_marras

# loading the labels
feature_type = "boroujeni_et_al"
filepath = './data/' + week_type + '-' + feature_type + '-' + course + '/feature_labels.csv'
labels = pd.read_csv(filepath)['label-pass-fail']
labels[labels.shape[0]] = 1
y = labels.values
# Loading the features
selected_features = []
feature_list = []
num_weeks =0
n_features=0
for i,feature_type in enumerate(feature_types):
    filepath = './data/' + week_type + '-' + feature_type + '-' + course
    if feature_type == "lalle_connati":
        filepath = './feature/eq_week-lalle_conati-' + course
    feature_current = np.load(filepath+'/feature_values.npz')['feature_values']
    if remove_obvious and feature_type=='marras_et_al':
      feature_current = np.delete(feature_current, [0,1,6], axis=2)
    shape = feature_current.shape
    nonNaN = (shape[0]*shape[1] - np.isnan(feature_current.reshape(-1,feature_current.shape[2])).sum(axis=0) > 0)
    feature_current = feature_current[:,:,nonNaN]
    selected = np.arange(shape[2])
    selected = selected[nonNaN]
    feature_current = fillNaN(feature_current)
    nonZero = (abs(feature_current.reshape(-1,feature_current.shape[2])).sum(axis=0)>0)
    selected = selected[nonZero]
    feature_current = feature_current[:,:,nonZero]
    selected_features.append(feature_names[i][selected])
    num_weeks = feature_current.shape[1]
    feature_current = fillNaN(feature_current)
    print(feature_current.shape)
    feature_list.append(feature_current)
features = np.concatenate(feature_list, axis=2)
[print(f.shape) for f in feature_list]
features_min = features.min(axis=0).reshape(-1)
features_max = features.max(axis=0)
# Here we reshape the data to a 2D dataframe for explainability goals.
features = features.reshape(features.shape[0],-1)
features = pd.DataFrame(features)
print('course: ', course)
print('week_type: ', week_type)
print('feature_type: ', feature_types)

print(num_weeks)

# calculate the number of features
n_features = sum([len(x) for x in selected_features])

# Loading feature names and transforming them to 2D format.
feature_names = []
final_features = []
for i,feature_type in enumerate(feature_types):
    [final_features.append(x) for x in selected_features[i]]
for i in np.arange(num_weeks):
    feature_type_name_with_weeks = [(x+'_InWeek'+str(i+1)) for x in final_features]
    feature_names.append(feature_type_name_with_weeks)
feature_names = np.concatenate(feature_names, axis=0)
feature_names = feature_names.reshape(-1)
features.columns = feature_names

# This block loads number of features in each feature set.
num_feature_type = []
for i,feature_type in enumerate(feature_types):
   num_feature_type.append(len(selected_features[i]))
print(num_feature_type)

# ## Making a predict_proba

# This module transforms our data to the 2D format biLSTM was trained with.
def transform_x(x, num_feature_type, num_weeks, features_min, features_max):
    x = np.array(x)
    num_feature_type = np.array(num_feature_type)
    num_features = num_feature_type.sum()
    x = x.reshape((-1, num_weeks, num_features))
    shape = x.shape
    features_max = np.where(
        features_max == 0, np.ones(features_max.shape), features_max
    )
    max_instance = 1.001 * features_max
    feature_current = np.vstack([x, max_instance.reshape((1,) + max_instance.shape)])
    features_max = features_max.reshape(-1)
    feature_norm = (feature_current.reshape(shape[0] + 1, -1) - features_min) / (
        1.001 * features_max - features_min
    )
    x = feature_norm[: feature_norm.shape[0] - 1, :]
    return x

print(features.shape)
model_name = "models/lstm_bi_"+course+"_new"
loaded_model = keras.models.load_model(model_name)
# This lambda returns a (NUM OF INSTANCES,2) array of prob of pass in first column and prob of fail in another column
predict_fn = lambda x: np.array([1-loaded_model.predict(transform_x(x,num_feature_type,num_weeks,features_min,features_max))]).T


Background_distribution = shap.utils.sample(features, 8500)

top_features = feature_names


for i,f in enumerate(top_features):
  fig = shap.plots.partial_dependence(
      i, predict_fn,Background_distribution, ice=False,
      model_expected_value=True, feature_expected_value=True,show=False
  )
  pyplot.title('Partial dependence plot for '+f)
  pyplot.savefig("./uniform_eq_results/PDP/" + course + "/Permutation/plots/"+"PDP_"+f+".png", bbox_inches = 'tight')
  pyplot.show()


if run_ice_plots:
    for f in top_features:
    fig = shap.plots.partial_dependence(
        f, predict_fn,Background_distribution , ice=True,
        model_expected_value=True, feature_expected_value=True,show=False
    )
    pyplot.title('Partial dependence plot for '+f)
    pyplot.savefig("./uniform_eq_results/PDP/" + course + "/Kernel/plots/"+"PDPwithICE_"+f+".png", bbox_inches = 'tight')
    pyplot.show()