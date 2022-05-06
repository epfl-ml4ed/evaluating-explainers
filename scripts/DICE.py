#!/usr/bin/env python
# coding: utf-8

# # Estimating local and global feature importance scores using DiCE
# 
# Summaries of counterfactual examples can be used to estimate importance of features. Intuitively, a feature that is changed more often to generate a proximal counterfactual is an important feature. We use this intuition to build a feature importance score. 
# 
# This score can be interpreted as a measure of the **necessity** of a feature to cause a particular model output. That is, if the feature's value changes, then it is likely that the model's output class will also change (or the model's output will significantly change in case of regression model).  
# 
# Below we show how counterfactuals can be used to provide local feature importance scores for any input, and how those scores can be combined to yield a global importance score for each feature.

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import dice_ml
from dice_ml import Dice
import matplotlib.pyplot as pyplot
import seaborn as sns
import tensorflow as tf
import json



# ## Preliminaries: Loading the data and ML model

# In[ ]:


# edit here for other courses and params
week_type = 'eq_week'
feature_types = [ "boroujeni_et_al", "chen_cui", "marras_et_al", "lalle_conati"]
course = 'microcontroleurs_003'
remove_obvious = True


# In[ ]:


# edit directory name for other courses !
# get_ipython().system('mkdir -p uniform_eq_results/Counterfactuals/' + course)


# In[ ]:


# Loading the features
feature_list = []
feature_type_list = []
for feature_type in feature_types:
  filepath = 'data/all/' + week_type + '-' + feature_type + '-' + course
  feature_current = np.load(filepath+'/feature_values.npz')['feature_values']
  if feature_type=='marras_et_al' and remove_obvious:
    feature_current = np.delete(feature_current, [0,1,6], axis=2)

  print(feature_current.shape)
  feature_norm = feature_current.reshape(-1,feature_current.shape[2] )
  print(feature_norm.shape)
  feature_type_list.append(pd.DataFrame(feature_norm))
  feature_list.append(feature_type_list)

print('course: ', course)
print('week_type: ', week_type)
print('feature_type: ', feature_types)


# In[ ]:


# Loading feature names
feature_names= []
for feature_type in feature_types:
    filepath = 'scripts/feature_names/' + feature_type + '.csv'
    feature_type_name = pd.read_csv(filepath,header=None)
    feature_type_name = feature_type_name.values.reshape(-1)
    feature_names.append(feature_type_name)


# In[ ]:


if remove_obvious:
  new_marras = feature_names[2][[2,3,4,5]]
  feature_names[2] = new_marras


# In[ ]:


def clean_name(feature):
  id = feature.find('<')
  if id==-1:
    return feature
  fct = feature[id+9:id+14].strip()
  return feature[0:id]+fct


# In[ ]:


feature_names = [np.array([clean_name(x) for x in feature_names[0]]),
 np.array([clean_name(x) for x in feature_names[1]]),
 np.array([clean_name(x) for x in feature_names[2]]),
 np.array([clean_name(x) for x in feature_names[3]])]


# In[ ]:


def fillNaN(feature):
    shape = feature.shape
    feature_min = np.nanmin(feature.reshape(-1,shape[2]),axis=0)#min of that feature over all weeks
    feature = feature.reshape(-1,shape[2])
    inds = np.where(np.isnan(feature))
    feature[inds] = np.take(feature_min.reshape(-1), inds[1])
    feature = feature.reshape(shape)
    return feature


# In[ ]:


# loading the labels
feature_type = "boroujeni_et_al"
filepath = 'data/all/' + week_type + '-' + feature_type + '-' + course + '/feature_labels.csv'
labels = pd.read_csv(filepath)['label-pass-fail']
labels[labels.shape[0]] = 1
y = labels.values
# Loading the features
feature_list = []
selected_features = []
num_weeks=0
n_features=0
for i,feature_type in enumerate(feature_types):
    filepath = 'data/all/' + week_type + '-' + feature_type + '-' + course
    feature_current = np.load(filepath+'/feature_values.npz')['feature_values']

    if remove_obvious and feature_type=='marras_et_al':
      feature_current = np.delete(feature_current, [0,1,6], axis=2)

    shape = feature_current.shape
    if i==0:
      num_weeks = shape[1]
    nonNaN = (shape[0]*shape[1] - np.isnan(feature_current.reshape(-1,feature_current.shape[2])).sum(axis=0) > 0)
    feature_current = feature_current[:,:,nonNaN]
    selected = np.arange(shape[2])
    selected = selected[nonNaN]
    feature_current = fillNaN(feature_current)
    nonZero = (abs(feature_current.reshape(-1,feature_current.shape[2])).sum(axis=0)>0)
    selected = selected[nonZero]
    feature_current = feature_current[:,:,nonZero]
    selected_features.append(feature_names[i][selected])
    n_features += len(feature_names[i][selected])
    ##### Normalization with min-max. I added the artifical 1.001 max row for solving the same min max problem
    ##### for features with max=0 I added 1 instead of 1.001 of maximum

    features_min = feature_current.min(axis=0).reshape(-1)
    features_max = feature_current.max(axis=0)
    features_max = np.where(features_max==0,np.ones(features_max.shape),features_max)
    max_instance = 1.001*features_max
    feature_current = np.vstack([feature_current,max_instance.reshape((1,)+max_instance.shape)])
    features_max = features_max.reshape(-1)
    feature_norm = (feature_current.reshape(shape[0]+1,-1)-features_min)/(1.001*features_max-features_min)
    feature_current = feature_norm.reshape(-1,feature_current.shape[1],feature_current.shape[2] )

    feature_list.append(feature_current)
features = np.concatenate(feature_list, axis=2)
features = features.reshape(features.shape[0],-1)
#features = pd.DataFrame(features)
SHAPE = features.shape
# print(np.isnan(features[0,0,-1]))
print(features.shape)
print('course: ', course)
print('week_type: ', week_type)
print('feature_type: ', feature_types)
print(selected_features)


# In[ ]:


selected_features={
    "boroujeni_et_al":list(selected_features[0]),
     "chen_cui":list(selected_features[1]),
    "marras_et_al":list(selected_features[2]),
    "lalle_conati":list(selected_features[3])
}
# Loading feature names and transforming them to 2D format.
feature_names= []
final_features = []
for feature_type in feature_types:
    [final_features.append(x) for x in selected_features[feature_type]]
for i in np.arange(num_weeks):
    feature_type_name_with_weeks = [(x+'_InWeek'+str(i+1)) for x in final_features]
    feature_names.append(feature_type_name_with_weeks)
feature_names = np.concatenate(feature_names, axis=0)
feature_names = feature_names.reshape(-1)

target = y
features = features.reshape(features.shape[0],-1)
features = pd.DataFrame(features,columns = feature_names)

numerical = features.columns
# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
transformations = ColumnTransformer(
    transformers=[('num', numeric_transformer, numerical)])

features['result'] = target


loaded_model = tf.keras.models.load_model("./lstm_bi_"+course+"_new")
print('loaded model')

backend = 'TF'+tf.__version__[0]
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

d = dice_ml.Data(dataframe=features, continuous_features=list(numerical.values), outcome_name='result')
m = dice_ml.Model(model=loaded_model, backend=backend)
print('dice model')


# ## Local feature importance
# 
# We first generate counterfactuals for a given input point. 

# In[ ]:


num_instances = np.load('uniform_'+course+'.npy')


# In[ ]:


import time
t1 = time.time()
exp = Dice(d, m)
t2 = time.time()
print(f'time taken: {t2-t1}')
query_instance = features[features.index.isin(num_instances)]
query_instance = query_instance.drop('result', axis=1)

for i in query_instance.index:
  print(i)
  x = pd.DataFrame(query_instance.loc[i]).T


  t1 = time.time()
  e = exp.generate_counterfactuals(x, total_CFs=10,
                                  desired_class="opposite")
  t2 = time.time()
  print(f'time taken to generate cfs for instance {i}: {t2-t1}')

  if i%10==0:
    e.visualize_as_dataframe(show_only_changes=True)

  e_json = e.to_json()
  e_dict = json.loads(e_json)

  for j in range(len(e_dict['cfs_list'][0])):
    cf = np.array(e_dict['cfs_list'][0][j])[:-1].reshape(1,-1)
    fpath = './uniform_eq_results/Counterfactuals/'+course+f'/{i}'+f'_cf_{j}.csv'
    pd.DataFrame(cf, columns=e_dict['feature_names']).to_csv()

  imps = exp.local_feature_importance(x, cf_examples_list=e.cf_examples_list)
  importances = pd.DataFrame(imps.local_importance)
  importances.insert(0, 'exp_num', i)
  importances.to_csv('./uniform_eq_results/Counterfactuals/' + course + '/'+f'feature_importances_{i}'+'.csv')
  

