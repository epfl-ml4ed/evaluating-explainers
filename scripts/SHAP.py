#!/usr/bin/env python
# coding: utf-8

# # SHAP experiments:

# ## Importing the libraries needed:

# In[ ]:


# If you can't import shap consider installing the visual package it suggests from the microsoft website.


# In[ ]:

# In[ ]:

import json
import shap
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
from tqdm.notebook import tqdm


# ## Loading the features and their names

# In[ ]:


# edit here for other courses !
week_type = 'eq_week'
feature_types = [ "boroujeni_et_al", "chen_cui", "marras_et_al", "lalle_conati"]
course = 'microcontroleurs_003'
remove_obvious = True


# In[ ]:


# We fill NaNs still in our data with this module. It fills NaNs with mean of that feature in that week. 
def fillNaN(feature):
    shape = feature.shape
    feature_min = np.nanmin(feature.reshape(-1,shape[2]),axis=0)#min of that feature ober all weeks
    feature = feature.reshape(-1,shape[2])
    inds = np.where(np.isnan(feature))
    feature[inds] = np.take(feature_min.reshape(-1), inds[1])
    feature = feature.reshape(shape)
    return feature


# In[ ]:


# get_ipython().system('mkdir -p uniform_eq_results/SHAP/dsp_001/Kernel/plots')
# get_ipython().system('mkdir -p uniform_eq_results/SHAP/dsp_001/Kernel/plots/global')
# get_ipython().system('mkdir -p uniform_eq_results/SHAP/dsp_001/Kernel/plots/forceplots')
# get_ipython().system('mkdir -p uniform_eq_results/SHAP/dsp_001/Permutation/plots/waterfalls')
# get_ipython().system('mkdir -p uniform_eq_results/SHAP/dsp_001/Permutation/plots/global')
# get_ipython().system('mkdir -p uniform_eq_results/SHAP/dsp_001/Permutation/plots/forceplots')


# In[ ]:


# Loading the features used in training in the biLSTM with loading the json file made in the biLSTM training notebook.
file = 'selected_features/' + course + '.json'
with open(file, 'r') as f:
    selected_features = json.load(f)


# In[ ]:


def clean_name(feature):
  id = feature.find('<')
  if id==-1:
    return feature
  fct = feature[id+9:id+14].strip()
  return feature[0:id]+fct


# In[ ]:


# Loading feature names
feature_names= []
for feature_type in feature_types:
    filepath = './scripts/feature_names/' + feature_type + '.csv'
    feature_type_name = pd.read_csv(filepath,header=None)
    feature_type_name = feature_type_name.values.reshape(-1)
    feature_names.append(feature_type_name)

if remove_obvious:
  new_marras = feature_names[2][[2,3,4,5]]
  feature_names[2] = new_marras


# In[ ]:


# loading the labels
feature_type = "boroujeni_et_al"
filepath = './data/all/' + week_type + '-' + feature_type + '-' + course + '/feature_labels.csv'
labels = pd.read_csv(filepath)['label-pass-fail']
labels[labels.shape[0]] = 1
y = labels.values
# Loading the features
feature_list = []
selected_features = []
n_weeks=0
n_features=0
for i,feature_type in enumerate(feature_types):
    filepath = './data/all/' + week_type + '-' + feature_type + '-' + course
    feature_current = np.load(filepath+'/feature_values.npz')['feature_values']

    if remove_obvious and feature_type=='marras_et_al':
      feature_current = np.delete(feature_current, [0,1,6], axis=2)

    shape = feature_current.shape
    if i==0:
      n_weeks = shape[1]
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
features = pd.DataFrame(features)
SHAPE = features.shape
# print(np.isnan(features[0,0,-1]))
print(features.shape)
print('course: ', course)
print('week_type: ', week_type)
print('feature_type: ', feature_types)
print(selected_features)
num_weeks = n_weeks


# In[ ]:


selected_features={
    "boroujeni_et_al":list(selected_features[0]),
     "chen_cui":list(selected_features[1]),
    "marras_et_al":list(selected_features[2]),
    "lalle_conati":list(selected_features[3])
}
print(selected_features)
file = 'selected_features/' + course + '_after.json'
with open(file, 'w') as f: 
    json.dump(selected_features, f)


# In[ ]:


# calculate the number of features
n_features = sum([len(x) for x in selected_features])


# In[ ]:


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
# print(feature_names)
features.columns = feature_names


# In[ ]:


# This block loads number of features in each feature set.
num_feature_type = []
for feature_type in feature_types:
   num_feature_type.append(len(selected_features[feature_type]))
print(num_feature_type)


# ## Making a predict_proba:

# Here we take the Keras model trained in its respective notebook and explain why it makes different predictions for different individuals. SHAP expects model functions to take a 2D numpy array as input, so we define a wrapper function around the original Keras predict function.

# In[ ]:


# This module transforms our data to the 2D format biLSTM was trained with.
def transform_x(x, num_feature_type, num_weeks, normal=True):
  try:
    r = np.array(x).reshape((x.shape[0],x.shape[1]))
  except IndexError:
    r = np.array(x).reshape((1,-1))
  return r


# In[ ]:


# EDIT HERE FOR OTHER MODELS
model_name = "lstm_bi_"+course+"_new"
loaded_model = keras.models.load_model(model_name)


# In[ ]:


predict_fn = lambda x: loaded_model.predict(transform_x(x,num_feature_type,num_weeks)).flatten()


# # Explaining biLSTM using SHAP:

# ## Modules for visualization:

# In[ ]:


def forceplot_all(shap_values,instances,features,real_labels,algorithm,group,max_display=10,show=True,explainer=None):
  for i,inst in enumerate(instances):
    print(inst)
    s='fail' if real_labels[inst] else 'pass'
    p=predict_fn(features.iloc[inst,:])
    title = ("Force plot for instance number {i:d} ".format(i=inst))+("with real label {}".format(s)+'({:.4f})'.format(p[0]))
    print(title)
    if (algorithm == 'p'):
      force = shap.force_plot(shap_values.base_values[i], shap_values.values[i], features.iloc[inst],contribution_threshold=0.05)
      shap.save_html("uniform_eq_results/SHAP/" + course + "/Permutation/plots/forceplots/"+str(inst)+".html", force)
    else:
      force = shap.force_plot(explainer.expected_value, shap_values[i], features.iloc[inst],contribution_threshold=0.05)
      shap.save_html("uniform_eq_results/SHAP/" + course + "/Kernel/plots/forceplots/"+str(inst)+".html", force)
  return
  


# Plots an explantion of a single prediction as a waterfall plot.
# 
# The SHAP value of a feature represents the impact of the evidence provided by that feature on the model’s output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature move the model output from our prior expectation under the background data distribution, to the final model prediction given the evidence of all the features. Features are sorted by the magnitude of their SHAP values with the smallest magnitude features grouped together at the bottom of the plot when the number of features in the models exceeds the max_display parameter.
# 
# 

# In[ ]:


def waterfall_all(shap_values,instances,features,real_labels,group,max_display=10,show=True):
  for i,inst in enumerate(instances):
    if show:
      plt.show()
    s='fail' if real_labels[inst] else 'pass'
    p=predict_fn(features.iloc[inst,:])
    title = ("Waterfall plot for instance number {i:d} ".format(i=inst))+("with real label {}".format(s)+'({:.4f})'.format(p[0]))
    plt.title(title)
    fig = shap.plots.waterfall(shap_values[i], max_display=max_display,show=False)
    plt.savefig("./uniform_eq_results/SHAP/" + course +"/Permutation/plots/waterfalls/"+str(inst)+".png", bbox_inches = 'tight')
  return


# # Instances chosen by SP-LIME:

# In[ ]:


# Loading the splime instances
num_instances = np.load('uniform_'+course+'.npy')


# ## permutation SHAP:

# In[ ]:


Background_distribution = shap.utils.sample(features, 100)
instances = features.iloc[num_instances]
explainer = shap.PermutationExplainer(predict_fn, Background_distribution)
shap_values = explainer(instances, max_evals=1500)


# In[ ]:


# the waterfall_plot shows how we get from shap_values.base_values to model.predict(X)[sample_ind]
waterfall_all(shap_values,num_instances,features,y,'r',max_display=15)


# In[ ]:


shaps = [[0.0 if np.isnan(x) else x for x in s] for s in shap_values.values]


# In[ ]:


shap.initjs()
forceplot_all(shap_values,num_instances,features,y,'p','r',max_display=15)


# In[ ]:


shap.plots.beeswarm(shap_values, max_display=14,show=False)
locs, labels = plt.yticks()  # Get the current locations and labels.
plt.title('global explanation of Uniform instances')
plt.savefig("./uniform_eq_results/SHAP/"+course+"/Permutation/plots/global/"+"summaryplot.png", bbox_inches = 'tight')
plt.show()
labels = labels[::-1]
top_features=[]
for i in np.arange(len(labels)):
  x=labels[i]
  top_features.append(x.get_text())
print(top_features[:10])
file = './uniform_eq_results/SHAP/'+course+'/Permutation/plots/top_features_permutation_splime.json'
with open(file, 'w') as f: 
    json.dump(top_features, f)


# Create a bar plot of a set of SHAP values.
# 
# If a single sample is passed then we plot the SHAP values as a bar chart. If an Explanation with many samples is passed then we plot the mean absolute value for each feature column as a bar chart.

# In[ ]:


fig = shap.plots.bar(shap_values,max_display=15,show=False)
plt.title('global explanation')
plt.savefig("./uniform_eq_results/SHAP/" + course + "/Permutation/plots/global/"+"barplot.png", bbox_inches = 'tight')
plt.show()


# In[ ]:


fig = shap.plots.heatmap(shap_values,max_display=15,show=False)
plt.title('global explanation')
plt.savefig("./uniform_eq_results/SHAP/"+course+"/Permutation/plots/global/heatmap.png", bbox_inches = 'tight')
plt.show()


# In[ ]:


shap.initjs()
force = shap.force_plot(shap_values.base_values[0], shap_values.values, features.iloc[num_instances],contribution_threshold=0.05)
shap.save_html("./uniform_eq_results/SHAP/" + course + "/Permutation/plots/global/"+"forceplot_for_all.html", force)


# In[ ]:


# saving SHAP values in dataframe
df_shap = pd.DataFrame(shap_values.values, columns = features.columns)
df_shap.insert(0, 'exp_num', num_instances)
df_shap.to_csv('uniform_eq_results/SHAP/Permutation/' + course + '.csv')


# ## Kernel shap:

# Uses the Kernel SHAP method to explain the output of any function.
# 
# Kernel SHAP is a method that uses a special weighted linear regression to compute the importance of each feature. The computed importance values are Shapley values from game theory and also coefficents from a local linear regression.

# You can find this algorithm's details on https://christophm.github.io/interpretable-ml-book/shap.html#kernelshap 

# In[ ]:


explainer = shap.KernelExplainer(predict_fn, Background_distribution)
instances = features.iloc[num_instances]
shap_values = explainer.shap_values(instances)


df_shap = pd.DataFrame(shap_values, columns = features.columns)
df_shap.insert(0, 'exp_num', num_instances)
df_shap.to_csv('uniform_eq_results/SHAP/Kernel/' + course + '.csv')

# In[ ]:


fig = shap.summary_plot(shap_values, instances,show=False)
locs, labels = plt.yticks()  # Get the current locations and labels.
plt.title('global explanation of Uniform instances')
plt.savefig("./uniform_eq_results/SHAP/"+course+"/Kernel/plots/global/"+"summaryplot.png", bbox_inches = 'tight')
plt.show()
labels = labels[::-1]
top_features=[]
for i in np.arange(len(labels)):
  x=labels[i]
  top_features.append(x.get_text())
print(top_features[:10])
file = "./uniform_eq_results/SHAP/"+course+"/Kernel/plots/top_features_kernel_uniform.json"
with open(file, 'w') as f: 
    json.dump(top_features, f)


# In[ ]:


# saving SHAP values in dataframe
# df_kernel_shap = pd.DataFrame(shap_values.values, columns = features.columns)
# df_kernel_shap.insert(0, 'exp_num', num_instances)
# df_kernel_shap.to_csv('uniform_eq_results/SHAP/Kernel/' + course + '.csv')

