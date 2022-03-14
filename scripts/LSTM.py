#!/usr/bin/env python
# coding: utf-8

# # Training biLSTM on ALL features for DSP001
# 

# In[18]:


from google.colab import drive
drive.mount('/content/drive')


# In[19]:


# modify filepath here
get_ipython().run_line_magic('cd', '/content/drive/MyDrive/semproj/epfl/ex-epfl-mooc/ex-epfl-mooc/')


# ## Importing the needed libraries:

# In[20]:


#importing the libraries needed
import numpy as np
import pandas as pd
import tensorflow as tf
from math import floor, ceil
import sklearn as sk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Bidirectional, LSTM,Masking,Embedding
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, accuracy_score, make_scorer
from sklearn.model_selection import cross_validate,train_test_split,GridSearchCV
from sklearn.preprocessing import normalize
from tensorflow.keras.models import load_model 
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as pyplot
import seaborn as sns
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import time
import json


# ## Exploring the data:

# In[21]:


# set week type, feature types, and courses here
week_type = 'eq_week'
feature_types = [ "boroujeni_et_al", "chen_cui", "marras_et_al", "lalle_conati"]
#courses = ['dsp_001', 'dsp_002', 'villesafricaines_001',
#           'progfun_002', 'geomatique_003']
courses = ['progfun_002', 'geomatique_003']

# boolean: if True, remove features directly related to student success in weekly quizzes:
# student shape, competency alignment, competency strength
remove_obvious = True

# set number of epochs to train models for each course:
params = {}
for course in courses:
  params[course] = {'num_epochs':15}


# In[22]:


# Loading the features
feature_list = {}
for course in courses:
    feature_type_list = []
    for feature_type in feature_types:
        filepath = 'data/all/' + week_type + '-' + feature_type + '-' + course
        feature_current = np.load(filepath+'/feature_values.npz')['feature_values']
        print(feature_current.shape)
        feature_norm = feature_current.reshape(-1,feature_current.shape[2] )
        print(feature_norm.shape)
        feature_type_list.append(pd.DataFrame(feature_norm))
    feature_list[course] = feature_type_list

print('course: ', courses)
print('week_type: ', week_type)
print('feature_type: ', feature_types)


# In[23]:


# Loading feature names
feature_names= []
for feature_type in feature_types:
    filepath = './scripts/feature_names/eq_week/' + feature_type + '.csv'
    feature_type_name = pd.read_csv(filepath,header=None)
    feature_type_name = feature_type_name.values.reshape(-1)
    feature_names.append(feature_type_name)
    print(feature_type_name.shape)

if remove_obvious:
  new_marras = feature_names[2][[2,3,4,5]]
  feature_names[2] = new_marras

  for course in courses:
    new_features = feature_list[course][2].drop([0,1,6], axis=1)
    feature_list[course][2] = new_features


# In[24]:


# reformat feature names
# ex: time_sessions_<function sum at 0x7f3bd02cc9d0> -> time_sessions_sum
def clean_name(feature):
  id = feature.find('<')
  if id==-1:
    return feature
  fct = feature[id+9:id+14].strip()
  return feature[0:id]+fct


# In[25]:


feature_names = [np.array([clean_name(x) for x in feature_names[0]]),
 np.array([clean_name(x) for x in feature_names[1]]),
 np.array([clean_name(x) for x in feature_names[2]]),
 np.array([clean_name(x) for x in feature_names[3]])]


# In[26]:


for i,feature_type in enumerate(feature_types):
    for course in courses:
        y=np.isnan(feature_list[course][i]).sum(axis=0)/(feature_list[course][i].shape[0])
        fig, ax = pyplot.subplots(figsize=(10, 5),facecolor='white')
        g = sns.barplot(x=feature_names[i], y=y, palette="Greens", ax =ax)
        g.set_xticklabels(g.get_xticklabels(),rotation=90)
        g.set_title( feature_type+" for "+course, fontsize=15)
        g.set_xlabel("", fontsize=15)
        g.set_ylabel("% of NaNs", fontsize=12)
        for index,data in enumerate(round(y,2)):
            pyplot.text(x=index-.2 , y =data , s=f"{data}" , fontdict=dict(fontsize=10))
        fig.savefig("NaNs_"+feature_type+"_for_"+course+".png", bbox_inches = 'tight', facecolor=fig.get_facecolor())


# In[27]:


for i,feature_type in enumerate(feature_types):
    for course in courses:
        y=(feature_list[course][i]==0).sum(axis=0)/(feature_list[course][i].shape[0])
        fig, ax = pyplot.subplots(figsize=(10, 5),facecolor='white')
        g = sns.barplot(x=feature_names[i], y=y, palette="Greens",ax =ax)
        g.set_xticklabels(g.get_xticklabels(),rotation=90)
        g.set_title( feature_type+" for "+course, fontsize=15)
        g.set_xlabel("", fontsize=15)
        g.set_ylabel("% of zeros", fontsize=12)
        for index,data in enumerate(round(y,2)):
            pyplot.text(x=index-.2 , y =data , s=f"{data}" , fontdict=dict(fontsize=10))
        fig.savefig("zeros_"+feature_type+"_for_"+course+".png", bbox_inches = 'tight', facecolor=fig.get_facecolor())


# ## Training on the dataset DSP001 with replacing NaNs:

# In[28]:


# Bidirection LSTM definition

def bidirectional_lstm(x_train, y_train, x_test, y_test, x_val, y_val, week_type, feature_types, course,n_weeks,n_features, num_epochs=100):
    n_dims = x_train.shape[0]
    look_back = 3
    # LSTM
    # define model
    lstm = Sequential()
    ###########Reshape layer################
    lstm.add(tf.keras.layers.Reshape((n_weeks, n_features), input_shape=(n_weeks*n_features,)))
    ##########deleting the 1.001 max row added###########
    lstm.add(Masking(mask_value = 1))
    lstm.add(Bidirectional(LSTM(64, return_sequences=True)))
    lstm.add(Bidirectional(LSTM(32)))
    # Add a sigmoid Dense layer with 1 units.
    lstm.add(Dense(1, activation='sigmoid'))
    # compile the model
    lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # fit the model
    history = lstm.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epochs, batch_size=32, verbose=1)
    # evaluate the model
    y_pred = lstm.predict(x_test)
    y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
    # evaluate the model
    model_params = {'model': 'LSTM-bi', 'epochs': num_epochs, 'batch_size': 32, 'loss': 'binary_cross_entropy'}
    scores = evaluate(None, x_test, y_test, week_type, feature_types, course, y_pred=y_pred, model_name="TF-LSTM-bi", model_params=model_params)
    #lstm.save('lstm_bi_'+current_timestamp)
    return history, scores


# In[29]:


def plot_history(history, filename):
  fig, axs = pyplot.subplots(1,1, figsize=(6,3))
  sns.lineplot(x=range(len(history.history['loss'])), y=history.history['loss'], label='train', ax=axs)
  sns.lineplot(x=range(len(history.history['loss'])), y=history.history['val_loss'], label='test',ax=axs)
  axs.set_title('Loss ')
  axs.set_xlabel('epoch')
  axs.set_ylabel('loss')
  pyplot.savefig(filename+'_loss.png')
  
  fig, axs = pyplot.subplots(1,1, figsize=(6,3))
  sns.lineplot(x=range(len(history.history['loss'])), y=history.history['accuracy'], label='train', ax=axs)
  sns.lineplot(x=range(len(history.history['loss'])), y=history.history['val_accuracy'], label='test',ax=axs)
  axs.set_title('Accuracy ')
  axs.set_xlabel('epoch')
  axs.set_ylabel('accuracy')
  #pyplot.savefig(filename+'_acc.png')


# In[30]:


def evaluate(model, x_test, y_test, week_type, feature_type, course, model_name=None, model_params=None, y_pred=None):
    scores={}
    scores['test_acc'] = accuracy_score(y_test, y_pred)
    scores['test_bac'] = balanced_accuracy_score(y_test, y_pred)
    scores['test_prec'] = precision_score(y_test, y_pred)
    scores['test_rec'] = recall_score(y_test, y_pred)
    scores['test_f1'] = f1_score(y_test, y_pred)
    scores['test_auc'] = roc_auc_score(y_test, y_pred)
    scores['feature_type'] = feature_type
    scores['week_type'] = week_type
    scores['course'] = course
    scores['data_balance'] = sum(y)/len(y)
    return scores


# In[31]:


# fillNaN function replaces NaNs in each week with the minimum of the feature over all weeks

def fillNaN(feature):
    shape = feature.shape
    feature_min = np.nanmin(feature.reshape(-1,shape[2]),axis=0)
    feature = feature.reshape(-1,shape[2])
    inds = np.where(np.isnan(feature))
    feature[inds] = np.take(feature_min.reshape(-1), inds[1])
    feature = feature.reshape(shape)
    return feature


# loading the data and normalizing it:

# In[32]:


def load_labels(course):
  feature_type = "boroujeni_et_al"
  filepath = 'data/all/' + week_type + '-' + feature_type + '-' + course + '/feature_labels.csv'
  labels = pd.read_csv(filepath)['label-pass-fail']
  labels[labels.shape[0]] = 1
  return labels.values

def load_features(course):
  feature_list = []
  selected_features = []
  num_weeks = 0
  num_features = 0
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
    num_features += len(feature_names[i][selected])


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
  return features, selected_features, num_weeks, num_features


# In[33]:


labels = {}
features = {}
selected_features = {}

for course in courses:
  labels[course] = load_labels(course)
  feats, sel_feats, num_weeks, num_features = load_features(course)
  features[course] = feats
  selected_features[course] = sel_feats
  params[course]['num_weeks'] = num_weeks
  params[course]['num_features'] = num_features


# In[34]:


for course in courses:
  sel_feats = {
      "boroujeni_et_al":list(selected_features[course][0]),
      "chen_cui":list(selected_features[course][1]),
      "marras_et_al":list(selected_features[course][2]),
      "lalle_conati":list(selected_features[course][3])
      }
  selected_features[course] = sel_feats
  file = 'selected_features/' + course + '.json'

  with open(file, 'w') as f: 
    json.dump(selected_features[course],f)


# In[35]:


# training models

for course in courses:

  fts = features[course].copy()
  target = labels[course]
  fts = fts.reshape(fts.shape[0], -1)
  train_size = 0.8
  x_train, x_rem, y_train, y_rem = train_test_split(fts, target, train_size=train_size, random_state=25)
  x_test, x_val, y_test, y_val = train_test_split(x_rem, y_rem, train_size=0.5, random_state=25)
  print(course+':')
  print(x_train.shape,x_test.shape,x_val.shape)
  print(y_train.shape,y_test.shape,y_val.shape)


  num_weeks = params[course]['num_weeks']
  num_features = params[course]['num_features']
  num_epochs = params[course]['num_epochs']

  current_timestamp = str(time.time())[:-2]
  model=bidirectional_lstm
  print(model.__name__)
  history, scores = model(x_train, y_train, x_test, y_test, x_val, y_val,
                          week_type, feature_types, course, n_weeks=num_weeks,
                          n_features=num_features, num_epochs=num_epochs)
  print("{:<15} {:<8} ".format('metric','value'))
  for ke, v in scores.items():
    if isinstance(v, float):
        v=round(v, 4)
    if ke!="feature_type":
        print("{:<15} {:<8} ".format(ke, v))
  run_name = model.__name__ + "_" + course + "_" + current_timestamp
  fig = plot_history(history=history, filename=run_name)


# In[35]:




