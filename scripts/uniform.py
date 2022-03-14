import numpy as np
import pandas as pd
import tensorflow.keras as keras
import json

week_type = "eq_week"
feature_types = ["boroujeni_et_al", "chen_cui", "marras_et_al", "lalle_conati"]
course = "microcontroleurs_003"
num_f = 50
num_p = 50
remove_obvious = True

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


#
# Loading the features used in training in the biLSTM with loading the json file made in the biLSTM training notebook.
file = './selected_features/' + course + '.json'
with open(file, 'r') as f:
    selected_features = json.load(f)
#
# Loading the labels
feature_type = "boroujeni_et_al"
filepath = './data/all/' + week_type + '-' + feature_type + '-' + course + '/feature_labels.csv'
labels = pd.read_csv(filepath)["label-pass-fail"]
labels[labels.shape[0]] = 1
y = labels.values

# Loading feature names
feature_names = []
for feature_type in feature_types:
    filepath = "./scripts/feature_names/" + feature_type + ".csv"
    if feature_type == "lalle_connati":
        filepath = "./scripts/feature_names/lalle_conati.csv"
    feature_type_name = pd.read_csv(filepath, header=None)
    feature_type_name = feature_type_name.values.reshape(-1)
    feature_names.append(feature_type_name)
    
if remove_obvious:
  new_marras = feature_names[2][[2,3,4,5]]
  feature_names[2] = new_marras
    
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

# In[ ]:


# calculate the number of features
n_features = sum([len(x) for x in selected_features])


# In[ ]:


# make feature names more readable
# ex: time_in__problem_<function sum at 0x7f3bd02cc9d0> -> time_in_problem_sum
def clean_name(feature):
  id = feature.find('<')
  if id==-1:
    return feature
  fct = feature[id+9:id+14].strip()
  return feature[0:id]+fct


# In[ ]:


selected_features = [np.array([clean_name(x) for x in selected_features[0]]),
 np.array([clean_name(x) for x in selected_features[1]]),
 np.array([clean_name(x) for x in selected_features[2]]),
 np.array([clean_name(x) for x in selected_features[3]])]


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


num_feature_type = []
for feature_type in feature_types:
    num_feature_type.append(len(selected_features[feature_type]))
print(num_feature_type)


# In[ ]:


# Loading feature names and transforming them to 2D format.
feature_names= []
final_features = []
for feature_type in feature_types:
    [final_features.append(x) for x in selected_features[feature_type]]
for i in np.arange(n_weeks):
    feature_type_name_with_weeks = [(x+'_InWeek'+str(i+1)) for x in final_features]
    feature_names.append(feature_type_name_with_weeks)
feature_names = np.concatenate(feature_names, axis=0)
feature_names = feature_names.reshape(-1)
# print(feature_names)
features.columns = feature_names


# ## Making a predict_proba

# In[ ]:


features_min = features.min(axis=0)
features_max = features.max(axis=0)
features_max = np.where(features_max==0, np.ones(features_max.shape),features_max)


# In[ ]:


# This module transforms our data to the 2D format biLSTM was trained with.
def transform_x(x, num_feature_type, num_weeks, features_min, features_max, normal=True):
    return np.array(x).reshape((x.shape[0],x.shape[1]))


# In[ ]:


features_min = features.min(axis=0)
features_max = features.max(axis=0)


# In[ ]:


# EDIT HERE FOR OTHER MODELS
model_name = "lstm_bi_"+course+"_new"
loaded_model = keras.models.load_model(model_name)


############################################################################################################
prediction = loaded_model.predict(transform_x(np.array(features), num_feature_type, n_weeks, features_min=features_min, features_max=features_max))
###################
print(prediction.shape, y.shape, features.shape)
features_with_prediction = features.copy()
features_with_prediction["prediction"] = prediction
features_with_prediction["real_label"] = y
features_with_prediction["abs_difference"] = abs(
    features_with_prediction["prediction"].values
    - features_with_prediction["real_label"].values
)
###################
failed_instances = y > 0
failed = features_with_prediction.iloc[failed_instances]
failed = failed.sort_values(by="abs_difference")
###################
passed_instances = y < 1
passed = features_with_prediction.iloc[passed_instances]
passed = passed.sort_values(by="abs_difference")
#################
chosen_p = passed.iloc[
    (np.ceil(np.linspace(0, passed.shape[0] - 1, num_p)))
].index.values
chosen_f = failed.iloc[
    (np.ceil(np.linspace(0, failed.shape[0] - 1, num_f)))
].index.values
instances = np.concatenate((chosen_f, chosen_p))
#################
np.save("uniform_" + course, instances)
