#!/usr/bin/env python
# coding: utf-8

# # Notebook for explaing biLSTM trained on DSP001 using tabular LIME

from lime import lime_tabular
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import matplotlib.pyplot as pyplot
import seaborn as sns
import time
from IPython.display import clear_output


# ## Loading the features and their names

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

week_type = 'eq_week'
feature_types = ["boroujeni_et_al", "chen_cui", "marras_et_al", "lalle_conati"]
course = 'dsp_001'
remove_obvious=True

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
# print(feature_names)
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
# EDIT HERE FOR OTHER MODELS
model_name = "models/lstm_bi_"+course+"_new"
loaded_model = keras.models.load_model(model_name)
# This lambda returns a (NUM OF INSTANCES,2) array of prob of pass in first column and prob of fail in another column
predict_fn = lambda x: np.array([[1-loaded_model.predict(transform_x(x,num_feature_type,num_weeks,features_min,features_max))],[loaded_model.predict(transform_x(x,num_feature_type,num_weeks,features_min,features_max))]]).reshape(2,-1).T# ## Creating a LIME tabular explainer for the Loaded features 
# This module makes tabular lime explainer with instance numbers given to it.
def instance_explainer(instance_numbers,features,feature_names,class_names,predict_fn, num_features=10,mode='classification',discretize_continuous=True, num_samples=5000, distance_metric='euclidean', model_regressor=None, sampling_method='gaussian'):
    explainers=[]
    features=np.array(features)
    feature_names=np.array(feature_names).reshape(-1)
    explainer = lime_tabular.LimeTabularExplainer(
      training_data=features,
      feature_names=feature_names,
      class_names=class_names,
      mode=mode,
      discretize_continuous=discretize_continuous
    )
    for i in instance_numbers:
        exp = explainer.explain_instance(features[i], predict_fn, num_features=num_features)
        explainers.append(exp)
    return explainers

def show_all_in_notebook(explainers,instances,real_labels,features,num_weeks,num_feature_type,group):
    for i,exp in enumerate(explainers):
        print("For instance number {i:d} the explanation is as follows:".format(i=instances[i]))
        s = 'fail' if real_labels[instances[i]] else 'pass'
        print("The real label for this instance is {}".format(s))
        p = predict_fn(features[instances[i],:])
        s = 'fail' if p[0,1]>0.5 else 'pass'
        print('The model predicted label for this instance is {} ({:.2f})'.format(s,max(p[0,:])))
        exp.show_in_notebook()
        h = exp.as_html()
        if (group == 'r'):
          Html_file= open("./uniform_eq_results/LIME/RandomPick/notebookExps/"+str(instances[i])+".html","w")
          Html_file.write(h)
          Html_file.close()
        elif (group == 's'):
          Html_file= open("./uniform_eq_results/LIME/SubmodularPick/notebookExps/"+str(instances[i])+".html","w")
          Html_file.write(h)
          Html_file.close()
        else:
          Html_file= open("./uniform_eq_results/LIME/notebookExps/"+str(instances[i])+".html","w")
          Html_file.write(h)
          Html_file.close()
    return

def pyplot_all(explainers,instances,real_labels,group):
    import matplotlib.pyplot as plt
    for i,exp in enumerate(explainers):
        s='fail' if real_labels[instances[i]] else 'pass'
        label=exp.available_labels()[0]
        expl = exp.as_list(label=label)
        fig = plt.figure(facecolor='white')
        vals = [x[1] for x in expl]
        names = [x[0] for x in expl]
        vals.reverse()
        names.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(expl)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        if exp.mode == "classification":
            title = 'Local explanation for class %s for instance %d with real class %s' % (exp.class_names[label],instances[i],s)
        else:
            title = 'Local explanation'
        plt.title(title)
        if (group == 'r'):
          plt.savefig("./uniform_eq_results/LIME/RandomPick/pyplots/"+str(instances[i])+".png", bbox_inches = 'tight', facecolor=fig.get_facecolor())
        elif (group == 's'):
          plt.savefig("./uniform_eq_results/LIME/SubmodularPick/pyplots/"+str(instances[i])+".png", bbox_inches = 'tight', facecolor=fig.get_facecolor())
        else:
          plt.savefig("./uniform_eq_results/LIME/" +course + "/pyplots/"+str(instances[i])+".png", bbox_inches = 'tight', facecolor=fig.get_facecolor())
    return

def DataFrame_all(explainers,instances,real_labels,group):
    df=pd.DataFrame({})
    class_names=['pass', 'fail']
    dfl=[]
    for i,exp in enumerate(explainers):
        this_label=exp.available_labels()[0]
        l=[]
        l.append(("exp number",instances[i]))
        l.append(("real value",'fail' if real_labels[instances[i]] else 'pass'))
        l.extend(exp.as_list(label=this_label))
        dfl.append(dict(l))
    df=df.append(pd.DataFrame(dfl))
    dfl=[pd.DataFrame.from_dict(x, orient='index') for x in dfl]
    for i,x in enumerate(dfl):
      if (group == 'r'):
        x.to_csv(r"./uniform_eq_results/LIME/RandomPick/dataframes/"+str(instances[i])+'.csv')
      elif (group == 's'):
        x.to_csv(r"./uniform_eq_results/LIME/SubmodularPick/dataframes/"+str(instances[i])+'.csv')
      else:
        x.to_csv(r"./uniform_eq_results/LIME/" + course + "/dataframes/"+str(instances[i])+'.csv')
    # for x in dfl:
    #     display(x.head(len(instances)))
    return df,dfl


group = 'UniformPick'

prediction = loaded_model.predict(features)
instances = np.load('uniform_'+course+'.npy')

# In[ ]:
print('training explainer')


import time
start = time.time()
class_names=['pass', 'fail']
explainers=instance_explainer(instances,features,feature_names,class_names,predict_fn)
end = time.time()
print(end - start)

# clear_output()
print('pyplot')
pyplot_all(explainers,instances,y,group)
# pyplot.close('all') #comment this to see results
print('df saving')
df,dfl=DataFrame_all(explainers,instances,y,group)
# clear_output() #comment this to see results

df.iloc[:,2::] = abs(df.iloc[:,2::])
ai = np.argsort(df.iloc[:,2::].values)
for j,c in enumerate(list(ai[:,:10])):
    df.iloc[j,c+2] = np.arange(1,11)
top_features = df.columns
top_features = top_features[2::]
top_features_type = np.array([s[0:s.find('InWeek')].split(' ')[-1] for s in top_features])
top_features_week = np.array([s[s.find('InWeek')+6::].split(' ')[0] for s in top_features])
count = (~(df.iloc[:,2::].isnull())).values.sum(axis=0)
top_features_type_unique = list({ k for k in top_features_type })
count_feature_type = [count[top_features_type==x].sum()/500 for x in top_features_type_unique]
top_features_week_unique = list({ k for k in top_features_week })
count_feature_week = [count[top_features_week==x].sum()/500 for x in top_features_week_unique]
# clear_output()

ind = np.argsort(count_feature_type)[::-1]
fig, ax = pyplot.subplots(figsize=(10, 5),facecolor='white')
g = sns.barplot(x=top_features_type_unique, y=count_feature_type, palette="rocket",ax =ax,order=np.take_along_axis(np.array(top_features_type_unique), ind, axis=0))
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title( 'important features with respect to their type', fontsize=15)
g.set_xlabel("", fontsize=15)
g.set_ylabel("percent of features", fontsize=12)
pyplot.savefig("./uniform_eq_results/LIME/" + course + "/pyplots/impFeaturesType.png", bbox_inches = 'tight', facecolor=fig.get_facecolor())


ind = np.argsort(count_feature_week)[::-1]
fig, ax = pyplot.subplots(figsize=(10, 5),facecolor='white')
g = sns.barplot(x=top_features_week_unique, y=count_feature_week, palette="rocket",ax =ax,order=np.take_along_axis(np.array(top_features_week_unique), ind, axis=0))
g.set_xticklabels(g.get_xticklabels(),rotation=0)
g.set_title( 'important features with respect to their week', fontsize=15)
g.set_xlabel("", fontsize=15)
g.set_ylabel("percent of features", fontsize=12)
pyplot.savefig("./uniform_eq_results/LIME/" + course + "/pyplots/impFeaturesWeek.png", bbox_inches = 'tight', facecolor=fig.get_facecolor())
zero_data = np.zeros(shape=(num_weeks,len(top_features_type_unique)))
d = pd.DataFrame(zero_data, columns=top_features_type_unique)
for i,f in enumerate(top_features_type):
    d[f][int(top_features_week[i])-1]+=count[i]
fig, ax = pyplot.subplots(figsize=(10, 10),facecolor='white')
g = sns.heatmap(d.values.T / instances.shape[0], annot=True, fmt=".2f",ax=ax)
l=list(np.arange(1,num_weeks+1))
g.set_xticklabels(['week'+str(i) for i in l],rotation=0)
g.set_yticklabels(top_features_type_unique,rotation=0)
pyplot.savefig("./uniform_eq_results/LIME/" + course + "/pyplots/heatmap.png", bbox_inches = 'tight', facecolor=fig.get_facecolor())

df.to_csv(r'./uniform_eq_results/LIME/'+course+'/dataframes/all_important_features.csv', index = False, header = True)
d.to_csv(r'./uniform_eq_results/LIME/'+course+'/dataframes/df_for_heatmap.csv', index = False, header = True)

