#!/usr/bin/env python
# coding: utf-8


# Importing the libraries needed
from locale import normalize
import numpy as np
import pandas as pd
import tensorflow as tf
from alibi.explainers import CEM
from data_helper import add_week, load_features, load_feature_names, load_labels

tf.get_logger().setLevel(40)  # suppress deprecation messages
tf.compat.v1.disable_v2_behavior()  # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Masking
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import matplotlib.pyplot as pyplot
import time


# Loading data

# Edit here for other courses, parameters
week_type = "eq_week"
feature_types = ["feature_set"]
course = "toy_course"
# Remove features directly related to student success from our Marras feature set.
remove_obvious = False

# Loading feature names
file_path = "../data/feature_names/"  # path to folder containing feature names.
# Cleaning feature names for better representation. Change clean_name in datahelper for adaption to your feature sets.
clean = False
feature_names = load_feature_names(
    feature_types,
    file_path=file_path,
    remove_obvious=remove_obvious,
    clean=clean,
)


# Loading the features
filepath = "../data/"
# Dropping features with all NaNs or all Zeros.
drop = False
# Filling the NaNs. Change FillNaN in data_helper for adaption.
fill = False
# minmax normalization. See "details.txt" for more info.
normalize = False
features, selected_features, num_weeks, n_features = load_features(
    filepath,
    feature_types,
    week_type,
    course,
    feature_names,
    remove_obvious=remove_obvious,
    fill=fill,
    drop=drop,
    normalize=normalize,
)
SHAPE = features.shape

# Loading the labels
filepath = (
    "../data/"
    + week_type
    + "-"
    + feature_types[0]
    + "-"
    + course
    + "/feature_labels.csv"
)
labels = load_labels(filepath, cem=True, normalize=normalize)
# Adding week number to feature_names
feature_names = add_week(selected_features, feature_types, num_weeks, week_type)

f = pd.DataFrame(features, columns=feature_names)
s_f = list(selected_features.values())
num_features = len([feature for feature_group in s_f for feature in feature_group])


# # Model

# A new model has to be trained for CEM, since it needs a target variable of a different shape (n_instances, 2)
def bidirectional_lstm(
    x_train,
    y_train,
    x_test,
    y_test,
    x_val,
    y_val,
    week_type,
    feature_types,
    course,
    n_weeks,
    n_features,
    num_epochs=100,
):
    n_dims = x_train.shape[0]
    look_back = 3
    # LSTM
    # define model
    lstm = Sequential()
    ###########Reshape layer################
    lstm.add(
        tf.keras.layers.Reshape(
            (n_weeks, n_features), input_shape=(n_weeks * n_features,)
        )
    )
    ##########deleting the 1.001 max row added###########
    lstm.add(Masking(mask_value=1))
    lstm.add(Bidirectional(LSTM(64, return_sequences=True)))
    lstm.add(Bidirectional(LSTM(32)))
    # Add a sigmoid Dense layer with 1 units.
    lstm.add(Dense(2, activation="sigmoid"))
    # compile the model
    lstm.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # fit the model
    history = lstm.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=num_epochs,
        batch_size=32,
        verbose=1,
    )
    # evaluate the model
    y_pred = lstm.predict(x_test)
    print(y_pred.shape)
    y_pred = np.array([1 if y >= 0.5 else 0 for y in y_pred[:, 1]])
    print(y_pred.shape)
    y_pred = np.concatenate(
        ((1 - y_pred).reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1
    )
    print(y_pred.shape)
    # evaluate the model
    model_params = {
        "model": "LSTM-bi",
        "epochs": num_epochs,
        "batch_size": 32,
        "loss": "binary_cross_entropy",
    }
    scores = evaluate(
        None,
        labels,
        x_test,
        y_test,
        week_type,
        feature_types,
        course,
        y_pred=y_pred,
        model_name="TF-LSTM-bi",
        model_params=model_params,
    )
    lstm.save("lstm_bi_" + course + "_cem")
    return history, scores


def plot_history(history, file_name):
    # plot loss during training
    pyplot.figure(0)
    pyplot.title("Loss " + file_name)
    pyplot.plot(history.history["loss"], label="train")
    pyplot.plot(history.history["val_loss"], label="test")
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.legend()
    pyplot.savefig(file_name + "_loss.png")
    # plot accuracy during training
    pyplot.figure(1)
    pyplot.title("Accuracy " + file_name)
    pyplot.plot(history.history["acc"], label="train")
    pyplot.plot(history.history["val_acc"], label="test")
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("accuracy")
    pyplot.savefig(file_name + "_acc.png")


def evaluate(
    model,
    labels,
    x_test,
    y_test,
    week_type,
    feature_type,
    course,
    y_pred=None,
):
    scores = {}
    y_test = y_test[:, 1]
    y_pred = y_pred[:, 1]
    scores["test_acc"] = accuracy_score(y_test, y_pred)
    scores["test_bac"] = balanced_accuracy_score(y_test, y_pred)
    scores["test_prec"] = precision_score(y_test, y_pred)
    scores["test_rec"] = recall_score(y_test, y_pred)
    scores["test_f1"] = f1_score(y_test, y_pred)
    scores["test_auc"] = roc_auc_score(y_test, y_pred)
    scores["feature_type"] = feature_type
    scores["week_type"] = week_type
    scores["course"] = course
    scores["data_balance"] = sum(labels[:, 1]) / len(labels[:, 1])
    return scores


train_size = 0.8
x_train, x_rem, y_train, y_rem = train_test_split(
    features, labels, train_size=train_size, random_state=25
)
x_test, x_val, y_test, y_val = train_test_split(
    x_rem, y_rem, train_size=0.5, random_state=25
)
print(x_train.shape, x_test.shape, x_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)


current_timestamp = str(time.time())[:-2]
model = bidirectional_lstm
print(model.__name__)
history, scores = model(
    x_train,
    y_train,
    x_test,
    y_test,
    x_val,
    y_val,
    week_type,
    feature_types,
    course,
    num_epochs=15,
    n_weeks=num_weeks,
    n_features=num_features,
)
print("{:<15} {:<8} ".format("metric", "value"))
for ke, v in scores.items():
    if isinstance(v, float):
        v = round(v, 4)
    if ke != "feature_type":
        print("{:<15} {:<8} ".format(ke, v))
run_name = model.__name__ + "_" + course + "_" + current_timestamp
plot_history(history, run_name)
print(run_name)


# # Explainers

bilstm = load_model("./lstm_bi_" + course + "_cem")


def CEM_all(num_instances, features, mode="PN"):
    shape = (1,) + features.shape[1:]  # instance shape
    kappa = 0.0  # minimum difference needed between the prediction probability for the perturbed instance on the
    # class predicted by the original instance and the max probability on the other classes
    # in order for the first loss term to be minimized
    beta = 0.1  # weight of the L1 loss term
    gamma = 100  # weight of the optional auto-encoder loss term
    c_init = 1.0  # initial weight c of the loss term encouraging to predict a different class (PN) or
    # the same class (PP) for the perturbed instance compared to the original instance to be explained
    c_steps = 10  # nb of updates for c
    max_iterations = 1000  # nb of iterations per value of c
    feature_range = (
        features.min(axis=0),
        features.max(axis=0),
    )  # feature range for the perturbed instance
    clip = (-1000.0, 1000.0)  # gradient clipping
    lr = 1e-2  # initial learning rate
    no_info_val = (
        -1.0
    )  # a value, float or feature-wise, which can be seen as containing no info to make a prediction
    # perturbations towards this value means removing features, and away means adding features
    # for our MNIST images, the background (-0.5) is the least informative,
    # so positive/negative perturbations imply adding/removing features
    cem = CEM(
        bilstm,
        mode,
        shape,
        kappa=kappa,
        beta=beta,
        feature_range=feature_range,
        gamma=gamma,
        ae_model=None,
        max_iterations=max_iterations,
        c_init=c_init,
        c_steps=c_steps,
        learning_rate_init=lr,
        clip=clip,
        no_info_val=no_info_val,
    )
    changes = []
    explanations = []
    for i in num_instances:
        try:
            X = features[i].reshape((1,) + features[0].shape)
            explanation = cem.explain(X)
            change = explanation.PN - X
            print(f"counterfactuals generated for instance {i}")
            changes.append(change)
            explanations.append(explanation)
        except TypeError:
            print(f"Error occured for instance {i}")
            print(change)
    return explanations, changes


num_instances = np.load("uniform_" + course + ".npy")

t1 = time.time()
explanation, changes = CEM_all(num_instances, features, mode="PN")
t2 = time.time()
print(f"time taken: {(t2-t1)/60.0} minutes")

instances = features[num_instances]
np.save(
    "uniform_eq_results/CEM/" + course + "/changes_pn",
    np.array(changes).reshape(len(num_instances), -1),
)

pns = np.array([explanation[i].PN for i in range(len(explanation))]).reshape(
    len(num_instances), -1
)
np.save("uniform_eq_results/CEM/" + course + "/pns", pns)

np.save("uniform_eq_results/CEM/" + course + "/instances", instances)

sds = pd.DataFrame(features, columns=feature_names).describe().loc["std", :]


# Getting importance scores

diffs = pd.DataFrame(
    np.array(changes).reshape(len(num_instances), -1), columns=feature_names
)
for col in diffs.columns:
    diffs[col] = diffs[col].apply(lambda x: np.abs(x * sds[col]))

diffs.insert(0, "exp_num", num_instances)
diffs.to_csv("uniform_eq_results/CEM/" + course + "/importances.csv")
