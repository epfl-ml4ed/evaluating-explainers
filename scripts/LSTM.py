#!/usr/bin/env python
# coding: utf-8

# # Training biLSTM on ALL features for several courses
#

# ## Importing the needed libraries:

from locale import normalize
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM, Masking
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
import seaborn as sns
import time
import json
from data_helper import load_features, load_labels

# set week type, feature types, and courses here
week_type = "eq_week"
feature_types = ["feature_set"]
courses = ["toy_course"]
# cem?
cem = False
if cem:
    tf.get_logger().setLevel(40)  # suppress deprecation messages
    tf.compat.v1.disable_v2_behavior()  # disable TF2 behaviour as alibi code still relies on TF1 constructs
# boolean: if True, remove features directly related to student success in weekly quizzes:
# student shape, competency alignment, competency strength
remove_obvious = True
# normalize
normalize = False
# set number of epochs to train models for each course:
params = {}
for course in courses:
    params[course] = {"num_epochs": 15}

# ## Training on the dataset DSP001 with replacing NaNs:

# Bidirection LSTM definition


def bidirectional_lstm(
    x_train,
    y_train,
    x_test,
    y_test,
    x_val,
    y_val,
    labels,
    week_type,
    feature_types,
    course,
    n_weeks,
    n_features,
    num_epochs=100,
    cem=False,
):
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
    # Add a sigmoid Dense layer with cem.
    sig_size = 2 if cem else 1
    lstm.add(Dense(sig_size, activation="sigmoid"))
    # compile the model
    lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
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
    y_pred = [1 if y[0] >= 0.5 else 0 for y in y_pred]
    if cem:
        y_pred = np.concatenate(
            ((1 - y_pred).reshape(-1, 1), y_pred.reshape(-1, 1)), axis=1
        )
    # evaluate the model
    model_params = {
        "model": "LSTM-bi",
        "epochs": num_epochs,
        "batch_size": 32,
        "loss": "binary_cross_entropy",
    }
    if cem:
        labels = labels[:, 1]
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
    if cem:
        lstm.save("lstm_bi_" + course + "_cem")
    else:
        lstm.save("lstm_bi_" + current_timestamp)
    return history, scores


def plot_history(history, filename):
    fig, axs = pyplot.subplots(1, 1, figsize=(6, 3))
    sns.lineplot(
        x=range(len(history.history["loss"])),
        y=history.history["loss"],
        label="train",
        ax=axs,
    )
    sns.lineplot(
        x=range(len(history.history["loss"])),
        y=history.history["val_loss"],
        label="test",
        ax=axs,
    )
    axs.set_title("Loss ")
    axs.set_xlabel("epoch")
    axs.set_ylabel("loss")
    pyplot.savefig(filename + "_loss.png")

    fig, axs = pyplot.subplots(1, 1, figsize=(6, 3))
    sns.lineplot(
        x=range(len(history.history["loss"])),
        y=history.history["accuracy"],
        label="train",
        ax=axs,
    )
    sns.lineplot(
        x=range(len(history.history["loss"])),
        y=history.history["val_accuracy"],
        label="test",
        ax=axs,
    )
    axs.set_title("Accuracy ")
    axs.set_xlabel("epoch")
    axs.set_ylabel("accuracy")


def evaluate(
    model,
    labels,
    x_test,
    y_test,
    week_type,
    feature_type,
    course,
    model_name=None,
    model_params=None,
    y_pred=None,
):
    scores = {}
    scores["test_acc"] = accuracy_score(y_test, y_pred)
    scores["test_bac"] = balanced_accuracy_score(y_test, y_pred)
    scores["test_prec"] = precision_score(y_test, y_pred)
    scores["test_rec"] = recall_score(y_test, y_pred)
    scores["test_f1"] = f1_score(y_test, y_pred)
    scores["test_auc"] = roc_auc_score(y_test, y_pred)
    scores["feature_type"] = feature_type
    scores["week_type"] = week_type
    scores["course"] = course
    scores["data_balance"] = sum(labels) / len(labels)
    return scores


# loading the data and normalizing it:


labels = {}
features = {}
selected_features = {}

for course in courses:

    filepath = (
        "../data/"
        + week_type
        + "-"
        + feature_types[0]
        + "-"
        + course
        + "/feature_labels.csv"
    )
    labels[course] = load_labels(filepath, cem=cem, normalize=normalize)
    feats, sel_feats, num_weeks, num_features = load_features(course)
    features[course] = feats
    selected_features[course] = sel_feats
    file = "selected_features/" + course + ".json"
    with open(file, "w") as f:
        json.dump(selected_features[course], f)
    params[course]["num_weeks"] = num_weeks
    params[course]["num_features"] = num_features

# training models

for course in courses:

    fts = features[course].copy()
    target = labels[course]
    fts = fts.reshape(fts.shape[0], -1)
    train_size = 0.8
    x_train, x_rem, y_train, y_rem = train_test_split(
        fts, target, train_size=train_size, random_state=25
    )
    x_test, x_val, y_test, y_val = train_test_split(
        x_rem, y_rem, train_size=0.5, random_state=25
    )
    print(course + ":")
    print(x_train.shape, x_test.shape, x_val.shape)
    print(y_train.shape, y_test.shape, y_val.shape)

    num_weeks = params[course]["num_weeks"]
    num_features = params[course]["num_features"]
    num_epochs = params[course]["num_epochs"]

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
        labels,
        week_type,
        feature_types,
        course,
        n_weeks=num_weeks,
        n_features=num_features,
        num_epochs=num_epochs,
        cem=cem,
    )
    print("{:<15} {:<8} ".format("metric", "value"))
    for ke, v in scores.items():
        if isinstance(v, float):
            v = round(v, 4)
        if ke != "feature_type":
            print("{:<15} {:<8} ".format(ke, v))
    run_name = model.__name__ + "_" + course + "_" + current_timestamp
    fig = plot_history(history=history, filename=run_name)
