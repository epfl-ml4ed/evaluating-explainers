import numpy as np
import pandas as pd
import tensorflow.keras as keras
from data_helper import load_features, load_labels, load_feature_names, add_week

week_type = "eq_week"
feature_types = ["feature_set"]
course = "toy_course"
num_f = 50
num_p = 50
remove_obvious = True
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
labels = load_labels(filepath, normalize=normalize)
# Adding week number to feature_names
feature_names = add_week(selected_features, feature_types, num_weeks, week_type)
features = pd.DataFrame(features, columns=feature_names)
# EDIT HERE FOR OTHER MODELS
model_name = "models/lstm_bi_" + course + "_new"
loaded_model = keras.models.load_model(model_name)

prediction = loaded_model.predict(np.array(features))
###################
features_with_prediction = features.copy()
features_with_prediction["prediction"] = prediction
features_with_prediction["real_label"] = labels
features_with_prediction["abs_difference"] = abs(
    features_with_prediction["prediction"].values
    - features_with_prediction["real_label"].values
)
###################
failed_instances = labels > 0
failed = features_with_prediction.iloc[failed_instances]
failed = failed.sort_values(by="abs_difference")
###################
passed_instances = labels < 1
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
