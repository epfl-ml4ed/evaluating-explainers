import numpy as np
import pandas as pd


def load_feature_names(
    feature_types,
    file_path="../data/feature_names/",
    remove_obvious=False,
    clean=False,
):
    for feature_type in feature_types:
        filepath = file_path + feature_type + ".csv"
        feature_type_name = pd.read_csv(filepath, header=None)
        feature_type_name = feature_type_name.values.reshape(-1)
        feature_names.append(feature_type_name)

    # Dropping student_shape, competency strength, competency alignment from feature names
    if remove_obvious:
        new_marras = feature_names[2][[2, 3, 4, 5]]
        feature_names[2] = new_marras

    # Cleaning feature names
    if clean:
        feature_names = [
            np.array([clean_name(x) for x in feature_names[i]])
            for i in len(feature_names)
        ]
    return feature_names


def load_labels(filepath, cem=False, normalize=False):
    labels = pd.read_csv(filepath)["label-pass-fail"]
    if normalize:
        labels[labels.shape[0]] = 1
    if cem:
        labels = np.concatenate(
            ((1 - labels.values).reshape(-1, 1), (labels.values).reshape(-1, 1)), axis=1
        )
    return labels


def load_features(
    filepath,
    feature_types,
    week_type,
    course,
    feature_names,
    remove_obvious=False,
    fill=False,
    drop=False,
    normalize=False,
):
    feature_list = []
    selected_features = []
    num_weeks = 0
    n_features = 0
    for i, feature_type in enumerate(feature_types):
        filepath = filepath + week_type + "-" + feature_type + "-" + course
        feature_current = np.load(filepath + "/feature_values.npz")["feature_values"]
        shape = feature_current.shape

        # Remove student shape, competency strength, competency alignment
        if feature_type == "marras_et_al" and remove_obvious:
            feature_current = np.delete(feature_current, [0, 1, 6], axis=2)

        shape = feature_current.shape
        if i == 0:
            num_weeks = shape[1]
        selected = np.arange(shape[2])
        if drop:
            nonNaN = (
                shape[0] * shape[1]
                - np.isnan(feature_current.reshape(-1, feature_current.shape[2])).sum(
                    axis=0
                )
                > 0
            )
            feature_current = feature_current[:, :, nonNaN]
            selected = selected[nonNaN]
        if fill:
            feature_current = fillNaN(feature_current)
        if drop:
            nonZero = (
                abs(feature_current.reshape(-1, feature_current.shape[2])).sum(axis=0)
                > 0
            )
            selected = selected[nonZero]
            feature_current = feature_current[:, :, nonZero]
        selected_features.append(feature_names[i][selected])
        n_features += len(feature_names[i][selected])
        if normalize:
            features_min = feature_current.min(axis=0).reshape(-1)
            features_max = feature_current.max(axis=0)
            features_max = np.where(
                features_max == 0, np.ones(features_max.shape), features_max
            )
            max_instance = 1.001 * features_max
            feature_current = np.vstack(
                [feature_current, max_instance.reshape((1,) + max_instance.shape)]
            )
            features_max = features_max.reshape(-1)
            feature_norm = (
                feature_current.reshape(shape[0] + 1, -1) - features_min
            ) / (1.001 * features_max - features_min)
            feature_current = feature_norm.reshape(
                -1, feature_current.shape[1], feature_current.shape[2]
            )
        feature_list.append(feature_current)

    features = np.concatenate(feature_list, axis=2)
    features = features.reshape(features.shape[0], -1)
    selected_features = {
        "boroujeni_et_al": list(selected_features[0]),
        "chen_cui": list(selected_features[1]),
        "marras_et_al": list(selected_features[2]),
        "lalle_conati": list(selected_features[3]),
    }
    return features, selected_features, num_weeks, n_features


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


def clean_name(feature):
    id = feature.find("<")
    if id == -1:
        return feature
    fct = feature[id + 9 : id + 14].strip()
    return feature[0:id] + fct


def add_week(selected_features, feature_types, num_weeks, week_type):
    inTill = "_InWeek" if week_type == "eq_week" else "_TillWeek"
    feature_names = []
    final_features = []
    for feature_type in feature_types:
        [final_features.append(x) for x in selected_features[feature_type]]
    for i in np.arange(num_weeks):
        feature_type_name_with_weeks = [
            (x + inTill + str(i + 1)) for x in final_features
        ]
        feature_names.append(feature_type_name_with_weeks)
    feature_names = np.concatenate(feature_names, axis=0)
    feature_names = feature_names.reshape(-1)
    return feature_names
