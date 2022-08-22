import numpy as np
import pandas as pd
from scipy.stats import rv_discrete 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import sys


def train_snv_classifier(features_table_filepath):
    # Get features 
    # Assume that the first N columns (6 or so?) will be ID information. The final column will be the truth column.
    df_train = pd.read_csv("df_dnm_features_dict_priv.tsv", sep = "\t").iloc[:, 12:]
    df_train["ClippingRankSum"] = 0
    
    clf_snv = RandomForestClassifier(
            n_estimators = 125,
            max_features = "auto",
            max_depth = 450,
            min_samples_split = 2,
            min_samples_leaf = 1,
            min_weight_fraction_leaf = 0.0,
            max_leaf_nodes = None,
            n_jobs = -1,
            class_weight = "balanced",
            random_state = 42,
            verbose = 0
            )
    clf_snv.fit(df_train.values[:, :-1], df_train.values[:, -1])

def train_indel_classifier(features_table_filepath):
    # Get features 
    # Assume that the first N columns (6 or so?) will be ID information. The final column will be the truth column.
    df_train = pd.read_csv("df_dnm_features_dict_priv.tsv", sep = "\t").iloc[:, 12:]
    df_train["ClippingRankSum"] = 0

    clf_indel = RandomForestClassifier(
            n_estimators = 125,
            max_features = "auto",
            max_depth = 450,
            min_samples_split = 2,
            min_samples_leaf = 1,
            min_weight_fraction_leaf = 0.0,
            max_leaf_nodes = None,
            n_jobs = -1,
            class_weight = "balanced",
            random_state = 42,
            verbose = 0
            )
    clf_indel.fit(df_train.values[:, :-1], df_train.values[:, -1])

"""
df_train = pd.read_csv("df_dnm_features_table.tsv", sep = "\t")

# Will add feature list...
features = ["VQSLOD", "QD"]
df_features = df_train[features]
df_train["truth"] = 1

X = df_features.values
y = df_train["truth"].values
y[0:50] = 0

# We should vary the hyperparameters in a grid search
rf_clf_snv = RandomForestClassifier(max_features = "auto")
#     n_estimators = 125,
     # max_features = df_train.shape[1] - 1,
#     max_depth = 450,
#     min_samples_split = 2,
#     min_samples_leaf = 1,
#     min_weight_fraction_leaf = 0.0,
#     max_leaf_nodes = None,
#     n_jobs = -1,
#     class_weight = "balanced",
#     random_state = 42,
#     verbose = 0)
# clf_snv.fit(X, y)
distributions = dict(n_estimators = rv_discrete(values = ([100, 125], [0.5, 0.5]) ),
                     min_weight_fraction_leaf = rv_discrete(values = ([0, 0.01, 0.05], [0.5, 0.2, 0.3]))
                     )
clf_snv = RandomizedSearchCV(rf_clf_snv, distributions, random_state = 42)
search = clf_snv.fit(X, y)
print(search.best_params_)

# dump(clf_snv, "clf_snv.joblib")
"""
