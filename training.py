import numpy as np
import pandas as pd
from scipy.stats import rv_discrete 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import sys

def train_random_forest_classifier(df_input,
        n_estimators = 125,
        criterion = "gini",
        max_depth = 450,
        min_samples_split = 2,
        min_samples_leaf = 1,
        min_weight_fraction_leaf = 0.0,
        max_features = "sqrt",
        max_leaf_nodes = None,
        class_weight = "balanced",
        random_state = 42,
        n_jobs = -1,
        verbose = 1):

    key = ["chrom", "pos", "ref", "alt", "iid", "sex", "phenotype"]
    df = df_input.drop(key, axis = 1)
    features_list = list(df.columns)

    clf = RandomForestClassifier(
            n_estimators = n_estimators,
            criterion = criterion,
            max_depth = max_depth,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf,
            min_weight_fraction_leaf = min_weight_fraction_leaf,
            max_features = max_features,
            max_leaf_nodes = max_leaf_nodes,
            class_weight = class_weight,
            random_state = random_state,
            n_jobs = n_jobs,
            verbose = verbose
            )
    clf.fit(df.values[:, 0:-1], df.values[:, -1])
    clf.feature_names = features_list
    return clf

def randomized_grid_search(df_input):
    distributions = dict(n_estimators = rv_discrete(values = ([100, 125], [0.5, 0.5]) ),
                         min_weight_fraction_leaf = rv_discrete(values = ([0, 0.01, 0.05], [0.5, 0.2, 0.3]))
                        )

    params = {"n_estimators": [50, 75, 100, 125, 150, 175, 200],
              "max_depth": [200, 250, 300, 350, 400, 450, 500]
             }

    key = ["chrom", "pos", "ref", "alt", "iid"]
    df = df_input.drop(key, axis = 1)
    features_list = list(df.columns)

    clf = RandomForestClassifier(random_state = 42,
                                 n_jobs = -1,
                                 verbose = 1)
    
    # random_search = RandomizedSearchCV(clf, distributions, random_state = 42)
    random_search = RandomizedSearchCV(clf, params, random_state = 42)
    random_search.fit(df.values[:, 0:-1], df.values[:, -1])
    return random_search 

# clf_snv.fit(X, y)
#distributions = dict(n_estimators = rv_discrete(values = ([100, 125], [0.5, 0.5]) ),
#                     min_weight_fraction_leaf = rv_discrete(values = ([0, 0.01, 0.05], [0.5, 0.2, 0.3]))
#                     )
#clf_snv = RandomizedSearchCV(rf_clf_snv, distributions, random_state = 42)
#search = clf_snv.fit(X, y)
#print(search.best_params_)

# dump(clf_snv, "clf_snv.joblib")
