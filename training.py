import numpy as np
import pandas as pd
from scipy.stats import rv_discrete 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import sys

def train_random_forest_classifier(df_input,
        n_estimators = 125,
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
    params = {"n_estimators": [50, 75, 100, 125, 150, 175, 200],
              "max_depth": [None, 100, 200, 300, 400, 500],
              "min_samples_split": [.01, .05, .1, 2],
              "min_samples_leaf": [.01, .05, .1, 1],
             }

    key = ["chrom", "pos", "ref", "alt", "iid"]
    df = df_input.drop(key, axis = 1)
    features_list = list(df.columns)

    clf = RandomForestClassifier(n_jobs = -1,
                                 random_state = 42,
                                 verbose = 0,
                                 warm_start = True)
    
    #random_search = RandomizedSearchCV(clf, params, random_state = 42, n_iter = 250)
    random_search = RandomizedSearchCV(clf, params, random_state = 42, n_iter = 10)
    random_search.fit(df.values[:, 0:-1], df.values[:, -1])
    return random_search 
