import pandas as pd
import joblib
# Don't make this a separate function? (key and features_list should be same across dfs..)
def classify(df_input, clf):
    preds = clf.predict_proba(df.values)
    return preds
