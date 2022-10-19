import pandas as pd
import joblib

def run_snv_classifier(df, clf_snv_filepath):
    df_snv = df[(df["ref"].str.len() == 1) & (df["alt"].str.len() == 1)]
    df_snv_values = df_snv[key_and_features].dropna().values
    clf_snv = joblib.load(clf_snv_filepath)
    preds_snv_values = clf_snv.predict_proba(df_snv_values[:, 5:])
    df_preds  = pd.DataFrame({"chrom": df_snv_values[:, 0], "pos": df_snv_values[:, 1], "ref": df_snv_values[:, 2], "alt": df_snv_values[:, 3], "iid": df_snv_values[:, 4], "prob": preds_snv_values[:, 0]})
    # Retain the index for easy sorting later on
    df_snv["copy_index"] = df_snv.index
    return df_snv.merge(df_preds, on = key, how = "left")

def run_indel_classifier(df, clf_indel_filepath):
    df_indel = df[(df["ref"].str.len() != 1) | (df["alt"].str.len() != 1)]
    df_indel_values = df_indel[key_and_features].dropna().values
    clf_indel = joblib.load(clf_indel_filepath)
    preds_indel_values = clf_indel.predict_proba(df_indel_values[:, 5:])
    df_preds  = pd.DataFrame({"chrom": df_indel_values[:, 0], "pos": df_indel_values[:, 1], "ref": df_indel_values[:, 2], "alt": df_indel_values[:, 3], "iid": df_indel_values[:, 4], "prob": preds_indel_values[:, 0]})
    df_indel["copy_index"] = df_indel.index
    return df_indel.merge(df_preds, on = key, how = "left")



