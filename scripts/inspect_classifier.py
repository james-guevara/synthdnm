import joblib
clf_snv = joblib.load("ssc-jg-snp-clf-half-sample.joblib")
clf_indel = joblib.load("ssc-jg-indel-clf-half-sample.joblib")
print(clf_snv.get_params())
print(clf_indel.get_params())
