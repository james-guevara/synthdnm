import numpy as np
import pandas as pd

df_train = pd.read_csv("df_dnm_features_table.tsv", sep = "\t")

# Will add feature list...
features = ["VQSLOD"]
df_train = df_train[[features]]

X = df_train[:, 0:-1]
y = df_train[:, -1]

# We should vary the hyperparameters in a grid search
clf_snv = RandomForestClassifier(
    n_estimators = 125,
    max_features = df_train.shape[1] - 1,
    max_depth = 450,
    min_samples_split = 2,
    min_samples_leaf = 1,
    min_weight_fraction_leaf = 0.0,
    max_leaf_nodes = None,
    n_jobs = -1,
    class_weight = "balanced",
    random_state = 42,
    verbose = 0)
clf_snv.fit(X, y)
dump(clf_snv, "clf_snv.joblib")


# Grid search (will certainly need to clean this up...)

# bash script... (the numbers correspond to hyperparameter arguments)
# python pydnm_gridsearch.py 250 auto 175 0.01 0.05 0.05 None
# python pydnm_gridsearch.py 250 auto 175 0.01 0.05 0.01 None
# python pydnm_gridsearch.py 250 auto 175 0.01 0.05 0 None
# python pydnm_gridsearch.py 250 auto 175 0.01 1 0.05 None
# python pydnm_gridsearch.py 250 auto 175 0.01 1 0.01 None
# python pydnm_gridsearch.py 250 auto 175 0.01 1 0 None
# python pydnm_gridsearch.py 250 auto 175 0.05 0.1 0.05 None
# python pydnm_gridsearch.py 250 auto 175 0.05 0.1 0.01 None
# python pydnm_gridsearch.py 250 auto 175 0.05 0.1 0 None
# python pydnm_gridsearch.py 250 auto 175 0.05 0.05 0.05 None
# python pydnm_gridsearch.py 250 auto 175 0.05 0.05 0.01 None
# python pydnm_gridsearch.py 250 auto 175 0.05 0.05 0 None
# python pydnm_gridsearch.py 250 auto 175 0.05 1 0.05 None
# python pydnm_gridsearch.py 250 auto 175 0.05 1 0.01 None
# python pydnm_gridsearch.py 250 auto 175 0.05 1 0 None
# python pydnm_gridsearch.py 250 auto 175 2 0.1 0.05 None
# python pydnm_gridsearch.py 250 auto 175 2 0.1 0.01 None
# python pydnm_gridsearch.py 250 auto 175 2 0.1 0 None


# pydnm_gridsearch.py
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score,roc_auc_score
# import matplotlib.pyplot as plt 
# import seaborn as sns
# import scipy
# import random,string
# def randomword(length):
# 	letters = string.ascii_lowercase
# 	return ''.join(random.choice(letters) for i in range(length))
# def mean_ci(data, confidence=0.95):
# 	if len(data)==0: return 'nan','nan','nan'
# 	a = 1.0*np.array(data)
# 	n = len(a)
# 	m, se = np.mean(a), scipy.stats.sem(a)
# 	h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
# 	return m, m-h, m+h
# def benchit(df,clf,ofh,param):
# 	###############
# 	# FEATURES
# 	start_col=12
# 	end_col=36
# 	###############
# 	X=df.as_matrix(columns=df.columns[np.r_[start_col:end_col]])
# 	y=np.array(df.TRUTH)
# 	cv = StratifiedKFold(n_splits=10,shuffle=True)
# 	_fdr,_f1,_acc,_auc,_tpr = [],[],[],[],[]
# 	for train_ind, test_ind in cv.split(X,y):
# 		X_train, X_test = X[train_ind], X[test_ind]
# 		y_train, y_test = y[train_ind], y[test_ind]
# 		clf.fit(X_train,y_train)
# 		pred = clf.predict(X_test)
# 		acc = accuracy_score(y_test,pred)
# 		auc=roc_auc_score(y_test, pred)
# 		tn,fp,fn,tp = confusion_matrix(y_test,pred).ravel()
# 		tpr=tp/(tp+fn)
# 		if tp+fp>0: fdr = fp / float(tp+fp)
# 		else: fdr='nan'
# 		if len(set(pred))!=1:
# 			f1 = f1_score(y_test,pred)
# 			auc = roc_auc_score(y_test,pred)
# 		else: 
# 			f1='nan'
# 			auc='nan'
# 		if fdr !='nan': _fdr.append(fdr)
# 		if f1 != 'nan': _f1.append(f1)
# 		if acc != 'nan' : _acc.append(acc)
# 		if auc != 'nan' : _auc.append(auc)
# 	o = list(mean_ci(_fdr))
# 	o += list(mean_ci(_f1))
# 	o += list(mean_ci(_acc))
# 	o += list(mean_ci(_auc))
# 	o += list(mean_ci(_tpr))
# 	ofh.write("{}\t{}\n".format(param,'\t'.join(map(str,o))))
# 	ofh.close()
# from sklearn.model_selection import StratifiedKFold
# import sys

# ####################################
# n_estimators = sys.argv[1]
# max_features = sys.argv[2]
# max_depth = sys.argv[3]
# min_samples_split = sys.argv[4]
# min_samples_leaf = sys.argv[5]
# min_weight_fraction_leaf = sys.argv[6]
# max_leaf_nodes = sys.argv[7]
# ####################################

# ####################################
# n_estimators = int(n_estimators)
# #
# if max_features != 'auto': max_features = int(max_features)
# ##
# max_depth = int(max_depth)
# ###
# if min_samples_split == '2': min_samples_split = int(min_samples_split)
# else : min_samples_split = float(min_samples_split)
# ####
# if min_samples_leaf == '1': min_samples_leaf=int(min_samples_leaf)
# else: min_samples_leaf = float(min_samples_leaf)
# #####
# min_weight_fraction_leaf = float(min_weight_fraction_leaf)
# ######
# if max_leaf_nodes != "None": max_leaf_nodes = int(max_leaf_nodes)
# else: max_leaf_nodes = None
# ####################################
# _params = 'n_estimators:{},max_features:{},max_depth:{},min_samples_split:{},min_samples_leaf:{},min_weight_fraction_leaf:{},max_leaf_nodes:{}'.format(n_estimators,max_features,max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_leaf_nodes)

# print ('PYDNM GRID SEARCH')
# print ('-----------------')
# print ('  '+_params.replace(',','\n  '))

# sys.stderr.write('PYDNM GRID SEARCH\n-----------------\n  '+_params.replace(',','\n  ')+'\n')

# _tag = '{}-{}-{}-{}-{}-{}-{}'.format(n_estimators,max_features,max_depth,min_samples_split,min_samples_leaf,min_weight_fraction_leaf,max_leaf_nodes) 


# # df = pd.read_csv("training_set.indels.reach-jg.autosome.txt",sep="\t")
# df = pd.read_csv("training_set.snps.reach-jg.autosome.txt",sep="\t")
# df["ClippingRankSum"] = 0
# df = df.dropna()
# df = df[~(df.chrom.str.contains("X"))]
# df = df[~(df.chrom.str.contains("Y"))]


# ofh = open("grid_search_snps/training_gridsearch_run{}.txt".format(_tag),'w')
# ofh.write('params\tmean_fdr\tlo_fdr\thi_fdr\tmean_f1\tlo_f1\thi_f1\tmean_acc\tlo_acc\thi_acc\tmean_auc\tlo_auc\thi_auc\tmean_tpr\tlo_tpr\thi_tpr\n')
# #############################
# ########### C L F ###########
# #############################
# clf = RandomForestClassifier(
# 	n_estimators=n_estimators,
# 	max_features=max_features,
# 	max_depth=max_depth,
# 	min_samples_split=min_samples_split,
# 	min_samples_leaf=min_samples_leaf,
# 	min_weight_fraction_leaf=min_weight_fraction_leaf,
# 	max_leaf_nodes=max_leaf_nodes,
# 	n_jobs=-1,
# 	class_weight="balanced",
# 	random_state=42,
# 	verbose=0)
# #############################
# benchit(df,clf,ofh,_params)
# #############################
# #df = pd.read_csv("/home/dantakli/pydnm_training/with_segdups/pydnm_test.txt",sep="\t")
# #df = df.dropna()
# #ofh = open("/home/dantakli/pydnm_training/with_segdups/gridsearch/test_gridsearch_run{}.txt".format(_tag),'w')
# #ofh.write('params\tmean_fdr\tlo_fdr\thi_fdr\tmean_f1\tlo_f1\thi_fi\tmean_acc\tlo_acc\thi_acc\tmean_auc\tlo_auc\thi_auc\n')
# #benchit(df,clf,ofh,_params)

