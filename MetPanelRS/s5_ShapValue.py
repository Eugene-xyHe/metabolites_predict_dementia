
import os
import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from Utility.DelongTest import delong_roc_test
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import scipy.stats as stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
import shap
import matplotlib.pyplot as plt

def get_nb_f(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)|(p_lst[i+2]<0.05)):
        i+=1
    return i

dpath = '/Volumes/JasonWork/Projects/Meta_Dementia/'
my_target = 'ACD'
#my_target = 'AD'
#my_target = 'VaD'
output_img_dir = dpath + 'Results/'+ my_target + '/MetPanel/s5_Shap_Plot.png'
met_code = pd.read_csv(dpath + 'Data/MetabolomicData/MetCode.csv', usecols=['Met_code', 'Met_name'])
met_dict = dict(zip(met_code.Met_code, met_code.Met_name))
target_df = pd.read_csv(dpath + 'Data/TargetOutcomes/'+ my_target +'/' + my_target + '_outcomes.csv')
met_auc_df = pd.read_csv(dpath + 'Results/'+ my_target + '/MetPanel/s1_AccumAUC.csv')
nb_f = get_nb_f(met_auc_df)
nb_f = 15
met_f_lst = met_auc_df.Met_code.tolist()[:nb_f]
met_df = pd.read_csv(dpath + 'Data/MetabolomicData/MetabolomicData.csv', usecols=['eid'] + met_f_lst)
met_df.rename(columns=met_dict, inplace=True)
met_f_lst = list((pd.Series(met_f_lst)).map(met_dict))
info_df = pd.read_csv(dpath + 'Data/Eid_info_data.csv', usecols = ['eid', 'Region_code'])

mydf = pd.merge(target_df, info_df, how = 'inner', on = ['eid'])
mydf = pd.merge(mydf, met_df, how = 'inner', on = ['eid'])
mydf.reset_index(inplace=True, drop=True)
fold_id_lst = [i for i in range(10)]

my_params = {'n_estimators': 500,
             'max_depth': 15,
             'num_leaves': 10,
             'subsample': 0.7,
             'learning_rate': 0.01,
             'colsample_bytree': 0.7}


X = mydf[met_f_lst]
y = mydf[['target_y']]
mykf = StratifiedKFold(n_splits = 10, random_state = 2023, shuffle = True)

for train_idx, test_idx in mykf.split(X, y):
    X_train, y_train = X.iloc[train_idx,:], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx,:], y.iloc[test_idx]

my_lgb = LGBMClassifier(objective = 'binary', metric = 'auc', is_unbalance = True, verbosity = 1, seed = 2023)
my_lgb.set_params(**my_params)
my_lgb.fit(X_train, y_train)

explainer = shap.Explainer(my_lgb)
shap_values = explainer(X_test)
shap.plots.beeswarm(shap_values[:, :, 1], max_display=X.shape[1], order=list(np.linspace(0, nb_f, nb_f+1).astype('uint8')))
plt.gcf().set_size_inches(19, 6)
ax = plt.gca()
ax.set_ylabel('Selected Metabolites', fontsize = 18, weight = 'bold')
ax.set_xlabel('SHAP Values', fontsize = 14, weight = 'bold')
ax.tick_params(axis='x', labelsize=14)
ylabels = [tick.get_text() for tick in ax.get_yticklabels()]
ax.set_yticklabels(ylabels, fontsize = 16, color = 'black')
plt.tight_layout()

#plt.savefig(output_img_dir, dpi=200)
