
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
pd.options.mode.chained_assignment = None  # default='warn'

def get_nb_f(mydf):
    p_lst = mydf.p_delong.tolist()
    i = 0
    while((p_lst[i]<0.05)|(p_lst[i+1]<0.05)|(p_lst[i+2]<0.05)):
        i+=1
    return i

dpath = '/Volumes/JasonWork/Projects/Meta_Dementia/'
#my_target = 'ACD'
my_target = 'AD'
#my_target = 'VaD'
target_df = pd.read_csv(dpath + 'Data/TargetOutcomes/'+ my_target +'/' + my_target + '_outcomes.csv')
met_auc_df = pd.read_csv(dpath + 'Results/'+ my_target + '/MetPanel/s1_AccumAUC.csv')
nb_f = get_nb_f(met_auc_df)
met_f_lst = met_auc_df.Met_code.tolist()[:nb_f]
met_df = pd.read_csv(dpath + 'Data/MetabolomicData/MetabolomicData.csv', usecols=['eid'] + met_f_lst)
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


y_train, y_yrs_train = mydf.target_y.tolist(), mydf.BL2Target_yrs.tolist()
X_train = mydf[met_f_lst]
my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, n_jobs=4, verbosity=-1, seed=2020)
my_lgb.set_params(**my_params)
calibrate = CalibratedClassifierCV(my_lgb, method='isotonic', cv=5)
calibrate.fit(X_train, y_train)
y_pred_train = np.round(calibrate.predict_proba(X_train)[:, 1], 10)
mydf['MetRS'] = y_pred_train
myout = mydf[['eid', 'Region_code', 'target_y', 'BL2Target_yrs', 'MetRS']]
myout.to_csv(dpath+'Results/AD/MetPanel/ADNI/' + my_target + '_MetRS_UKB.csv', index = False)
