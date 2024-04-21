
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
my_target = 'ACD'
my_target = 'AD'
my_target = 'VaD'
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

for fold_id in fold_id_lst:
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    eid_train, eid_test = mydf.eid.iloc[train_idx].tolist(), mydf.eid.iloc[test_idx].tolist()
    region_train, region_test = mydf.Region_code.iloc[train_idx].tolist(), mydf.Region_code.iloc[test_idx].tolist()
    y_yrs_train, y_yrs_test = mydf.BL2Target_yrs.iloc[train_idx].tolist(), mydf.BL2Target_yrs.iloc[test_idx].tolist()
    X_train, X_test = mydf.iloc[train_idx][met_f_lst], mydf.iloc[test_idx][met_f_lst]
    X_train.reset_index(inplace = True, drop = True)
    X_test.reset_index(inplace = True, drop = True)
    y_train, y_test = mydf.target_y.iloc[train_idx].tolist(), mydf.target_y.iloc[test_idx].tolist()
    my_lgb = LGBMClassifier(objective='binary', metric='auc', is_unbalance=False, n_jobs=4, verbosity=-1, seed=2020)
    my_lgb.set_params(**my_params)
    calibrate = CalibratedClassifierCV(my_lgb, method='isotonic', cv=5)
    calibrate.fit(X_train, y_train)
    y_pred_train = np.round(calibrate.predict_proba(X_train)[:, 1], 10)
    y_pred_test = np.round(calibrate.predict_proba(X_test)[:, 1], 10)
    tmp_train_df = pd.DataFrame([eid_train, region_train, y_train, y_yrs_train, y_pred_train]).T
    tmp_train_df.columns = ['eid', 'Region_code', 'target_y', 'BL2Target_yrs', 'MetRS']
    tmp_train_df[['eid', 'Region_code', 'target_y']] = tmp_train_df[['eid', 'Region_code', 'target_y']].astype('int')
    tmp_test_df = pd.DataFrame([eid_test, region_test, y_test, y_yrs_test, y_pred_test]).T
    tmp_test_df.columns = ['eid', 'Region_code', 'target_y', 'BL2Target_yrs', 'MetRS']
    tmp_test_df[['eid', 'Region_code', 'target_y']] = tmp_test_df[['eid', 'Region_code', 'target_y']].astype('int')
    os.mkdir(dpath + 'Results/'+ my_target + '/MetPanel/MetRS/TestFold' + str(fold_id))
    tmp_train_df.to_csv(dpath + 'Results/'+ my_target + '/MetPanel/MetRS/TestFold' + str(fold_id) + '/training.csv', index = False)
    tmp_test_df.to_csv(dpath + 'Results/'+ my_target + '/MetPanel/MetRS/TestFold' + str(fold_id) + '/testing.csv', index = False)

