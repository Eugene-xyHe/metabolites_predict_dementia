
import os
import numpy as np
import pandas as pd
from Utility.Training_Utilities import *
from lightgbm import LGBMClassifier
import warnings
import re
import shap
pd.options.mode.chained_assignment = None  # default='warn'

dpath = '/Volumes/JasonWork/Projects/Meta_Dementia/'
my_target = 'ACD'
#my_target = 'AD'
#my_target = 'VaD'
target_df = pd.read_csv(dpath + 'Data/TargetOutcomes/'+ my_target +'/' + my_target + '_outcomes.csv')
met_f_df = pd.read_csv(dpath + 'Results/Cox/nmr_Cox_'+my_target+ '.csv')
met_f_df = met_f_df.loc[met_f_df.fdr<0.05]
met_f_lst = met_f_df.Characteristics.tolist()
len(met_f_lst)
met_df = pd.read_csv(dpath + 'Data/MetabolomicData/MetabolomicData.csv', usecols = ['eid'] + met_f_lst)
#met_f_lst = met_df.columns.tolist()[1:]
info_df = pd.read_csv(dpath + 'Data/Eid_info_data.csv')

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

def normal_imp(mydict):
    mysum = sum(mydict.values())
    mykeys = mydict.keys()
    for key in mykeys:
        mydict[key] = mydict[key]/mysum
    return mydict

tg_imp_cv = Counter()
tc_imp_cv = Counter()
shap_imp_cv = np.zeros(len(met_f_lst))

for fold_id in fold_id_lst:
    train_idx = mydf['Region_code'].index[mydf['Region_code'] != fold_id]
    test_idx = mydf['Region_code'].index[mydf['Region_code'] == fold_id]
    X_train, X_test = mydf.iloc[train_idx][met_f_lst], mydf.iloc[test_idx][met_f_lst]
    y_train, y_test = mydf.iloc[train_idx].target_y, mydf.iloc[test_idx].target_y
    my_lgb = LGBMClassifier(objective = 'binary', metric = 'auc', is_unbalance = True, verbosity = 1, seed = 2023)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)
    totalgain_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    totalgain_imp = dict(zip(my_lgb.booster_.feature_name(), totalgain_imp.tolist()))
    totalcover_imp = my_lgb.booster_.feature_importance(importance_type='split')
    totalcover_imp = dict(zip(my_lgb.booster_.feature_name(), totalcover_imp.tolist()))
    tg_imp_cv += Counter(normal_imp(totalgain_imp))
    tc_imp_cv += Counter(normal_imp(totalcover_imp))
    explainer = shap.TreeExplainer(my_lgb)
    shap_values = explainer.shap_values(X_test)
    shap_values = np.abs(np.average(shap_values[0], axis=0))
    shap_imp_cv += shap_values / np.sum(shap_values)


shap_imp_df = pd.DataFrame({'Met_code': met_f_lst,
                            'ShapValues_cv': shap_imp_cv/len(fold_id_lst)})
shap_imp_df.sort_values(by = 'ShapValues_cv', ascending = False, inplace = True)

tg_imp_cv = normal_imp(tg_imp_cv)
tg_imp_df = pd.DataFrame({'Met_code': list(tg_imp_cv.keys()),
                          'TotalGain_cv': list(tg_imp_cv.values())})

tc_imp_cv = normal_imp(tc_imp_cv)
tc_imp_df = pd.DataFrame({'Met_code': list(tc_imp_cv.keys()),
                          'TotalCover_cv': list(tc_imp_cv.values())})

my_imp_df = pd.merge(left = shap_imp_df, right = tg_imp_df, how = 'left', on = ['Met_code'])
my_imp_df = pd.merge(left = my_imp_df, right = tc_imp_df, how = 'left', on = ['Met_code'])
my_imp_df['Ensemble'] = (my_imp_df['ShapValues_cv'] + my_imp_df['TotalGain_cv'] + my_imp_df['TotalCover_cv'])/3
my_imp_df.sort_values(by = 'TotalGain_cv', ascending = False, inplace = True)

my_imp_df.to_csv(dpath + 'Results/'+ my_target + '/MetPanel/s0_MetImportance.csv', index = False)

print('finished')



