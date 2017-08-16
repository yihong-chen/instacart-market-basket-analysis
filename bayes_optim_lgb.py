import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

import gc
from joblib import Parallel, delayed
import multiprocessing
from copy import deepcopy
from matplotlib import pyplot as plt
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score
import constants, feats, transactions, utils, evaluation, inference

pd.options.mode.chained_assignment = None
train = pd.read_hdf('/data/Instacart/train_vip.h5') # 扩充特征至2倍数
train, test = utils.train_test_split_users(train, 0.2) # shuffle user

del test
# lgb params
params =  {
    'boosting_type': 'gbdt', # 'rf' for random forest
    'objective': 'binary',
    'metric': 'binary_logloss,auc', # auc
    'learning_rate': 0.05,
    'early_stopping_round': 10,
    'scale_pos_weight': 1, 
    'verbose': 0,
    # 'max_bin': 63, 
    'num_threads': 28,
    'device': 'gpu',
    'gpu_device_id': 0,
    'gpu_platform_id': 0,
}

num_rounds = 600

# bayes params
num_iter = 10
init_points = 5

def lgb_cv(cv_train, cv_test, params, low_bound, topk, idx):
        print('CV Fold {}/5'.format(idx))
        params['gpu_device_id'] = idx - 1
        train_gid, train_feat, train_label = utils.preprocess_xgb(cv_train)
        del cv_train
        # print('| Training Data: Adding product id & aisle id & department id ...')
        # train_feat['product_id'], train_feat['aisle_id'], train_feat['department_id'] = train_gid['product_id'], train_gid['aisle_id'], train_gid['department_id']
        test_gid, test_feat, test_label = utils.preprocess_xgb(cv_test)
        del cv_test
        # print('| Test Data: Adding product id & aisle id & department id ...')
        # test_feat['product_id'], test_feat['aisle_id'], test_feat['department_id'] = test_gid['product_id'], test_gid['aisle_id'], test_gid['department_id']
        print('| Construct lgb Dataset ...')
        lgb_train = lgb.Dataset(train_feat, train_label, free_raw_data=True)#, categorical_feature=['product_id', 'aisle_id', 'department_id'])
        del train_feat, train_label
        # lgb_test = lgb.Dataset(test_feat, test_label, free_raw_data=True)
        print('| Training ...')
        gbm = lgb.train(params, 
                    lgb_train, 
                    num_boost_round=num_rounds,
                    valid_sets=lgb_train)
        del lgb_train
        y_scores = gbm.predict(test_feat, num_iteration = gbm.best_iteration)
        del test_feat
        test_auc_score = roc_auc_score(test_label, y_scores)
        print('| test auc: %s'%test_auc_score)   
        gc.collect()

        user_product = test_gid[['user_id', 'product_id', 'order_id']]
        user_product['label'] = test_label
        user_product['score'] = y_scores
        user_product = user_product.sort_values(['user_id', 'order_id', 'score'], ascending = False)
        gold = evaluation.get_gold(user_product) 
        op = user_product.copy()
        # op = utils.shing_f1_optim(op, low_bound, int(topk))
        op = utils.faron_f1_optim(op, low_bound, int(topk))
        op['products'] = op['products'].apply(lambda x: [int(i) if i != 'None' else i for i in x.split()])
        op = pd.merge(pd.DataFrame({'order_id':user_product.order_id.unique()}),
                       op, on = ['order_id'], how = 'left')

        res = evaluation.evaluation(gold, op[['order_id', 'products']])
        mf1 = res.f1score.mean()
        with open(constants.LGB_DIR + 'lgb_{}_{:.6f}_{:.6f}'.format(params['boosting_type'], 
                                                                test_auc_score,
                                                                mf1), 'wb') as f:
            pickle.dump(gbm, f, pickle.HIGHEST_PROTOCOL)
            del gbm
        print('F1 Optimization Result: mean-f1-score {}'.format(mf1))
        del user_product, op, gold, res
        gc.collect()
        return mf1
    
def lgb_evaluate(# params
                 num_leaves,
                 max_depth,
                 min_data_in_leaf,
                 feature_fraction,
                 bagging_freq,
                 bagging_fraction,
                 # f1 params
                 low_bound,
                 topk
                 ):# end of params
       
    params['num_leaves'] = int(num_leaves)
    params['max_depth'] = int(max_depth)
    params['min_data_in_leaf'] = int(min_data_in_leaf)
    params['feature_fraction'] = feature_fraction
    params['bagging_fraction'] = bagging_fraction
    params['bagging_freq'] = int(bagging_freq)

    cv = GroupKFold(n_splits=4)
    # cv_test_auc = []
    cv_mean_f1 = []
    cv_train = []
    cv_test = []
    for i, (train_index, test_index) in enumerate(cv.split(train, groups=train['user_id'].values)):
        cv_train.append(train.iloc[train_index])
        cv_test.append(train.iloc[test_index])
    del cv, train_index, test_index
    gc.collect()
    cv_mean_f1 = Parallel(n_jobs=3, temp_folder='/data/tmp/')(delayed(lgb_cv)(tra, 
                                                                              tes, 
                                                                              params, 
                                                                              low_bound,
                                                                              topk,
                                                                              idx) for tra, tes, idx in zip(cv_train, cv_test, [1, 2, 3, 4, 5]))
    del cv_train, cv_test
    gc.collect()
    return 100 * np.mean(cv_mean_f1)
    # return np.mean(cv_test_auc)

lgbBO = BayesianOptimization(lgb_evaluate, {'num_leaves': (64, 256),
                                            'max_depth': (7, 12),
                                            'min_data_in_leaf': (10, 100),
                                            'feature_fraction': (0.6, 1),
                                            'bagging_freq': (5, 20),
                                            'bagging_fraction': (0.6, 1),
                                            # f1 params
                                            'low_bound': (0, 0.1),
                                            'topk': (80, 100)
                                                })

lgbBO.maximize(init_points=init_points, n_iter=num_iter)