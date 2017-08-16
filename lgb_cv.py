import gc; import pickle; from copy import deepcopy
import numpy as np; import pandas as pd
import lightgbm as lgb

from matplotlib import pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, roc_auc_score

import constants, feats, transactions, utils, evaluation, inference
pd.options.mode.chained_assignment = None

train = pd.read_hdf('/data/Instacart/train_vip.h5')
train, test = utils.train_test_split_users(train, 0.2) # shuffle user

params =  {
    'boosting_type': 'gbdt', # 'rf' for random forest
    'objective': 'binary',
    'metric': 'binary_logloss,auc', # auc
    'num_leaves': 256,
    'max_depth': 8,
    'min_data_in_leaf': 200, # avoid over_fitting
    # 'min_sum_hessian_in_leaf': 20, # avoid over fitting
    'learning_rate': 0.025,
    "device" : "gpu",
    'gpu_device_id': 3,
    'gpu_platform_id': 0,
    # 'feature_fraction': 0.8, # colsample
    'bagging_fraction': 0.8, # subsample # avoid overfitting & speed up
    # 'bagging_freq': 5,
    'early_stopping_round': 50,
    # lambda_l1,
    # lambda_l2,
    # min_gain_to_split
    # 'is_unbalance':True,
    'scale_pos_weight': 1, 
    'verbose': 0,
    'num_threads': 28
}
num_rounds = 1000

cv = GroupKFold(n_splits=5)
cv_test_auc = []
for i, (train_index, test_index) in enumerate(cv.split(train, groups=train['user_id'].values)):
    print('CV fold {}/5'.format(i))
    params['gpu_device_id'] = i % 3
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    train_gid, train_feat, train_label = utils.preprocess_xgb(cv_train)
    del cv_train
    # print('Training Data: Adding product id & aisle id & department id ...')
    # train_feat['product_id'], train_feat['aisle_id'], train_feat['department_id'] = train_gid['product_id'], train_gid['aisle_id'], train_gid['department_id']
    
    test_gid, test_feat, test_label = utils.preprocess_xgb(cv_test)
    del cv_test
    # print('Test Data: Adding product id & aisle id & department id ...')
    # test_feat['product_id'], test_feat['aisle_id'], test_feat['department_id'] = test_gid['product_id'], test_gid['aisle_id'], test_gid['department_id']
    
    print('Construct lgb Dataset ...')
    lgb_train = lgb.Dataset(train_feat, train_label, free_raw_data=True)#, categorical_feature=['product_id', 'aisle_id', 'department_id'])
    del train_feat, train_label
    
    print('Training ...')
    gbm = lgb.train(params, 
                    lgb_train, 
                    num_boost_round=num_rounds,
                    valid_sets=lgb_train)
    del lgb_train
    y_scores = gbm.predict(test_feat, num_iteration = gbm.best_iteration)
    del test_feat
    test_auc_score = roc_auc_score(test_label, y_scores)
    print('test auc: %s'%test_auc_score)
    cv_test_auc.append(test_auc_score)   

    # user_product = test_gid[['user_id', 'product_id', 'order_id']]
    # user_product['label'] = test_label   
    # user_product['score'] = y_scores
    # gold = evaluation.get_gold(user_product) 

    # Shing's
    # op = user_product.copy()
    # op = utils.shing_f1_optim(op, low_bound=0.01, topk=95)
    # op['products'] = op['products'].apply(lambda x: [int(i) if i != 'None' else i for i in x.split()])
    # op = pd.merge(pd.DataFrame({'order_id':user_product.order_id.unique()}),
    #                op, on = ['order_id'], how = 'left')
    # res = evaluation.evaluation(gold, op[['order_id', 'products']])
    # shing_f1 = res.f1score.mean()
    # print('Sh1ng F1 Optimization Result: mean-f1-score {}'.format(shing_f1))
    # del op, res
    # Faron's
    # op = user_product.copy()
    # op = utils.tarbox_f1_optim(op, low_bound=0.01)
    # op['products'] = op['products'].apply(lambda x: [int(i) if i != 'None' else i for i in x.split()])
    # op = pd.merge(pd.DataFrame({'order_id':user_product.order_id.unique()}),
    #               op, on = ['order_id'], how = 'left')
    # res = evaluation.evaluation(gold, op[['order_id', 'products']])
    # faron_f1 = res.f1score.mean()
    # print('Faron F1 Optimization Result: mean-f1-score {}'.format(faron_f1))
    # del op, res, user_product
    with open(constants.LGB_DIR + 'lgb_train_vip_{}_{:.6f}'.format(params['boosting_type'], 
                                                                test_auc_score, 
                                                                ), 'wb') as f:
        pickle.dump(gbm, f, pickle.HIGHEST_PROTOCOL)
        del gbm
    gc.collect()