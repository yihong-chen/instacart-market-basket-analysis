import feats, utils, constants, inference, evaluation, transactions

import pickle
from time import time
from copy import  deepcopy
from importlib imMport reload

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold,train_test_split

from matplotlib import pyplot as plt
from matplotlib.pylab import rcParams

pd.options.mode.chained_assignment = None
train = pd.read_hdf('/data/Instacart/train_vip.h5')

param = {'eta':0.025, # 0.01
         'booster':'gbtree',
         'max_depth': 7, # 7 
         'min_child_weight': 200,
         'gamma': 10,
         'subsample':0.8,
         'colsample_bytree': 0.8,
         'scale_pos_weight': 1, # 9.25, 
         # 'eta': 0.3, # ?
         'silent': 1,
         'nthread':20,
         # 'max_delta_step': 100, # 10, # 5, #0,
         # 'reg_alpha':0,
         # 'reg_lambda':0,
         'objective': 'binary:logistic',
         # 'tree_method': 'gpu_hist', 
         # 'tree_method': 'hist',
         # 'tree_method': 'exact',
         # 'gpu_id': 3,
         # 'n_gpus': 4,
         'seed':2018,
         'eval_metric': 'auc'}
num_round = 2000 # 1000 # 2000, 4000

print('Train/Test split by user ... ...')
train, test = utils.train_test_split_users(train, 0.2) # shuffle user
print('Preprocess ... ...')
train_gid, train_feat, train_label = utils.preprocess_xgb(train)
test_gid, test_feat, test_label = utils.preprocess_xgb(test)
del train, test
print('Train/Eval split ... ...')
X_train, X_eval, y_train, y_eval = train_test_split(train_feat, train_label, test_size = 0.2)
del train_feat, train_label
print('Create DMatrix ... ...')
dtrain = xgb.DMatrix(X_train, label=y_train)
deval = xgb.DMatrix(X_eval, label=y_eval)
dtest = xgb.DMatrix(test_feat, label=test_label)
del X_train, y_train, X_eval, y_eval, test_feat
eval_list = [(dtrain, 'train'), (deval, 'eval')]
print('Train ... ...')
bst = xgb.train(param, dtrain, num_round, eval_list, early_stopping_rounds=50, maximize=True)

y_scores = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
test_auc_score = roc_auc_score(test_label, y_scores)
print('test auc: %s'%test_auc_score)

###  Feature Importance

plt.figure(figsize=(10, 20))
feat_imp = pd.Series(bst.get_fscore()).sort_values(ascending=False) 
feat_imp.plot(kind='barh', title='Feature Importances') 
plt.ylabel('Feature Importance Score')
plt.savefig('feature importance.png')

### Evaluation

user_product = test_gid[['user_id', 'product_id', 'order_id']]
user_product['label'] = test_label
user_product['score'] = y_scores
gold = evaluation.get_gold(user_product) 

# 2mins 0.3753
op = user_product.copy()
op = utils.shing_f1_optim(op, low_bound=0, topk=95)
op['products'] = op['products'].apply(lambda x: [int(i) if i != 'None' else i for i in x.split()])

op = pd.merge(pd.DataFrame({'order_id':user_product.order_id.unique()}),
                    op, on = ['order_id'], how = 'left')

res = evaluation.evaluation(gold, op[['order_id', 'products']])

print('F1 Optimization Result: mean-f1-score {}'.format(res.f1score.mean()))

# 9 mins 0.3765
op = user_product.copy()
op = utils.tarbox_f1_optim(op, low_bound=0)
op['products'] = op['products'].apply(lambda x: [int(i) if i != 'None' else i for i in x.split()])
op = pd.merge(pd.DataFrame({'order_id':user_product.order_id.unique()}),
                    op, on = ['order_id'], how = 'left')
res = evaluation.evaluation(gold, op[['order_id', 'products']])

print('F1 Optimization Result: mean-f1-score {}'.format(res.f1score.mean()))

with open(constants.XGB_DIR + 'xgb_vip_{}_{:.6f}_{:.6f}'.format(param['booster'],
                                                                 test_auc_score,
                                                                 res.f1score.mean()), 'wb') as f:
    pickle.dump(bst, f, pickle.HIGHEST_PROTOCOL)