import pickle
import numpy as np
import pandas as pd

import itertools
import multiprocessing
from joblib import Parallel, delayed
from mp_generic import mp_groupby

from math import sqrt
from scipy.stats import entropy
from sklearn.utils import shuffle
from numpy.random import binomial, beta
from sklearn.metrics import f1_score, recall_score, precision_score, mean_squared_error

import lightgbm as lgb
import xgboost as xgb 

import constants, inference
from f1optim import F1Optimizer # disable numba acceleration

###### Log  Extraction

def flatten_multiidx(df):
    '''
        Given a df where the columns are multiindex(>=2 levels), flat it into one-level index
        Useful for dealing with groupby-agg results
    Args:
        df: pandas DataFrame
    Return:
        pandas DataFrame with flatten index
    '''
    def sel_level(col):
        '''
            Select which level of index to use as new col names
        Args:
            col: tuple, (col_name_level_0, col_name_level_1, col_name_level_2, ... )
        Return:
            col: string, new col name
        Example:
            col = ('price', 'max') --> 'price_max' 
        '''
        col = [level for level in col if level != '']
        return '_'.join(col)
    
    df.columns = [sel_level(col) for col in df.columns.values]
    return df

######BEFORE TRAIN UTILS

def train_test_split_users(train, test_size, seed = 1993):
    
    train_size = 1 - test_size

    np.random.seed(seed)
    uids = np.random.permutation(train.user_id.unique())

    train_uids = uids[:int(train_size * len(uids))]
    test_uids = uids[int(train_size * len(uids)):]

    test = train[train.user_id.isin(test_uids)]
    train = train[train.user_id.isin(train_uids)]
    
    return train, test


def check_inf_nan(df):
    '''
        check if there exists np.inf, -np.inf in df
    '''
    print("Checking inf ...")
    print(df[np.isinf(df)].stack()) # stack:turn cols into rows & throw out NAN 
    print("Checking NAN ...")
    print(df.columns[df.isnull().any()])
    return True



def preprocess_xgb(train, is_submission = False, label_col = 'label'):
    '''
    figure out feat columns, id columns & label columns
    label_col = ['label', 'label_none']
    '''
    if label_col == 'label':
        id_cols = ['user_id', 'product_id', 'aisle_id', 'department_id', 'order_id']
    else:
        id_cols = ['user_id', 'order_id']
    feat_cols = [x for x in train.columns if not x in id_cols + ['label_none', 'label']]
    ids = train[id_cols]
    feats = train[feat_cols]
    if not is_submission:
        labels = train[label_col]
        return ids, feats, labels
    else:
        return ids, feats
    
def feat_imp_load(prefix):
    '''
        load feature importance
    '''
    with open(constants.FEAT_DATA_DIR + 'feat_imp_%s.pkl'%prefix, 'rb') as f:
        feat_imp = pickle.load(f)
    return feat_imp

def feat_imp_cache(feat_imp, prefix):
    '''
        cache feature importance
    '''
    with open(constants.FEAT_DATA_DIR + 'feat_imp_%s.pkl'%prefix, 'wb') as f:
        pickle.dump(feat_imp, f, pickle.HIGHEST_PROTOCOL)
    print('Successfully pickled feat_imp_%s.pkl!'%prefix)


def feat_check_out(train, rm_cols):
    '''
        remove feat cols in rm_cols
    '''
    for col in rm_cols:
        del train[col]
    return train

####### LDA UTILS
def series_to_str(subf):
    '''turn a series into space separated string',
    useful for creating sentenses'''
    subf = subf.astype(str)
    r = ' '.join(subf)
    return r

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print("\n".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def sym_kl_distance(p, q):
    return 0.5*(entropy(p, q) + entropy(q, p))

########## Transaction Feats
def is_organic(row):
    if 'organic' in row or 'Organic' in row:
        return 1
    else:
        return 0

########## Prediction Utils

def get_feat_col(bst):
    if isinstance(bst, lgb.basic.Booster):
        return bst.feature_name()
    else:
        return bst.feature_names

def get_predition(bst, feat):
    '''
        bst: lgb model or xgb model
        feat: pd.DataFrame
    '''
    if isinstance(bst, lgb.basic.Booster):
        return bst.predict(feat, num_iteration=bst.best_iteration)
    else:
        dfeat = xgb.DMatrix(feat)
        return bst.predict(dfeat, ntree_limit=bst.best_ntree_limit)

    
########## ORDER BASED THRESHOLDING UTILS
def shing_f1_optim(user_product, low_bound=0.01, topk = 80):
    user_product['not_a_product'] = 1 -user_product['score']
    gp = user_product.groupby('order_id')['not_a_product'].apply(lambda x: np.multiply.reduce(x.values)).reset_index()
    gp.rename(columns={'not_a_product': 'score'}, inplace=True)
    gp['product_id'] = 50000 # none_idx

    # 将none与正常商品组合在一起
    user_product = pd.concat([user_product, gp], axis=0)
    user_product.product_id = user_product.product_id.astype(np.int32)
    # throw away small value
    user_product = user_product.loc[user_product.score > low_bound, ['order_id', 'score', 'product_id']]
    user_product = applyParallel(user_product.groupby(['order_id']), construct_orders, topk).reset_index()
    return user_product

def tarbox_f1_optim(user_product, low_bound=0.01):
    user_product = user_product.loc[user_product.score > low_bound, ['order_id', 'product_id', 'score']]
    # Group products per order here
    user_product = applyParallel(user_product.groupby(user_product.order_id), construct_orders_faron).reset_index()
    return user_product
    
def faron_f1_optim(user_product):
    u_scores = user_product.groupby(['user_id', 'order_id']).apply(lambda x: np.sort(np.array(x.score))[::-1][:80]).reset_index() # top 80
    u_scores.columns = ['user_id', 'order_id', 'score']
    
    is_pNone_given = False
    args = [is_pNone_given]
    mp_args = {'n_cpus': 28, 'queue': True, 'n_queues': None}
    u_scores = optim_f1_k(u_scores, args, mp_args)

    user_product = pd.merge(user_product, 
                        u_scores[['user_id', 'order_id', 'thres', 'pred_none']], 
                        on = ['user_id', 'order_id'], how = 'left')
    
    pred = inference.order_based_search(user_product)
    up = pd.DataFrame({'order_id':user_product.order_id.unique()})
    pred = pd.merge(up, pred, on = ['order_id'], how = 'left')
    return pred, user_product, u_scores

def construct_orders(grp, topk=80):
    '''
    based on shing's baseline
    '''
    products = grp.product_id.values # products in the order
    prob = grp.score.values # reorder prob of products in the order
    sort_idx = np.argsort(prob)[::-1] # descending order
    values = f1_fast_search(prob[sort_idx][:topk], dtype=np.float64) # use the largest 80 prob
    index = np.argmax(values)
    # print('product num {}, optimal value: top {}'.format(grp.shape[0], index))
    best = ' '.join(map(lambda x: str(x) if x != 50000 else 'None', products[sort_idx][0:index]))
    grp = grp[0:1] # keep order_id
    grp.loc[:, 'products'] = best
    return grp

def construct_orders_faron(df, func_args=None):
    # print(df.product_id.values.shape)
    products = df.product_id.values
    prob = df.score.values

    sort_index = np.argsort(prob)[::-1]
    L2 = products[sort_index]
    P2 = prob[sort_index]

    opt = F1Optimizer.maximize_expectation(P2)

    best_prediction = ['None'] if opt[1] else []
    best_prediction += list(L2[:opt[0]])

    best = ' '.join(map(lambda x: str(x), best_prediction))
    df = df[0:1]
    df.loc[:, 'products'] = best
    return df

def expected_f1_beta(scores, a, b, n_samples, n_samples_tcdis):
    '''
        assume $thres \sim Beta(\alpha, \beta)$
    '''

    # sample thres from Beta
    thres_sample = beta(a, b, n_samples)
    # thres_sample = np.arange(0.1, 1, 0.05)

    n_samples = len(thres_sample)
    
    max_f1 = 0
    argmax_thres = None
    for i in range(n_samples):
        pred_thres = np.array(scores)
        pred_thres[pred_thres >= thres_sample[i]] = 1
        pred_thres[pred_thres < thres_sample[i]] = 0
        
        # target's conditional distribution
        # n_samples_tcdis = len(scores) # n_samples for p(t|s)
        target_sample = binomial(1, scores, (n_samples_tcdis, 1, len(scores)))
    
        sum_f1_target = 0 
        for j in range(n_samples_tcdis):
            sum_f1_target += f1_score(target_sample[j][0], pred_thres) 

        if (sum_f1_target / n_samples_tcdis >=  max_f1):
            max_f1 = sum_f1_target
            argmax_thres = thres_sample[i]   
    return argmax_thres

def wrap_expected_f1_beta(x, a, b, n_samples, n_samples_tcdis):
    if isinstance(x, pd.DataFrame):
        # print(x.iloc[0]['user_id'])
        return pd.DataFrame({'user_id':x.iloc[0]['user_id'],
                             'expected_f1':expected_f1_beta(x.iloc[0]['score'], a, b, n_samples, n_samples_tcdis)},
                            index = np.arange(1))
    else:
        # print(x['user_id'])
        return expected_f1_beta(x['score'], a, b, n_samples, n_samples_tcdis)

def approx_expected_f1(u_scores, args, mp_args):
    # 5mins 56 cpu
    u_f1 = mp_groupby(u_scores, 
                      ['user_id'],
                      wrap_expected_f1_beta,
                      *args,
                      **mp_args).reset_index()
    u_scores = pd.merge(u_scores, u_f1[['user_id', 'expected_f1']], how = 'left', on = ['user_id'])
    return u_scores

def wrap_f1_optim(x, is_pNone_given = False):
    '''
        wrap Fraon's f1optim for mp_groupby
    '''
    if is_pNone_given:
        best_k, pred_none, max_f1 = F1Optimizer.maximize_expectation(x.iloc[0]['score'], x.iloc[0]['none_score'])
    else:
        best_k, pred_none, max_f1 = F1Optimizer.maximize_expectation(x.iloc[0]['score'], None)
    return pd.DataFrame({'user_id': x.iloc[0]['user_id'],
                         'best_k': best_k,
                         'num_up': len(x.iloc[0]['score']),
                             # 'max_f1': max_f1,
                         'thres': x.iloc[0]['score'][best_k - 1], # select 0, ... , k-1
                         'pred_none': pred_none}, index = np.arange(1))

def optim_f1_k(u_scores, args, mp_args):
    u_f1 = mp_groupby(u_scores, ['user_id'], wrap_f1_optim, *args, **mp_args).reset_index()
    u_scores = pd.merge(u_scores, u_f1, on = ['user_id'], how = 'left')
    del u_scores['index']
    return u_scores

def f1_fast_search(prob, dtype=np.float32):
    size = len(prob)
    fk = np.zeros((size + 1), dtype=dtype)
    C = np.zeros((size + 1, size + 1), dtype=dtype)
    S = np.empty((2 * size + 1), dtype=dtype)
    S[:] = np.nan
    for k in range(1, 2 * size + 1):
        S[k] = 1./k
    roots = (prob - 1.0) / prob
    for k in range(size, 0, -1):
        poly = np.poly1d(roots[0:k], True)
        factor = np.multiply.reduce(prob[0:k])
        C[k, 0:k+1] = poly.coeffs[::-1]*factor
        for k1 in range(size + 1):
            fk[k] += (1. + 1.) * k1 * C[k, k1]*S[k + k1]
        for i in range(1, 2*(k-1)):
            S[i] = (1. - prob[k-1])*S[i] + prob[k-1]*S[i+1]
    return fk
########## EVALUATION UTILS


def cal_rmse(y_pred, y_gold):
    return sqrt(mean_squared_error(y_gold, y_pred))
    
def cal_f1score_for_sets(gold, pred):
    '''
    2 lists f1score
    '''
    gold = pd.DataFrame({'pid':gold, 'gold':1}, index = range(len(gold)))
    pred = pd.DataFrame({'pid':pred, 'pred':1}, index = range(len(pred)))
    res = pd.merge(gold, pred, on=['pid'], how='outer').fillna(0)
    return f1_score(y_true=res.gold, y_pred=res.pred)

def cal_precision_for_sets(gold, pred):
    '''
    2 lists f1score
    '''
    gold = pd.DataFrame({'pid':gold, 'gold':1}, index = range(len(gold)))
    pred = pd.DataFrame({'pid':pred, 'pred':1}, index = range(len(pred)))
    res = pd.merge(gold, pred, on=['pid'], how='outer').fillna(0)
    return precision_score(y_true=res.gold, y_pred=res.pred)

def cal_recall_for_sets(gold, pred):
    '''
    2 lists f1score
    '''
    gold = pd.DataFrame({'pid':gold, 'gold':1}, index = range(len(gold)))
    pred = pd.DataFrame({'pid':pred, 'pred':1}, index = range(len(pred)))
    res = pd.merge(gold, pred, on=['pid'], how='outer').fillna(0)
    return recall_score(y_true=res.gold, y_pred=res.pred)

def wrap_cal_f1(subf):
    return cal_f1score_for_sets(subf['gold_reorder'], subf['pred_reorder'])

def wrap_cal_precision(subf):
    return cal_precision_for_sets(subf['gold_reorder'], subf['pred_reorder'])

def wrap_cal_recall(subf):
    return cal_recall_for_sets(subf['gold_reorder'], subf['pred_reorder'])

############ Checkpoint utils

def cache_res(gid, label, score, prefix, train_or_test):
    res = gid.copy()
    if train_or_test == 'train':
        res[prefix + '_label'] = label
    res[prefix + '_score'] = score
    with open(constants.FEAT_DATA_DIR + prefix+'_res_%s.pkl'%train_or_test, 'wb') as f:
        print(constants.FEAT_DATA_DIR + prefix+'_res_%s.pkl'%train_or_test)
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
    return res

############ Submission utils

def submission_format(subf):
    if isinstance(subf, list):
        subf = [str(i) for i in subf]
        subf = ' '.join(subf)
    return subf


def applyParallel(dfGrouped, func, func_args = None):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count() // 2)(delayed(func)(group, func_args) for name, group in dfGrouped)
    return pd.concat(retLst)
