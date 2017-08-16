from constants import FEAT_DATA_DIR

import pickle

import numpy as np
import pandas as pd

def topk_pairs(subf, num_std, num_mean):
    
    order_id = subf.head(1)['order_id']
    u_reorder_pnum_mean = subf.head(1)['reorder_pnum_mean']
    u_reorder_pnum_std = subf.head(1)['reorder_pnum_std']
    n_neighbors = int(num_mean * np.floor(u_reorder_pnum_mean.values[0])) + \
                  int(num_std * np.floor(u_reorder_pnum_std.values[0]))
    if n_neighbors > len(subf):
        n_neighbors = len(subf)
    if n_neighbors > 0:
        neighbors = list(subf['product_id'].head(n_neighbors))
    else:
        neighbors = 'None'
    return neighbors
    
    
def nn_search(ups, num_std, num_mean):
    '''
    nearest neighbor search 
    define distance as -log(score)
    ups:user_id, product_id, order_id, score(loyalty)
    '''
    with open(FEAT_DATA_DIR + 'user_reorder_est.pkl', 'rb') as f:
        avg_reorder_est = pickle.load(f)
    
    ups = pd.merge(ups,
                   avg_reorder_est[['user_id', 'reorder_pnum_mean', 'reorder_pnum_std']],
                   on = ['user_id'],
                   how = 'left').sort_values(['user_id', 'score'], ascending = False)
    pred = ups.groupby(['order_id']).apply(lambda x: topk_pairs(x, num_std, num_mean)).reset_index()
    return pred
  
def raw_search(user_product):
    '''
    user_product:(u, p)
    prediction:threshold (u, p) score 
    '''
    pred = user_product.groupby(['order_id'])['product_id']\
           .apply(list).reset_index()
    return pred

def quantile_mask(row, quantile):
    score = np.array(row['score'])
    product_id = np.array(row['product_id'])
    return pd.Series({'pred_reorder':product_id[score > np.percentile(score, quantile)].tolist(),'order_id':row['order_id']})
        
def quantile_search(user_product, quantile):

    user_product = user_product.sort_values(['order_id', 'score'], ascending=False)
    user_order = user_product.groupby(['order_id']).agg({'product_id':lambda x:list(x),
                                           'label':lambda x: list(x), 
                                           'score':lambda x: list(x)}).reset_index()
    pred = user_order.apply(lambda x: quantile_mask(x, quantile), axis = 1)
    return pred

def none_fill(row, thres):
    if row['none_score'] > thres:
        if row['pred_reorder'] != ['None']:
            if not 'None' in row: # for 1st row
                row['pred_reorder'].append('None')
    return row

def f1_none_fill(row):
    if row['pred_none'] is True:
        if row['pred_reorder'] != ['None']:
            if not 'None' in row: # for 1st row
                row['pred_reorder'].append('None')
    return row

def none_rep(row, thres):
    ''' replace prediction with ['None'] if none_score > thres'''
    if row['none_score'] > thres:
        row['pred_reorder'] = ['None']
    return row

######## order based inference

def order_based_search(user_product):
    pred = user_product[user_product['score'] >= user_product['thres']].groupby(['order_id'])['product_id'].apply(list).reset_index()
    return pred

