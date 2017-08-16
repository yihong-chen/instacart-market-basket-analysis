from utils import wrap_cal_f1, wrap_cal_precision, wrap_cal_recall

import pickle

import numpy as np
import pandas as pd

import tensorflow as tf
import edward as ed

import pdb

from edward.models import Bernoulli, Beta
from sklearn.metrics import f1_score

def get_gold(user_product):
    '''
    user_product:(u, p)
    get gold: get products into lists which users reorder 
    '''
    gold = user_product[user_product.label == 1]\
           .groupby(['order_id'])['product_id']\
           .apply(list).reset_index()
            
    return gold

def evaluation(gold, pred):
    res = pd.merge(gold, pred, 
                    on = ['order_id'], 
                    how = 'outer')
    res.columns = ['order_id', 'gold_reorder', 'pred_reorder']
    for row in res.loc[res.gold_reorder.isnull(), 'gold_reorder'].index:
        res.at[row, 'gold_reorder'] = ['None']
    for row in res.loc[res.pred_reorder.isnull(), 'pred_reorder'].index:
        res.at[row, 'pred_reorder'] = ['None']
    res['f1score'] = res.apply(wrap_cal_f1, axis = 1)
    # res['precision'] = res.apply(wrap_cal_precision, axis = 1)
    # res['recall'] = res.apply(wrap_cal_recall, axis = 1)
    return res 

