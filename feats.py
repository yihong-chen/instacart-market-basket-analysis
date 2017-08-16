'''
    train set
'''
import os
import numpy as np
import pandas as pd
import constants, transactions

def make_train(train_or_test):
    if os.path.exists(constants.FEAT_DATA_DIR + '{}.h5'.format(train_or_test)):
        train = pd.read_hdf(constants.FEAT_DATA_DIR + '{}.h5'.format(train_or_test))
        tle = transactions.TransLogExtractor(constants.RAW_DATA_DIR, constants.FEAT_DATA_DIR)
    else:
        tle = transactions.TransLogExtractor(constants.RAW_DATA_DIR, constants.FEAT_DATA_DIR)
        users_orders = tle.get_users_orders(prior_or_train='prior')
        train = users_orders[['user_id', 'product_id']].drop_duplicates()
        # label 
        if train_or_test == 'train':
            print('getting labels !')
            label = tle.craft_label()
            train = pd.merge(train, label, on=['user_id', 'product_id'], how='left')
            train.label.fillna(0, inplace=True)
            del label
        
        # context feat, check original version !!!!
        print('geting context feat !')
        ctx = tle.craft_context(train_or_test)
        train = pd.merge(train, ctx, on=['user_id'], how='right') # keep only train orders
        del ctx
        # user feat
        print('getting user feat !')
        usr = tle.craft_feat_user()
        train = pd.merge(train, usr, on=['user_id'], how='left')
        del usr

        train['u_ct_mean_interval'] = train['ct_days_since_prior_order'] / train['u_mean_interval']
        train['u_ct_mean_interval'].replace(np.inf, 20, inplace=True)
        train['u_ct_mean_interval'].fillna(1, inplace=True)
        
        train['u_ct_hod'] = abs(train['ct_order_hour_of_day'] - train['u_hod_argmost']).map(lambda x: min(x, 24-x)).astype(np.int8)
        train['u_ct_dow'] = abs(train['ct_order_dow'] - train['u_dow_argmost']).map(lambda x: min(x, 7-x)).astype(np.int8)
        # del train['ct_days_since_prior_order'] # same as u_active_last

        # pad feat
        print('getting pad feat !')
        pad = tle.craft_feat_pad()
        train = pd.merge(train, pad, on=['product_id'], how='left')

        # interaction feat
        print('getting interact feat, user vs product !')
        upi = tle.craft_feat_interact('product_id')
        train = pd.merge(train, upi, on=['user_id', 'product_id'], how='left')
        del upi

        train['up_ct_hod_argmost'] = abs(train['up_hod_argmost'] - train['ct_order_hour_of_day']).map(lambda x: min(x, 24-x)).astype(np.int8)
        train['up_u_hod_argmost'] = abs(train['up_hod_argmost'] - train['u_hod_argmost']).map(lambda x: min(x, 24-x)).astype(np.int8)

        train['up_ct_dow_argmost'] = abs(train['up_dow_argmost'] - train['ct_order_dow']).map(lambda x: min(x, 7-x)).astype(np.int8)
        train['up_u_dow_argmost'] = abs(train['up_dow_argmost'] - train['u_dow_argmost']).map(lambda x: min(x, 7-x)).astype(np.int8)

        train['up_ct_last_hod'] = abs(train['up_last_hod'] - train['ct_order_hour_of_day']).map(lambda x: min(x, 24-x)).astype(np.int8)
        train['up_u_last_hod'] = abs(train['up_last_hod'] - train['u_hod_argmost']).map(lambda x: min(x, 24-x)).astype(np.int8)

        train['up_ct_last_dow'] = abs(train['up_last_dow'] - train['ct_order_dow']).map(lambda x: min(x, 7-x)).astype(np.int8)
        train['up_u_last_dow'] = abs(train['up_last_dow'] - train['u_dow_argmost']).map(lambda x: min(x, 7-x)).astype(np.int8)
        
        train['up_ct_active_last'] = train['up_days_to_last'] / train['ct_days_since_prior_order']
        train['up_ct_active_last'].replace(np.inf, 400, inplace=True)
        train['up_ct_active_last'].fillna(1, inplace=True)

        train['up_ct_avg_interval'] = train['up_avg_interval'] / train['ct_days_since_prior_order']
        train['up_ct_avg_interval'].replace(np.inf, 400, inplace=True)
        train['up_ct_avg_interval'].fillna(1, inplace=True)
        
        train['up_u_avg_interval'] = train['up_avg_interval'] / train['u_p_avg_interval']
        train['up_u_avg_interval'].replace(np.inf, 20, inplace=True)
        train['up_u_avg_interval'].fillna(1, inplace=True)
        
        train['up_u_med_interval'] = train['up_median_interval'] / train['u_p_med_interval']
        train['up_u_med_interval'].replace(np.inf, 40, inplace=True)
        train['up_u_med_interval'].fillna(1, inplace=True)
        
        train['up_last_interval'] = train['up_days_to_last'] / train['up_avg_interval']
        train['up_last_interval'].replace(np.inf, 300, inplace=True)
        train['up_last_interval'].fillna(1, inplace=True)
        
        print('getting interact feat user vs aisle !')
        uai = tle.craft_feat_interact('aisle_id')
        train = pd.merge(train, uai, on=['user_id', 'aisle_id'], how='left')
        del uai

        train['ua_ct_hod_argmost'] = abs(train['ua_hod_argmost'] - train['ct_order_hour_of_day']).map(lambda x: min(x, 24-x)).astype(np.int8)
        train['ua_u_hod_argmost'] = abs(train['ua_hod_argmost'] - train['u_hod_argmost']).map(lambda x: min(x, 24-x)).astype(np.int8)

        train['ua_ct_dow_argmost'] = abs(train['ua_dow_argmost'] - train['ct_order_dow']).map(lambda x: min(x, 7-x)).astype(np.int8)
        train['ua_u_dow_argmost'] = abs(train['ua_dow_argmost'] - train['u_dow_argmost']).map(lambda x: min(x, 7-x)).astype(np.int8)

        train['ua_ct_last_hod'] = abs(train['ua_last_hod'] - train['ct_order_hour_of_day']).map(lambda x: min(x, 24-x)).astype(np.int8)
        train['ua_u_last_hod'] = abs(train['ua_last_hod'] - train['u_hod_argmost']).map(lambda x: min(x, 24-x)).astype(np.int8)

        train['ua_ct_last_dow'] = abs(train['ua_last_dow'] - train['ct_order_dow']).map(lambda x: min(x, 7-x)).astype(np.int8)
        train['ua_u_last_dow'] = abs(train['ua_last_dow'] - train['u_dow_argmost']).map(lambda x: min(x, 7-x)).astype(np.int8)
        
        train['ua_ct_active_last'] = train['ua_days_to_last'] / train['ct_days_since_prior_order']
        train['ua_ct_active_last'].replace(np.inf, 300, inplace=True)
        train['ua_ct_active_last'].fillna(1, inplace=True)
        
        train['ua_ct_avg_interval'] = train['ua_avg_interval'] / train['ct_days_since_prior_order']
        train['ua_ct_avg_interval'].replace(np.inf, 200, inplace=True)
        train['ua_ct_avg_interval'].fillna(1, inplace=True)
        
        train['ua_u_avg_interval'] = train['ua_avg_interval'] / train['u_a_avg_interval']
        train['ua_u_avg_interval'].replace(np.inf, 20, inplace=True)
        train['ua_u_avg_interval'].fillna(1, inplace=True)
        
        train['ua_u_med_interval'] = train['ua_median_interval'] / train['u_a_med_interval']
        train['ua_u_med_interval'].replace(np.inf, 40, inplace=True)
        train['ua_u_med_interval'].fillna(1, inplace=True)
        
        train['ua_last_interval'] = train['ua_days_to_last'] / train['ua_avg_interval']
        train['ua_last_interval'].replace(np.inf, 200, inplace=True)
        train['ua_last_interval'].fillna(1, inplace=True)
        
        print('getting interact feat user vs department !')
        udi = tle.craft_feat_interact('department_id')
        train = pd.merge(train, udi, on=['user_id', 'department_id'], how='left')
        del udi
        
        print('getting ud_ct_hod_argmost !')
        train['ud_ct_hod_argmost'] = abs(train['ud_hod_argmost'] - train['ct_order_hour_of_day']).map(lambda x: min(x, 24-x)).astype(np.int8)
        train['ud_u_hod_argmost'] = abs(train['ud_hod_argmost'] - train['u_hod_argmost']).map(lambda x: min(x, 24-x)).astype(np.int8)
        
        print('getting ud_ct_dow_argmost !')
        train['ud_ct_dow_argmost'] = abs(train['ud_dow_argmost'] - train['ct_order_dow']).map(lambda x: min(x, 7-x)).astype(np.int8)
        train['ud_u_dow_argmost'] = abs(train['ud_dow_argmost'] - train['u_dow_argmost']).map(lambda x: min(x, 7-x)).astype(np.int8)
        
        print('getting ud_ct_last_hod !')
        train['ud_ct_last_hod'] = abs(train['ud_last_hod'] - train['ct_order_hour_of_day']).map(lambda x: min(x, 24-x)).astype(np.int8)
        train['ud_u_last_hod'] = abs(train['ud_last_hod'] - train['u_hod_argmost']).map(lambda x: min(x, 24-x)).astype(np.int8)
        
        print('getting ud_ct_last_dow !')
        train['ud_ct_last_dow'] = abs(train['ud_last_dow'] - train['ct_order_dow']).map(lambda x: min(x, 7-x)).astype(np.int8)
        train['ud_u_last_dow'] = abs(train['ud_last_dow'] - train['u_dow_argmost']).map(lambda x: min(x, 7-x)).astype(np.int8)
        
        print('getting ud_ct_active_last !')
        train['ud_ct_active_last'] = train['ud_days_to_last'] / train['ct_days_since_prior_order']
        train['ud_ct_active_last'].replace(np.inf, 300, inplace=True)
        train['ud_ct_active_last'].fillna(1, inplace=True)

        print('getting ud_ct_avg_interval !')
        train['ud_ct_avg_interval'] = train['ud_avg_interval'] / train['ct_days_since_prior_order']
        train['ud_ct_avg_interval'].replace(np.inf, 200, inplace=True)
        train['ud_ct_avg_interval'].fillna(1, inplace=True)
        
        train['ud_u_avg_interval'] = train['ud_avg_interval'] / train['u_d_avg_interval']
        train['ud_u_avg_interval'].replace(np.inf, 20, inplace=True)
        train['ud_u_avg_interval'].fillna(1, inplace=True)
        
        train['ud_u_med_interval'] = train['ud_median_interval'] / train['u_d_med_interval']
        train['ud_u_med_interval'].replace(np.inf, 40, inplace=True)
        train['ud_u_med_interval'].fillna(1, inplace=True)
        
        train['ud_last_interval'] = train['ud_days_to_last'] / train['ud_avg_interval']
        train['ud_last_interval'].replace(np.inf, 200, inplace=True)
        train['ud_last_interval'].fillna(1, inplace=True)
        
        print('getting up_ua_reorder !')
        train['up_ua_reorder'] = train['up_reorder_num'] / train['ua_reorder_num']
        train['up_ua_reorder'].fillna(0, inplace=True)
        
        train['up_ud_reorder'] = train['up_reorder_num'] / train['ud_reorder_num']
        train['up_ud_reorder'].fillna(0, inplace=True)
        
        train['ua_ud_reorder'] = train['ua_reorder_num'] / train['ud_reorder_num']
        train['ua_ud_reorder'].fillna(0, inplace=True)

        train['up_ua_order'] = train['up_order_num'] / train['ua_order_num']
        train['up_ua_order'].fillna(0, inplace=True)
        train['up_ud_order'] = train['up_order_num'] / train['ud_order_num']
        train['up_ud_order'].fillna(0, inplace=True)
        train['ua_ud_order'] = train['ua_order_num'] / train['ud_order_num']
        train['ua_ud_order'].fillna(0, inplace=True)
        
        print('getting up_ua_avg_add2cart_order !')
        train['up_ua_avg_add2cart'] = train['up_avg_add2cart_order'] / train['ua_avg_add2cart_order']
        train['up_ud_avg_add2cart'] = train['up_avg_add2cart_order'] / train['ud_avg_add2cart_order']
        train['ua_ud_avg_add2cart'] = train['ua_avg_add2cart_order'] / train['ud_avg_add2cart_order']
      
        train['up_ua_std_interval'] = train['up_std_interval'] / train['ua_std_interval']
        train['up_ua_std_interval'].replace(np.inf, 50, inplace=True)
        train['up_ua_std_interval'].fillna(1, inplace=True)
        
        train['up_ud_std_interval'] = train['up_std_interval'] / train['ud_std_interval']
        train['up_ud_std_interval'].replace(np.inf, 80, inplace=True)
        train['up_ud_std_interval'].fillna(1, inplace=True)
        
        train['ua_ud_std_interval'] = train['ua_std_interval'] / train['ud_std_interval']
        train['ua_ud_std_interval'].replace(np.inf, 50, inplace=True)
        train['ua_ud_std_interval'].fillna(1, inplace=True)
        
        # order streak
        print('getting order streak feat !')
        stk = tle.craft_order_streak()
        train = pd.merge(train, stk, on=['user_id', 'product_id'], how='left')
        del stk
        train['order_streak'].fillna(-5, inplace=True)

        #######Automatic Feat
        # 1 word2vec feat
        print('getting word2vec feat !')
        p_w2v = tle.craft_p_w2v()
        train = pd.merge(train, p_w2v, on='product_id', how='left')
        del p_w2v
        
        # 2 lda feat
        print('getting topic feat !')
        # 2.1 sklearn 22 u_topic, a_topic dis
        # 2.2 gensim 10, 10, 4 pad_p,pad_p_u dis
        up_topic_pc = tle.craft_topic_pc()
        train = pd.merge(train, up_topic_pc, on=['user_id', 'product_id'], how='left')
        del up_topic_pc

        up_topic_dis = tle.craft_topic_dis()
        train = pd.merge(train, up_topic_dis, on=['user_id', 'product_id'], how='left')
        del up_topic_dis

        # 3 dream feat
        
        # 3.0 original dream score
        dream_feat = tle.craft_dream_final()
        train = pd.merge(train, dream_feat, on=['user_id', 'product_id'], how='left')
        del dream_feat
        
        # 3.1 dream score next
        # 3.2 dream score dynamic user
        # 3.3 drem score item embedding
        
        # 3.4 reorder dream score
        # 3.5 reorder dream dynamic user
        # 3.6 reorder dream item embedding
        reorder_dream_feat = tle.craft_dream_final(is_reordered=True)
        train = pd.merge(train, reorder_dream_feat, on=['user_id', 'product_id'], how='left')
        del reorder_dream_feat
        
        # 4 lstm interval feat
        lstm_interval = tle.craft_up_interval()
        train = pd.merge(train, lstm_interval, on=['user_id', 'product_id'], how='left')
        del lstm_interval
        train['up_delta'] = train.up_delta.fillna(train.up_avg_interval_m - train.up_days_to_last)
        train['up_abs_delta'] = train.up_abs_delta.fillna(train.up_delta.apply(abs))

        train.to_hdf(constants.FEAT_DATA_DIR + '{}.h5'.format(train_or_test), train_or_test, mode = 'w', complevel = 3)
    return train

# groups of features

