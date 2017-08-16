import gc
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.spatial.distance import euclidean

from constants import NUM_TOPIC
# from utils import is_organic, flatten_multiidx

import pdb

class TransLogConstructor:
    def __init__(self, raw_data_dir, cache_dir):
        self.raw_data_dir = raw_data_dir
        self.cache_dir = cache_dir

    def clear_cache(self):
        for root, dirs, files in os.walk(self.raw_data_dir):
            for name in files:
                if name.endswith(".h5"):
                    os.remove(os.path.join(root, name))
                    print("Delete %s"%os.path.join(root, name))
        print("Clear all cached h5!")

    def get_orders(self):
        '''
            get order context information
        '''
        if os.path.exists(self.raw_data_dir + 'orders.h5'):
            orders = pd.read_hdf(self.raw_data_dir + 'orders.h5')
        else:
            orders = pd.read_csv(self.raw_data_dir + 'orders.csv',
                dtype = {'order_id': np.int32,
                         'user_id': np.int32,
                         'eval_set': 'category',
                         'order_number': np.int16,
                         'order_dow': np.int8,
                         'order_hour_of_day' : np.int8,
                         'days_since_prior_order': np.float32})

            orders['days_since_prior_order'] = orders['days_since_prior_order'].fillna(0.0)
            orders['days'] = orders.groupby(['user_id'])['days_since_prior_order'].cumsum()
            orders['days_last'] = orders.groupby(['user_id'])['days'].transform(max)
            orders['days_up_to_last'] = orders['days_last'] - orders['days']
            del orders['days_last']
            del orders['days']
            orders.to_hdf(self.raw_data_dir + 'orders.h5', 'orders', mode = 'w', format = 'table')
        return orders
    
    def get_orders_items(self, prior_or_train):
        '''
            get detailed information of prior or train orders 
        '''
        if os.path.exists(self.raw_data_dir + 'order_products__%s.h5'%prior_or_train):
            order_products = pd.read_hdf(self.raw_data_dir + 'order_products__%s.h5'%prior_or_train)
        else:
            order_products = pd.read_csv(self.raw_data_dir + 'order_products__%s.csv'%prior_or_train,
                dtype = {'order_id': np.int32,
                         'product_id': np.uint16,
                         'add_to_cart_order': np.int16,
                         'reordered': np.int8})
            order_products.to_hdf(self.raw_data_dir + 'order_products__%s.h5'%prior_or_train, 'op', mode = 'w', format = 'table')
        return order_products
    
    def get_users_orders(self, prior_or_train, pad = 'product_id'):
        '''
            get users' detailed orders
            oid, uid, pid, aid, did, reordered, days_since_prior_order, days_up_to_last, 
            hod, dow, pad[0]_purchase_times, pad[0]_purchase_interval
        '''
        if os.path.exists(self.raw_data_dir + 'user_orders_%s_%s.h5'%(prior_or_train, pad[:-3])):
            user_orders = pd.read_hdf(self.raw_data_dir + 'user_orders_%s_%s.h5'%(prior_or_train, pad[:-3]))
        else:
            orders = self.get_orders()
            del orders['eval_set']
            order_items = self.get_orders_items(prior_or_train)
            products = self.get_items('products')[['product_id', 'aisle_id', 'department_id']]

            user_orders = pd.merge(order_items, orders, on = ['order_id'], how = 'left')
            user_orders = pd.merge(user_orders, products, on = ['product_id'], how = 'left')
            del order_items, products, orders

            if prior_or_train == 'prior':
                prefix = pad[0] + '_'
                user_orders[prefix + 'purchase_times'] = (user_orders.sort_values(['user_id', pad, 'order_number'])
                                                            .groupby(['user_id', pad]).cumcount()+1)
                user_orders[prefix + 'purchase_interval'] = (user_orders.sort_values(['user_id', pad, 'order_number'], ascending = False)
                                                            .groupby(['user_id', pad])['days_up_to_last'].diff())
                user_orders[prefix + 'purchase_interval'] = user_orders[prefix + 'purchase_interval'].fillna(-1) # 1st time purchase
            user_orders.to_hdf(self.raw_data_dir + 'user_orders_%s_%s.h5'%(prior_or_train, pad[:-3]), 'user_orders', mode = 'w')
        return user_orders
    
    def get_items(self, gran):
        '''
            get items' information
            gran = [departments, aisles, products]
        '''
        items = pd.read_csv(self.raw_data_dir + '%s.csv'%gran)
        return items

class TransLogExtractor(TransLogConstructor):
    def __init__(self, raw_data_dir, cache_dir):
        super().__init__(raw_data_dir, cache_dir)
    
    def clear_cache(self, include_raw = False):
        if include_raw:
            super().clear_cache()
        for root, dirs, files in os.walk(self.cache_dir):
            for name in files:
                if name.endswith("_feat.pkl") or name.endswith('_feat.h5'):
                    os.remove(os.path.join(root, name))
                    print("Delete %s"%os.path.join(root, name))
                if name == 'train.h5' or name == 'test.h5':
                    os.remove(os.path.join(root, name))
                    print("Delete %s"%os.path.join(root, name))
        print("Clear all cached !")

    def cal_first_second(self, user_orders, pad, gcol):

        prefix = pad[0] + '_'
        is_user = 'u_' if gcol == 'user_id' else ''

        first_purchase = (user_orders[user_orders[prefix + 'purchase_times'] == 1].groupby(gcol)[prefix + 'purchase_times']
                                     .aggregate({is_user + prefix + 'first_times': 'count'}).reset_index())
        second_purchase = (user_orders[user_orders[prefix + 'purchase_times'] == 2].groupby(gcol)[prefix + 'purchase_times']
                                     .aggregate({is_user + prefix + 'second_times': 'count'}).reset_index())
        first_second = pd.merge(first_purchase, second_purchase, on = gcol, how = 'left')
        first_second[is_user + prefix + 'second_times'] = first_second[is_user + prefix + 'second_times'].fillna(0)
        first_second[is_user + prefix + 'reorder_prob'] = first_second[is_user + prefix + 'second_times'] / first_second[is_user + prefix + 'first_times']
        del user_orders
        return first_second

    def cal_dow_hod(self, user_orders, prefix, gcol):

            dow = user_orders.groupby(gcol)['order_dow'].value_counts().unstack(fill_value = 0.0)
            dow_entropy = dow.apply(lambda x: entropy(x.values, np.ones(len(x))), axis = 1).rename(prefix + 'dow_entropy').reset_index()
            dow_most = dow.apply(lambda x: max(x.values), axis = 1).rename(prefix + 'dow_most').reset_index()
            dow_argmost = dow.apply(lambda x: np.argmax(x.values), axis = 1).rename(prefix + 'dow_argmost').reset_index()
            dow = dow_entropy.merge(dow_most, on = gcol, how = 'left')
            dow = dow.merge(dow_argmost, on = gcol, how = 'left')

            hod = user_orders.groupby(gcol)['order_hour_of_day'].value_counts().unstack(fill_value = 0.0)
            hod_entropy = hod.apply(lambda x: entropy(x.values, np.ones(len(x))), axis = 1).rename(prefix + 'hod_entropy').reset_index()
            hod_most = hod.apply(lambda x: max(x.values), axis = 1).rename(prefix + 'hod_most').reset_index()
            hod_argmost = hod.apply(lambda x: np.argmax(x.values), axis = 1).rename(prefix + 'hod_argmost').reset_index()
            hod = hod_entropy.merge(hod_most, on = gcol, how = 'left')
            hod = hod.merge(hod_argmost, on = gcol, how = 'left')
            dow_hod = dow.merge(hod, on = gcol, how = 'left')
            del user_orders
            return dow_hod

    def cal_pad_agg(self, user_orders, prefix, pad, agg_col, agg_ops):
        ''' user feat'''
        mid = pad[0] + '_'
        suffix = agg_col[10:]
        pad_agg = (user_orders.groupby(['user_id', pad])[agg_col].aggregate({agg_col: agg_ops}).reset_index()
                              .groupby(['user_id'])[agg_col].aggregate({
                                      prefix + mid + 'avg' + suffix: 'mean',
                                      prefix + mid + 'std' + suffix: 'std',
                                      prefix + mid + 'min' + suffix: 'min',
                                      prefix + mid + 'max' + suffix: 'max',
                                      prefix + mid + 'med' + suffix: 'median'}).reset_index())
        del user_orders
        return pad_agg

    def craft_label_none(self):
        if os.path.exists(self.cache_dir + 'label_none.pkl'):
            with open(self.cache_dir + 'label_none.pkl', 'rb') as f:
                label_none = pickle.load(f)
        else:
            user_product = self.get_users_orders('train')
            o_is_none = user_product.groupby(['order_id']).agg({'reordered':{'o_reordered_num':sum}})#.reset_index()
            o_is_none.columns = o_is_none.columns.droplevel(0)
            o_is_none.reset_index(inplace=True)
            user_product = pd.merge(user_product, 
                                    o_is_none, 
                                    on = ['order_id'], 
                                    how = 'left')
            user_product['label_none'] = user_product['o_reordered_num'].apply(lambda x : int(x == 0))
            label_none = user_product[['user_id', 'order_id', 'label_none']].drop_duplicates()
            with open(self.cache_dir + 'label_none.pkl', 'wb') as f:
                pickle.dump(label_none, f, pickle.HIGHEST_PROTOCOL)       
        return label_none 

    def craft_label(self):
        if os.path.exists(self.cache_dir + 'label.pkl'):
            with open(self.cache_dir + 'label.pkl', 'rb') as f:
                label = pickle.load(f)
        else:
            # orders = self.get_orders()
            # order_products_train = self.get_orders_items('train')
            # user_product = pd.merge(order_products_train, orders, on = ['order_id'], how = 'left')
            user_product = self.get_users_orders('train')
            label = user_product[user_product.reordered == 1][['user_id', 'product_id', 'reordered']]
            label.columns = ['user_id', 'product_id', 'label']
            with open(self.cache_dir + 'label.pkl', 'wb') as f:
                pickle.dump(label, f, pickle.HIGHEST_PROTOCOL)       
        return label  

    def craft_context(self, train_or_test):
        '''
            train_or_test = ['train', 'test']
        '''
        if os.path.exists(self.cache_dir + 'context_feat_%s.pkl'%train_or_test):
            with open(self.cache_dir + 'context_feat_%s.pkl'%train_or_test, 'rb') as f:
                context_feat = pickle.load(f)
        else:
            orders = self.get_orders()
            orders = orders[orders.eval_set == train_or_test]
            context_feat = orders[['order_id', 'user_id', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
            context_feat.columns = ['order_id', 'user_id', 'ct_order_dow', 'ct_order_hour_of_day', 'ct_days_since_prior_order']
            with open(self.cache_dir + 'context_feat_%s.pkl'%train_or_test, 'wb') as f:
                pickle.dump(context_feat, f, pickle.HIGHEST_PROTOCOL)       
        return context_feat  

    def craft_feat_user(self):
        ''' all users feat'''
        if os.path.exists(self.cache_dir + 'user_feat.h5'):
            user_feat = pd.read_hdf(self.cache_dir + 'user_feat.h5')
        else:
            prefix = 'u_'
            dfs = [self.get_users_orders('prior', 'product_id'),
                   self.get_users_orders('prior', 'aisle_id')[['order_id', 'a_purchase_times', 'a_purchase_interval']],
                   self.get_users_orders('prior', 'department_id')[['order_id', 'd_purchase_times', 'd_purchase_interval']]]
            dfs =[df.set_index('order_id', drop=True)for df in dfs]
            user_orders = pd.concat(dfs, axis=1, join='outer', copy=False)
            user_orders.reset_index(drop=False, inplace=True)
            del dfs

            grouped = user_orders.groupby(['user_id']).agg({
                'order_number' : {'u_total_orders' : max}, 
                'reordered' : {'u_total_reorders' : sum,
                               'u_reorder_ratio':'mean'},
                'product_id' : {'u_total_prods' : pd.Series.nunique},
                'aisle_id':{prefix + 'total_aisles': pd.Series.nunique},
                'department_id':{prefix + 'total_deps':pd.Series.nunique},
                'days_up_to_last': {'u_active_first' : max,
                                    'u_active_last': min},
                'add_to_cart_order':{ 'u_min_add2cart_order': min,
                                      'u_max_add2cart_order': max,
                                      'u_avg_add2cart_order':'mean', 
                                      'u_std_add2cart_order':'std',
                                      'u_med_add2cart_order':'median'}})#.reset_index()
            grouped.columns = grouped.columns.droplevel(0)
            grouped.reset_index(inplace = True)
            # grouped = flatten_multiidx(grouped)
            grouped['u_active_last_30'] = grouped['u_active_last'] % 30
            grouped['u_active_last_21'] = grouped['u_active_last'] % 21
            grouped['u_active_last_14'] = grouped['u_active_last'] % 14
            grouped['u_active_last_7'] = grouped['u_active_last'] % 7

            grouped['u_active_period'] = grouped['u_active_first'] - grouped['u_active_last']
            grouped['u_avg_reorders'] = grouped['u_total_reorders'] / grouped['u_total_orders']
            grouped['u_mean_interval'] = grouped['u_active_period'] / grouped['u_total_orders']
            grouped['u_mean_basket'] = grouped['u_total_prods'] / grouped['u_total_orders']
            # grouped['u_al_vs_mi'] = grouped['u_active_last'] / grouped['u_mean_interval']

            for pad in ['product_id', 'aisle_id', 'department_id']:
                agg_col = pad[0] + '_' + 'purchase_times' # p purchase_times, a_purchase_times, d_purchase_times
                pad_agg = self.cal_pad_agg(user_orders, prefix, pad, agg_col, 'max')
                grouped = grouped.merge(pad_agg, on = 'user_id', how = 'left')
                del pad_agg

                agg_col = pad[0] + '_' + 'purchase_interval'
                pad_agg = self.cal_pad_agg(user_orders[(user_orders.p_purchase_interval != -1)], prefix, pad, agg_col, 'mean')
                grouped = grouped.merge(pad_agg, on = 'user_id', how = 'left')
                del pad_agg
            
            dow_hod = self.cal_dow_hod(user_orders, prefix, 'user_id')                                                          
            grouped = grouped.merge(dow_hod, on = ['user_id'], how = 'left')
            del dow_hod

            reorder_pnum = (user_orders[user_orders.reordered == 1] 
                       .groupby(['user_id', 'order_id'])['product_id'] 
                       .agg({'reorder_pnum':'count'}).reset_index()
                       .groupby(['user_id'])['reorder_pnum'] 
                       .agg({'u_reorder_pnum_mean':'mean', 'u_reorder_pnum_std':'std'}).reset_index())
            grouped =grouped.merge(reorder_pnum, on = ['user_id'], how = 'left')
            del reorder_pnum

            grouped = grouped.merge(self.cal_first_second(user_orders, 'product_id', 'user_id'), on = ['user_id'], how = 'left')
            grouped = grouped.merge(self.cal_first_second(user_orders, 'aisle_id', 'user_id'), on = ['user_id'], how = 'left')
            user_feat = grouped.merge(self.cal_first_second(user_orders, 'department_id', 'user_id'), on = ['user_id'], how = 'left')
            del grouped, user_orders           

            na_cols = ['u_p_avg_interval', 'u_p_med_interval', 'u_p_min_interval', 'u_p_max_interval',
                       'u_a_avg_interval', 'u_a_med_interval', 'u_a_min_interval', 'u_a_max_interval',
                       'u_d_avg_interval', 'u_d_med_interval', 'u_d_min_interval', 'u_d_max_interval']
            for col in na_cols:
                user_feat[col] = user_feat[col].fillna(user_feat['u_mean_interval'])
            na_cols = ['u_p_std_interval', 'u_a_std_interval', 'u_d_std_interval',
                       'u_p_std_times', 'u_a_std_times', 'u_d_std_times',
                       'u_reorder_pnum_std', 'u_reorder_pnum_mean']
            user_feat[na_cols] = user_feat[na_cols].fillna(0)

            user_feat.to_hdf(self.cache_dir + 'user_feat.h5', 'user', mode = 'w')       
        return user_feat 


    def craft_feat_item(self, pad):
        '''
            pad = [product_id, aisle_id, department_id]
        '''
        if os.path.exists(self.cache_dir + '%s_feat.h5'%pad[:-3]):
            item_feat = pd.read_hdf(self.cache_dir + '%s_feat.h5'%pad[:-3])
        else:
            prefix = pad[0] + '_'
            user_orders = self.get_users_orders('prior', pad)

            grouped = user_orders.groupby(pad).agg(
                {prefix + 'purchase_times':{prefix + 'max_times':max,
                                            prefix + 'min_times':min},
                 'user_id':{prefix + 'num_purchsers': pd.Series.nunique},
                 'reordered':{prefix + 'reorder_sum':sum, 
                              prefix + 'reorder_total':'count'},
                 'days_up_to_last':{prefix + 'days_to_last':min, 
                                    prefix + 'days_to_first':max},
                 'add_to_cart_order':{prefix + 'min_add2cart_order':min, 
                                      prefix + 'max_add2cart_order':max,
                                      prefix + 'avg_add2cart_order':'mean', 
                                      prefix + 'std_add2cart_order':'std',
                                      prefix + 'med_add2cart_order':'median'}})#.reset_index()
            grouped.columns = grouped.columns.droplevel(0)
            grouped.reset_index(inplace=True)
            # grouped = flatten_multiidx(grouped)
            grouped[prefix + 'std_add2cart_order'] = grouped[prefix + 'std_add2cart_order'].fillna(0)
            grouped[prefix + 'active_period'] = grouped[prefix + 'days_to_first'] - grouped[prefix + 'days_to_last']
            grouped[prefix + 'reorder_ratio'] = grouped[prefix + 'reorder_sum'] / grouped[prefix + 'reorder_total']
            
            first_second = self.cal_first_second(user_orders, pad, pad)
            grouped = grouped.merge(first_second, on = [pad], how = 'left')
            del first_second

            grouped[prefix + 'order_pp'] = grouped[prefix + 'reorder_total'] /grouped[prefix + 'first_times']
            grouped[prefix + 'reorder_pp'] = grouped[prefix + 'reorder_sum'] / grouped[prefix + 'first_times']

            dow_hod = self.cal_dow_hod(user_orders, prefix, pad)
            grouped = grouped.merge(dow_hod, on = [pad], how = 'left')
            del dow_hod

            interval_feat = user_orders[user_orders[prefix + 'purchase_interval'] != -1].groupby([pad]).agg(
                            {prefix + 'purchase_interval':{prefix + 'mean_interval': 'mean',
                                                           prefix + 'median_interval': 'median', 
                                                           prefix + 'std_interval': 'std',
                                                           prefix + 'min_interval': min, 
                                                           prefix + 'max_interval': max}})#.reset_index()
            interval_feat.columns = interval_feat.columns.droplevel(0)
            interval_feat.reset_index(inplace=True)
            # interval_feat = flatten_multiidx(interval_feat)
            interval_feat[prefix + 'std_interval'] = interval_feat[prefix + 'std_interval'].fillna(0)
            grouped = grouped.merge(interval_feat, on = [pad], how = 'left')
            del interval_feat, user_orders

            times = self.craft_feat_interact(pad)[[pad, 'u'+prefix+'order_num']]
            times_feat = times.groupby(pad).agg(
                {'u'+prefix+'order_num':{prefix + 'mean_times':'mean',
                                         prefix + 'median_times':'median',
                                         prefix + 'std_times':'std'}})# .reset_index()
            del times
            times_feat.columns = times_feat.columns.droplevel(0)
            times_feat.reset_index(inplace=True)
            # times_feat = flatten_multiidx(times_feat)
            times_feat[prefix + 'std_times'] = times_feat[prefix + 'std_times'].fillna(0)
            item_feat = grouped.merge(times_feat, on = [pad], how = 'left')
            del times_feat, grouped

            na_cols = [prefix + 'mean_interval', prefix + 'median_interval', prefix + 'min_interval', prefix + 'max_interval']
            for col in na_cols:
                item_feat[col] = item_feat[col].fillna(item_feat[prefix + 'days_to_last']) # only purchase once
            item_feat[prefix + 'std_interval'] = item_feat[prefix + 'std_interval'].fillna(0)

            item_feat.to_hdf(self.cache_dir + '%s_feat.h5'%pad[:-3], 'item', mode = 'w')
        return item_feat


    # def craft_feat_textual(self, item):
    #     '''
    #     TODO textual feat from item name
    #     word2vec 
    #     '''
    #     if os.path.exists(self.cache_dir  + 'textual_feat.pkl'):
    #         with open(self.cache_dir  + 'textual_feat.pkl', 'rb') as f:
    #             textual_feat = pickle.load(f)
    #     else:
    #         item_info = self.get_items(item)
    #         item_info[item[0] + '_organic'] = item_info[item[:-1] + '_name'].apply(is_organic)
    #         textual_feat = item_info[[item[:-1] + '_id', item[0] + '_organic']]
    #         with open(self.cache_dir  + 'textual_feat.pkl', 'wb') as f:
    #             pickle.dump(textual_feat, f, pickle.HIGHEST_PROTOCOL)        
    #     return textual_feat          
                        
    def craft_feat_pad(self):
        '''
            combine product, department, aisle
        '''
        if os.path.exists(self.cache_dir  + 'pad_feat.h5'):
            pad_feat = pd.read_hdf(self.cache_dir + 'pad_feat.h5')
        else:
            pad_feat =  (self.craft_feat_item('product_id')
                             .merge(self.get_items('products')[['product_id', 'department_id', 'aisle_id']], 
                                    on = ['product_id'], how = 'left'))
            pad_feat = pad_feat.merge(self.craft_feat_item('aisle_id'), on = ['aisle_id'], how = 'left')
            pad_feat = pad_feat.merge(self.craft_feat_item('department_id'), on = ['department_id'], how = 'left')
            # pad_feat = pad_feat.merge(self.craft_feat_textual('products'), on = ['product_id'], how = 'left')
            pad_feat['p_a_market_share'] = pad_feat['p_reorder_total'] / pad_feat['a_reorder_total']
            pad_feat['p_d_market_share'] = pad_feat['p_reorder_total'] / pad_feat['d_reorder_total']
            pad_feat['a_d_market_share'] = pad_feat['a_reorder_total'] / pad_feat['d_reorder_total']
            pad_feat['p_a_avg_add2cart'] = pad_feat['p_avg_add2cart_order'] / pad_feat['a_avg_add2cart_order']
            pad_feat['p_d_avg_add2cart'] = pad_feat['p_avg_add2cart_order'] / pad_feat['d_avg_add2cart_order']
            pad_feat['a_d_avg_add2cart'] = pad_feat['a_avg_add2cart_order'] / pad_feat['d_avg_add2cart_order']
            
            pad_feat['p_a_max_times'] = pad_feat['p_max_times'] / pad_feat['a_max_times']
            pad_feat['p_d_max_times'] = pad_feat['p_max_times'] / pad_feat['d_max_times']
            pad_feat['a_d_max_times'] = pad_feat['a_max_times'] / pad_feat['d_max_times']

            pad_feat['p_a_std_interval'] = pad_feat['p_std_interval'] / pad_feat['a_std_interval']
            pad_feat['p_d_std_interval'] = pad_feat['p_std_interval'] / pad_feat['d_std_interval']
            pad_feat['a_d_std_interval'] = pad_feat['a_std_interval'] / pad_feat['d_std_interval']

            pad_feat.to_hdf(self.cache_dir + 'pad_feat.h5', 'pad', mode = 'w')    
        return pad_feat 
                        
    def craft_feat_interact(self, pad):
        '''
        all users interact feat
        pad = ['product_id', 'aisle_id', 'department_id']
        '''
        if os.path.exists(self.cache_dir  + 'interact_feat_%s.h5'%pad[:-3]):
            interact_feat = pd.read_hdf(self.cache_dir +'interact_feat_%s.h5'%pad[:-3])
        else:
            user_product = self.get_users_orders('prior', pad).sort_values(['user_id', 'order_number']) 
            prefix = 'u'+ pad[0] + '_' 
            prefix_without_u = pad[0] + '_'                         
            grouped = user_product.groupby(['user_id', pad]).agg(
                {'reordered':{prefix +'reorder_num':sum,
                              prefix + 'order_num':'count'},
                 'order_number':{prefix + 'first_order':min,
                                 prefix + 'last_order':max},
                 'days_up_to_last':{prefix + 'days_to_last':min, # last purchase
                                    prefix + 'days_to_first':max}, # first purchase 
                 'add_to_cart_order':{prefix + 'min_add2cart_order':min, 
                                      prefix + 'max_add2cart_order':max,
                                      prefix + 'avg_add2cart_order':'mean', 
                                      prefix + 'std_add2cart_order':'std',
                                      prefix + 'med_add2cart_order':'median'}})#.reset_index()
            grouped.columns = grouped.columns.droplevel(0)
            grouped.reset_index(inplace=True)
            # grouped = flatten_multiidx(grouped) 

            grouped[prefix + 'active_days'] = grouped[prefix + 'days_to_first'] - grouped[prefix + 'days_to_last']

            grouped[prefix + 'std_add2cart_order'] = grouped[prefix + 'std_add2cart_order'].fillna(0)
            grouped = pd.merge(grouped, self.craft_feat_user()[['user_id', 
                                                       'u_total_orders',
                                                       'u_total_reorders',
                                                       'u_min_add2cart_order',
                                                       'u_max_add2cart_order',
                                                       'u_avg_add2cart_order',
                                                       'u_std_add2cart_order',
                                                       'u_med_add2cart_order']],
                               on = ['user_id'], how = 'left')
            
            grouped[prefix + 'order_since_last'] = grouped['u_total_orders'] - grouped[prefix + 'last_order']
            grouped[prefix + 'order_ratio_last'] = grouped[prefix + 'order_since_last'] / grouped['u_total_orders']

            grouped[prefix + 'order_ratio'] = grouped[prefix + 'order_num'] / grouped['u_total_orders']
            grouped[prefix + 'reorder_ratio'] = grouped[prefix + 'reorder_num'] / grouped['u_total_reorders']
            grouped[prefix + 'order_ratio_first'] = grouped[prefix + 'order_num'] / (grouped['u_total_orders'] - grouped[prefix + 'first_order'] + 1)           
            
            grouped[prefix + 'min_add2cart_ratio'] = grouped[prefix + 'min_add2cart_order'] / grouped['u_min_add2cart_order']
            grouped[prefix + 'max_add2cart_ratio'] = grouped[prefix + 'max_add2cart_order'] / grouped['u_max_add2cart_order']
            grouped[prefix + 'med_add2cart_ratio'] = grouped[prefix + 'med_add2cart_order'] / grouped['u_med_add2cart_order']
            grouped[prefix + 'avg_add2cart_ratio'] = grouped[prefix + 'avg_add2cart_order'] / grouped['u_avg_add2cart_order']
            grouped[prefix + 'std_add2cart_ratio'] = grouped[prefix + 'std_add2cart_order'] / grouped['u_std_add2cart_order']
            
            grouped[prefix + 'days_to_last_7'] = grouped[prefix + 'days_to_last'] % 7
            grouped[prefix + 'days_to_last_14'] = grouped[prefix + 'days_to_last'] % 14
            grouped[prefix + 'days_to_last_21'] = grouped[prefix + 'days_to_last'] % 21
            grouped[prefix + 'days_to_last_30'] = grouped[prefix + 'days_to_last'] % 30

            dow_hod = self.cal_dow_hod(user_product, prefix, ['user_id', pad])
            grouped = grouped.merge(dow_hod, on = ['user_id', pad], how = 'left')
            del dow_hod 

            user_product['last_order'] =user_product.groupby(['user_id', pad])['order_number'].transform(max)
            last_order = user_product[user_product['last_order'] == user_product['order_number']][['user_id', pad, 'order_hour_of_day', 'order_dow', 'days_since_prior_order']].drop_duplicates()
            last_order.columns = ['user_id', pad, prefix + 'last_hod', prefix + 'last_dow', prefix + 'last_days_since_prior']
            grouped = grouped.merge(last_order, on = ['user_id', pad], how = 'left')
            del last_order, user_product['last_order']

            avg_interval = (user_product[user_product.reordered == 1].groupby(['user_id', pad]) 
                                        ['days_since_prior_order'].mean().reset_index()) # fillna with last purchase
            avg_interval.columns = ['user_id', pad, prefix + 'avg_interval']
            grouped = grouped.merge(avg_interval, on = ['user_id', pad], how = 'left')
            del avg_interval

            grouped[prefix + 'avg_interval_m'] = grouped[prefix + 'days_to_first'] - grouped[prefix + 'days_to_last'] / grouped[prefix + 'order_num']
            
            interval_feat = (user_product[user_product[prefix_without_u + 'purchase_interval'] != -1].groupby(['user_id', pad]).agg({
                            prefix_without_u + 'purchase_interval':{prefix + 'median_interval': 'median', 
                                                                    prefix + 'std_interval': 'std',
                                                                    prefix + 'min_interval': min, 
                                                                    prefix + 'max_interval': max}}))#.reset_index()
            interval_feat.columns = interval_feat.columns.droplevel(0)
            interval_feat.reset_index(inplace=True)
            interval_feat[prefix + 'std_interval'] = interval_feat[prefix + 'std_interval'].fillna(0)
            grouped = grouped.merge(interval_feat, on = ['user_id', pad], how = 'left')
            del interval_feat

            user_product['order_number_last'] = user_product.groupby('user_id')['order_number'].transform(max)
            is_last_purchase = (user_product[user_product.order_number == user_product.order_number_last]
                                            .groupby(['user_id', pad]).apply(lambda x:1).reset_index())
            is_last_purchase.columns = [['user_id', pad, prefix + 'is_purchase_last']]
            interact_feat = grouped.merge(is_last_purchase, on = ['user_id', pad], how = 'left')
            del is_last_purchase
   
            na_cols = [prefix + 'avg_interval', prefix + 'median_interval', prefix + 'min_interval', prefix + 'max_interval']
            for col in na_cols:
                interact_feat[col] = interact_feat[col].fillna(interact_feat[prefix + 'days_to_last']) # only purchase once
            na_cols = [prefix + 'reorder_ratio', prefix + 'std_interval', prefix + 'is_purchase_last']
            interact_feat[na_cols] = interact_feat[na_cols].fillna(0)
            na_cols = [prefix + 'std_add2cart_ratio']
            interact_feat[na_cols] = interact_feat[na_cols].fillna(1)

            del interact_feat['u_total_orders']
            del interact_feat['u_total_reorders']
            del interact_feat['u_min_add2cart_order']
            del interact_feat['u_max_add2cart_order']
            del interact_feat['u_avg_add2cart_order']
            del interact_feat['u_std_add2cart_order']
            del interact_feat['u_med_add2cart_order']
            
            interact_feat.to_hdf(self.cache_dir + 'interact_feat_%s.h5'%pad[:-3], 'interact', mode = 'w')
        return interact_feat

    def craft_user_topic(self, filepath = None):
        '''
        TODO
        user_topic from lda model
        '''
        if filepath is None:
            filepath = self.cache_dir  + 'user_topic_%d.pkl'%NUM_TOPIC
        else:
            filepath = self.cache_dir + filepath
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                user_topic = pickle.load(f)
        else:
            print(filepath)
            pass
        return user_topic

    def craft_product_topic(self, filepath = None):
        '''
        TODO
        user_topic from lda model
        '''
        if filepath is None:
            filepath = self.cache_dir  + 'topic_product_%d.pkl'%NUM_TOPIC
        else:
            filepath = self.cache_dir + filepath
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                topic_product = pickle.load(f)
        else:
            print(filepath)
            pass
        return topic_product
    
    def craft_up_distance(self, filepath = None, num_topic = NUM_TOPIC, pad = 'product_id'):
        '''
        calculate (u,p) pairs distance 
        using LDA embedded representation
        '''
        if isinstance(filepath, list):
            p_filepath, u_filepath = filepath[0], filepath[1]
            filepath = self.cache_dir + p_filepath[:6] + 'feat.pkl'
            prefix = p_filepath[:6]
        else:
            p_filepath, u_filepath = None, None
            filepath = self.cache_dir  + 'upd_feat_%d.pkl'%num_topic
            prefix = ''

        if os.path.exists(filepath):
            upd = pd.read_pickle(filepath)
        else:
            def cal_up_distance(subf):
                u_topic = subf[[prefix + "u_topic_%d"%x for x in range(num_topic)]]
                p_topic = subf[[prefix + "p_topic_%d"%x for x in range(num_topic)]]
                upd = euclidean(u_topic, p_topic)
                return upd
        
            upd = pd.merge(self.get_users_orders('prior')[['user_id', pad]].drop_duplicates(),
                               self.craft_user_topic(u_filepath), 
                               on = ['user_id'],
                               how = 'left')

            upd.columns = ['user_id', pad] + [prefix + "u_topic_%d"%x for x in range(num_topic)] 

            upd = pd.merge(upd,
                               self.craft_product_topic(p_filepath), 
                               on = [pad],
                               how = 'left')

            upd.columns = ['user_id', pad] + [prefix + "u_topic_%d"%x for x in range(num_topic)] + [prefix + "p_topic_%d"%x for x in range(num_topic)]
            
            for col in [prefix + "p_topic_%d"%x for x in range(num_topic)]:
                upd[col] = upd[col].fillna(upd[col].mean())
                
            upd[prefix + 'up_dis'] = upd.apply(cal_up_distance, axis = 1)
            upd[prefix + 'up_dis'] = upd[prefix + 'up_dis'].fillna(upd[prefix + 'up_dis'].mean())
            
            with open(filepath, 'wb') as f:
                pickle.dump(upd, f, pickle.HIGHEST_PROTOCOL)
        return upd
    
    def craft_p_w2v(self):
        filepath = self.cache_dir + 'p_w2v_feat.pkl'
        p_w2v = pd.read_pickle(filepath)
        return p_w2v
    
    def craft_topic_pc(self):
        ''' compressed topic feat by PCA'''
        filepath = self.cache_dir  + 'up_topic_pc.h5'
        up_topic_pc = pd.read_hdf(filepath)
        return up_topic_pc
    
    def craft_topic_dis(self):
        filepath = self.cache_dir  + 'up_topic_dis.h5'
        up_topic_dis = pd.read_hdf(filepath)
        return up_topic_dis  
    
    def craft_up_interval(self):
        filepath = self.cache_dir + 'up_delta.pkl' 
        up_delta = pd.read_pickle(filepath)
        return up_delta
    
    def craft_dream_score(self):
        filepath = self.cache_dir + 'dream_score.pkl'
        dream_score = pd.read_pickle(filepath)
        return dream_score
    
    def craft_dream_score_next(self, is_reordered=False):
        if is_reordered is True:
            filepath = self.cache_dir + 'reorder_dream_score_next.pkl'
        else:           
            filepath = self.cache_dir + 'dream_score_next.pkl'
        dream_score = pd.read_pickle(filepath)
        return dream_score
    
    def craft_dream_final(self, is_reordered=False):
        if is_reordered is True:
            filepath = self.cache_dir + 'reorder_dream_final.pkl'
        else:       
            filepath = self.cache_dir + 'dream_final.pkl'
        dream_final = pd.read_pickle(filepath)
        return dream_final

    def craft_dream_dynamic_u(self, is_reordered=False):
        if is_reordered is True:
            filepath = self.cache_dir + 'reorder_dream_dynamic_u.pkl'
        else:       
            filepath = self.cache_dir + 'dream_dynamic_u.pkl'
        dream_dynamic_u = pd.read_pickle(filepath)
        return dream_dynamic_u

    def craft_dream_item_embed(self, is_reordered=False):
        if is_reordered is True:
            filepath = self.cache_dir + 'reorder_dream_item_embed.pkl'
        else:       
            filepath = self.cache_dir + 'dream_item_embed.pkl'
        dream_item_embed = pd.read_pickle(filepath)
        return dream_item_embed
    
    def craft_order_streak(self):
        with open(self.cache_dir + 'up_order_streak.pkl', 'rb') as f:
            up_order_streak = pickle.load(f)
        return up_order_streak
