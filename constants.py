XGB_DIR = '/home/public/Instacart/xgb/'
LGB_DIR = '/home/public/Instacart/lgb/'

NUM_TOPIC = 22
LDA_DIR = '/home/public/Instacart/lda/'
W2V_DIR = '/home/public/Instacart/w2v/'

RAW_DATA_DIR = '/home/public/Instacart/raw/'
FEAT_DATA_DIR = '/home/public/Instacart/feat/'
EVA_DATA_DIR = '/home/public/Instacart/eval/'

ID_COLS = ['{}_id'.format(i) for i in ['user', 'order', 'product', 'aisle', 'department']]

MODEL_PATH_1 = [
'/home/public/Instacart/lgb/lgb_gbdt_0.8397670268252372',
'/home/public/Instacart/lgb/lgb_gbdt_0.8397598897794645',
'/home/public/Instacart/lgb/lgb_gbdt_0.8402202937445229',
]

MODEL_PATH_2 = [
'/home/public/Instacart/lgb/lgb_mtwdr_goss_0.839882_0.403618',
'/home/public/Instacart/lgb/lgb_mtwdr_goss_0.839443_0.402621',
'/home/public/Instacart/lgb/lgb_mtwdr_goss_0.837824_0.401767',
'/home/public/Instacart/lgb/lgb_mtwdr_goss_0.838340_0.401663',
]

MODEL_PATH_3 = [
'/home/public/Instacart/xgb/xgb_vip_0.839904_0.404341',
]

MODEL_PATH_4 = [
'/home/public/Instacart/lgb/lgb_train_vip_gbdt_0.840647',
'/home/public/Instacart/lgb/lgb_train_vip_gbdt_0.840869',
]

MODEL_PATH_5 = [
'/home/public/Instacart/xgb/xgb_vip_gbtree_0.840101_0.404459']

MODEL_PATH_6 = [
'/home/public/Instacart/lgb/lgb_train_vip_gbdt_0.839489',
'/home/public/Instacart/lgb/lgb_train_vip_gbdt_0.840898']

MODEL_PATH_7 = [
'/home/public/Instacart/lgb/lgb_train_vip_gbdt_0.840190',
'/home/public/Instacart/lgb/lgb_train_vip_gbdt_0.839205'
]

MODEL_PATH_8 = [
'/home/public/Instacart/lgb/lgb_train_vip_gbdt_0.840647'
]