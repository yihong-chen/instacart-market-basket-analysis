from constants import LDA_DIR, FEAT_DATA_DIR

import pickle
import logging
import pandas as pd
from time import time

import gensim
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', 
    level=logging.DEBUG, 
    filename='./logs/gensim_search0.log',
    filemode='w')

n_topics = [20, 40, 60, 80]

logging.info('Product ID ......')
t0 = time()
with open(FEAT_DATA_DIR + 'product_gensim_tf', 'rb') as f:
	p_term_matrix = pickle.load(f)

with open(FEAT_DATA_DIR + 'product_gensim_dict', 'rb') as f:
	p_dict = pickle.load(f)

for n in n_topics:
    logging.info("number of topics: %d"%n)
    lda = LdaMulticore(p_term_matrix,
    	               num_topics = n,
    	               workers = 13,
    	               id2word = p_dict,
                       passes = 50,
                       eval_every = 1,
    	               iterations = 1000)

    lda.save(LDA_DIR + 'p_gensim_lda_%d'%n)
    logging.info("done in %0.3fs." % (time() - t0))
logging.info("Product finished!")

# logging.info('Aisle ID ......')
# t0 = time()
# with open(FEAT_DATA_DIR + 'aisle_gensim_tf', 'rb') as f:
# 	p_term_matrix = pickle.load(f)

# with open(FEAT_DATA_DIR + 'aisle_gensim_dict', 'rb') as f:
# 	p_dict = pickle.load(f)

# for n in n_topics:
#     logging.info("number of topics: %d"%n)
#     lda = LdaMulticore(p_term_matrix,
#     	               num_topics = n,
#     	               workers = 13,
#     	               id2word = p_dict,
#                        passes = 50,
#                        eval_every = 1,
#     	               iterations = 1000)
#     lda.save(LDA_DIR + 'a_gensim_lda_%d'%n)

#     logging.info("done in %0.3fs." % (time() - t0))
# logging.info("finished aisle!")

# logging.info('Department ID ......')
# t0 = time()
# with open(FEAT_DATA_DIR + 'department_gensim_tf', 'rb') as f:
# 	p_term_matrix = pickle.load(f)

# with open(FEAT_DATA_DIR + 'department_gensim_dict', 'rb') as f:
# 	p_dict = pickle.load(f)

# for n in n_topics:
#     logging.info("number of topics: %d"%n)
#     lda = LdaMulticore(p_term_matrix,
#     	               num_topics = n,
#     	               workers = 13,
#     	               id2word = p_dict,
#                        passes = 50,
#                        eval_every = 1,
#     	               iterations = 1000)
#     lda.save(LDA_DIR + 'd_gensim_lda_%d'%n)

#     logging.info("done in %0.3fs." % (time() - t0))
# logging.info("finished department!")


	



	
