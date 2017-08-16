from constants import LDA_DIR, FEAT_DATA_DIR
from utils import series_to_str

import pickle
import pandas as pd
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

#n_topics = [20, 50, 100]
#n_topics = [10, 30, 50, 70, 90]
n_topics = [10, 60, 110, 160, 210]

t0 = time()
with open(FEAT_DATA_DIR + 'product_tf_matrix', 'rb') as f:
	tf_matrix = pickle.load(f)

for n in n_topics:
    print("number of topics: %d"%n)
    lda = LatentDirichletAllocation(n_topics=n,
                                	evaluate_every  = 5,
                                	max_iter = 1000,
                                	n_jobs = 1,
                               		verbose = 1)

    lda.fit(tf_matrix)
    with open(LDA_DIR + 'p_lda_%d.model'%n, 'wb') as f:
        pickle.dump(lda, f, pickle.HIGHEST_PROTOCOL)

    print("done in %0.3fs." % (time() - t0))
print("finished!")


	
