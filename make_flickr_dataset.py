import pandas as pd
import numpy as np
import os
import scipy
import ipdb
import json
import cPickle
from sklearn.feature_extraction.text import CountVectorizer

annotation_path = './data/results_20130124.token'
flickr_image_path = '../show_attend_and_tell/images/flickr30k/'

annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
annotations['image_num'] = annotations['image'].map(lambda x: x.split('#')[1])
annotations['image'] = annotations['image'].map(lambda x: os.path.join(flickr_image_path,x.split('#')[0]))

captions = annotations['caption'].values

vectorizer = CountVectorizer(max_features=3000 - 3, token_pattern='\\b\\w+\\b').fit(captions)
dictionary = vectorizer.vocabulary_
dictionary = pd.Series(dictionary) + 3
#dictionary = pd.Series(dictionary) + 5

dictionary['#END#'] = 0
dictionary['UNK'] = 1
dictionary['#START#'] = 2
#dictionary[','] = 2
#dictionary['!'] = 3
#dictionary['?'] = 4

with open('./data/dictionary.pkl', 'wb') as f:
    cPickle.dump(dictionary, f)
with open('./data/vectorizer.pkl', 'wb') as f:
    cPickle.dump(vectorizer, f)
