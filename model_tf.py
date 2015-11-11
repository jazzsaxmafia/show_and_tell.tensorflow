#-*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import pandas as pd

from tensorflow.models.rnn import rnn_cell
from cnn_util import *

n_words = 10000

dim_embed = 512
dim_hidden = 512
dim_image = 4096
batch_size = 100

num_steps = 20
learning_rate = 0.001

image = tf.placeholder(tf.float32, [batch_size, dim_image])
sentence = tf.placeholder(tf.int32, [batch_size, num_steps])
mask = tf.placeholder(tf.float32, [batch_size, num_steps])


Wemb = tf.Variable( # 워드 임베딩
        tf.random_uniform([n_words, dim_embed], -1.0, 1.0))

lstm = rnn_cell.BasicLSTMCell(dim_hidden)

encode_img_W = tf.Variable(
        tf.truncated_normal([dim_image, dim_hidden], stddev=0.1)
        )
encode_img_b = tf.Variable(
        tf.zeros([dim_hidden]),
        )

hidden_emb_W = tf.Variable(
        tf.truncated_normal([dim_hidden, dim_embed], stddev=0.1)
        )
hidden_embed_b = tf.Variable(
        tf.zeros([dim_embed]),
        )

embed_word_W = tf.Variable(
        tf.truncated_normal([dim_embed, n_words], stddev=0.1)
        )
embed_word_b = tf.Variable(
        tf.zeros([n_words]),
        )


image_emb = tf.matmul(image, encode_img_W) + encode_img_b # (batch_size, dim_hidden)
image_emb = tf.expand_dims(image_emb, dim=1) # 이미지와 sentence가 차원이 달라서 더미 차원 추가해줘야함.
sentence_emb = tf.nn.embedding_lookup(Wemb, sentence) # (batch_size, n_samples, dim_hidden)

sentence_emb = tf.concat(concat_dim=1, values=[image_emb, sentence_emb])
sentence_emb = tf.transpose(sentence_emb, perm=[1,0,2])
#inputs = map(lambda x: tf.squeeze(x), tf.split(1, num_steps+1, sentence_emb))

initial_state = state = tf.zeros([batch_size, lstm.state_size])
#for i in range(num_steps):
with tf.variable_scope("RNN"):
    for i in range(21):
        if i > 0 : tf.get_variable_scope().reuse_variables()
        output, state = lstm(sentence_emb[i,:,:], state)


################################################################
def get_caption_data(annotation_path, image_path):
     annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
     annotations['image'] = annotations['image'].map(lambda x: os.path.join(image_path, x.split('#')[0]))
     images = annotations['image'].values
     captions = annotations['caption'].values

     return images, captions

vgg_model = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_19_layers.caffemodel'
vgg_deploy = '/home/taeksoo/Package/caffe/models/vgg/VGG_ILSVRC_19_layers_deploy.prototxt'

image_path = '/home/taeksoo/Study/show_attend_and_tell/images/'
data_path = '/home/taeksoo/Study/show_attend_and_tell/data/flickr30k'
annotation_path = os.path.join(data_path, 'results_20130124.token')
flickr_image_path = os.path.join(image_path, 'flickr30k-images')
dictionary_path = os.path.join(data_path, 'dictionary.pkl')

cnn = CNN(
     deploy=vgg_deploy,
     model=vgg_model,
     batch_size=20,
     width=224,
     height=224)

dictionary = pd.read_pickle(dictionary_path)
images, captions = get_caption_data(annotation_path, flickr_image_path)
