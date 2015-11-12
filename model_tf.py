#-*- coding: utf-8 -*-
import os
import ipdb
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

n_lstm_steps = 20
learning_rate = 0.001

class Caption_Generator():
    def init_weight(self, dim_in, dim_out, stddev=0.1):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev))

    def init_bias(self, dim_out):
        return tf.Variable(tf.zeros([dim_out]))

    def __init__(self, dim_image, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words):
        self.dim_image = dim_image
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.n_words = n_words


        self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -1.0, 1.0))

        self.lstm = rnn_cell.BasicLSTMCell(dim_hidden)

        self.encode_img_W = self.init_weight(dim_image, dim_hidden)
        self.encode_img_b = self.init_bias(dim_hidden)

        self.hidden_emb_W = self.init_weight(dim_hidden, dim_embed)
        self.hidden_emb_b = self.init_bias(dim_embed)

        self.embed_word_W = self.init_weight(dim_embed, n_words)
        self.embed_word_b = self.init_bias(n_words)

    def build_model(self):
        image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        sentence = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        image_emb = tf.matmul(image, self.encode_img_W) + self.encode_img_b # (batch_size, dim_hidden)
        image_emb = tf.expand_dims(image_emb, dim=1) # 이미지와 sentence가 차원이 달라서 더미 차원 추가해줘야함.
        sentence_emb = tf.nn.embedding_lookup(self.Wemb, sentence) # (batch_size, n_samples, dim_hidden)

        sentence_emb = tf.concat(concat_dim=1, values=[image_emb, sentence_emb])

        initial_state = state = tf.zeros([self.batch_size, self.lstm.state_size])

        loss = 0.0
        output_list = []
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps):
                if i > 0 : tf.get_variable_scope().reuse_variables()
                labels = tf.expand_dims(sentence[:, i], 1) # (batch_size)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat(1, [indices, labels])
                onehot_labels = tf.sparse_to_dense(
                        concated, tf.pack([batch_size, self.n_words]), 1.0, 0.0) # (batch_size, n_words)

                output, state = self.lstm(sentence_emb[:,i,:], state) # (batch_size, dim_hidden)
                output_list.append(output)

                logits = tf.matmul(output, self.hidden_emb_W) + self.hidden_emb_b # (batch_size, dim_embed)
                logits = tf.nn.tanh(logits)

                logit_words = tf.matmul(logits, self.embed_word_W) + self.embed_word_b # (batch_size, n_words)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
                cross_entropy = cross_entropy * mask[:,i]#tf.expand_dims(mask, 1)

                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss

        loss = loss / tf.reduce_sum(mask)



        ipdb.set_trace()

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

caption_generator = Caption_Generator(
        dim_image=dim_image,
        dim_hidden=dim_hidden,
        dim_embed=dim_embed,
        batch_size=batch_size,
        n_lstm_steps=n_lstm_steps,
        n_words=n_words)

dictionary = pd.read_pickle(dictionary_path)
images, captions = get_caption_data(annotation_path, flickr_image_path)

caption_generator.build_model()
