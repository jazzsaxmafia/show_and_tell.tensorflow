#-*- coding: utf-8 -*-
import math
import os
import ipdb
import tensorflow as tf
import numpy as np
import pandas as pd
import cPickle

from tensorflow.models.rnn import rnn_cell
import tensorflow.python.platform
from keras.preprocessing import sequence
from cnn_util import *

class Caption_Generator():
    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, dim_image, dim_embed, dim_hidden, batch_size, n_lstm_steps, n_words):

        self.dim_image = np.int(dim_image)
        self.dim_embed = np.int(dim_embed)
        self.dim_hidden = np.int(dim_hidden)
        self.batch_size = np.int(batch_size)
        self.n_lstm_steps = np.int(n_lstm_steps)
        self.n_words = np.int(n_words)

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -1.0, 1.0), name='Wemb')

        self.bemb = self.init_bias(dim_embed, name='bemb')

        self.lstm = rnn_cell.BasicLSTMCell(dim_hidden)

        self.encode_img_W = self.init_weight(dim_image, dim_hidden, name='encode_img_W')
        self.encode_img_b = self.init_bias(dim_hidden, name='encode_img_b')

        self.hidden_emb_W = self.init_weight(dim_hidden, dim_embed, name='hidden_emb_W')
        self.hidden_emb_b = self.init_bias(dim_embed, name='hidden_emb_b')

        self.embed_word_W = self.init_weight(dim_embed, n_words, name='embed_word_W')
        self.embed_word_b = self.init_bias(n_words, name='embed_word_b')

    def build_model(self):

        image = tf.placeholder(tf.float32, [self.batch_size, self.dim_image])
        sentence = tf.placeholder(tf.int32, [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        image_emb = tf.matmul(image, self.encode_img_W) + self.encode_img_b # (batch_size, dim_hidden)
        image_emb = tf.expand_dims(image_emb, dim=1) # 이미지와 sentence가 차원이 달라서 더미 차원 추가해줘야함.

        with tf.device("/cpu:0"):
            sentence_emb = tf.nn.embedding_lookup(self.Wemb, sentence)

        sentence_emb += self.bemb

        sentence_emb = tf.concat(concat_dim=1, values=[image_emb, sentence_emb])
        sentence_emb = tf.nn.dropout(sentence_emb, 0.5)
        state = tf.zeros([self.batch_size, self.lstm.state_size])

        loss = 0.0
        with tf.variable_scope("RNN"):
            for i in range(self.n_lstm_steps): # maxlen + 1
                if i > 0 : tf.get_variable_scope().reuse_variables()
                labels = tf.expand_dims(sentence[:, i], 1) # (batch_size)
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat(1, [indices, labels])
                onehot_labels = tf.sparse_to_dense(
                        concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0) # (batch_size, n_words)

                output, state = self.lstm(sentence_emb[:,i,:], state) # (batch_size, dim_hidden)

                logits = tf.matmul(output, self.hidden_emb_W) + self.hidden_emb_b # (batch_size, dim_embed)
                logits = tf.nn.relu(logits)
                logits = tf.nn.dropout(logits, 0.5)

                logit_words = tf.matmul(logits, self.embed_word_W) + self.embed_word_b # (batch_size, n_words)
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
                cross_entropy = cross_entropy * mask[:,i]#tf.expand_dims(mask, 1)

                current_loss = tf.reduce_sum(cross_entropy)
                loss = loss + current_loss

        loss = loss / tf.reduce_sum(mask)
        return loss, image, sentence, mask

    def build_generator(self, maxlen):
        image = tf.placeholder(tf.float32, [1, self.dim_image])
        image_emb = tf.matmul(image, self.encode_img_W) + self.encode_img_b

        state = tf.zeros([1, self.lstm.state_size])
        #last_word = image_emb # 첫 단어 대신 이미지
        generated_words = []

        with tf.variable_scope("RNN"):
            output, state = self.lstm(image_emb, state)
            last_word = tf.nn.embedding_lookup(self.Wemb, [2]) + self.bemb

            for i in range(maxlen):
                tf.get_variable_scope().reuse_variables()

                output, state = self.lstm(last_word, state)

                logits = tf.matmul(output, self.hidden_emb_W) + self.hidden_emb_b
                logits = tf.nn.relu(logits)

                logit_words = tf.matmul(logits, self.embed_word_W) + self.embed_word_b
                max_prob_word = tf.argmax(logit_words, 1)

                with tf.device("/cpu:0"):
                    last_word = tf.nn.embedding_lookup(self.Wemb, max_prob_word)

                last_word += self.bemb

                generated_words.append(max_prob_word)

        return image, generated_words

    def generate_from_image(self, k):
        image = tf.placeholder(tf.float32, [1, self.dim_image])
        image_emb = tf.matmul(image, self.encode_img_W) + self.encode_img_b

        state = tf.zeros([1., self.lstm.state_size])

        with tf.variable_scope("RNN"):
            output, state = self.lstm(image_emb, state)
            tf.get_variable_scope().reuse_variables()

            starting_emb = tf.nn.embedding_lookup(self.Wemb, [2]) + self.bemb
            output, state = self.lstm( starting_emb, state ) # 이미지 넣고 그다음 2(#START#) 넣고 시작
            logits = tf.matmul(output, self.hidden_emb_W) + self.hidden_emb_b
            logits = tf.nn.relu(logits)

            logit_words = tf.matmul(logits, self.embed_word_W) + self.embed_word_b

            softmax = tf.nn.softmax(logit_words)
            top_k_scores, top_k_indices = tf.nn.top_k(softmax, k)

        return image, state, output, top_k_indices, top_k_scores

    def generate_from_word(self, k):
        last_state = tf.placeholder(tf.float32, [1, self.dim_hidden*2])
        current_word = tf.placeholder(tf.int32, [1])
        with tf.device("/cpu:0"):
            current_emb = tf.nn.embedding_lookup(self.Wemb, current_word)

        current_emb += self.bemb

        with tf.variable_scope("RNN"):
            tf.get_variable_scope().reuse_variables()
            output, state = self.lstm(current_emb, last_state)
            logits = tf.matmul(output, self.hidden_emb_W) + self.hidden_emb_b
            logits = tf.nn.relu(logits)

            logit_words = tf.matmul(logits, self.embed_word_W) + self.embed_word_b

            softmax = tf.nn.softmax(logit_words)
            top_k_scores, top_k_indices = tf.nn.top_k(softmax, k)

        return last_state, current_word, state, output, top_k_indices, top_k_scores

def get_caption_data(annotation_path, feat_path):
     feats = np.load(feat_path)
     annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
     captions = annotations['caption'].values

     return feats, captions

################### 학습 관련 Parameters #####################
n_words = 10000

dim_embed = 512
dim_hidden = 512
dim_image = 4096
batch_size = 200

#learning_rate = 0.001
n_epochs = 1000
###############################################################
#################### 잡다한 Parameters ########################
model_path = './models'
data_path = './data'
feat_path = './data/feats.npy'
annotation_path = os.path.join(data_path, 'results_20130124.token')
dictionary_path = os.path.join(data_path, 'dictionary.pkl')
################################################################

def train():

    learning_rate = 0.001
    dictionary = pd.read_pickle(dictionary_path)
    feats, captions = get_caption_data(annotation_path, feat_path)

    index = np.arange(len(feats))
    np.random.shuffle(index)

    feats = feats[index]
    captions = captions[index]

    sess = tf.InteractiveSession()
    maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )
    caption_generator = Caption_Generator(
            dim_image=dim_image,
            dim_hidden=dim_hidden,
            dim_embed=dim_embed,
            batch_size=batch_size,
            n_lstm_steps=maxlen+2,
            n_words=n_words)

    loss, image, sentence, mask = caption_generator.build_model()

    saver = tf.train.Saver(max_to_keep=50)
    tf.initialize_all_variables().run()

    step = 0
    for epoch in range(n_epochs):
        train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        #train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)
        for start, end in zip( \
                range(0, len(feats), batch_size),
                range(batch_size, len(feats), batch_size)
                ):

            current_feats = feats[start:end]
            current_captions = captions[start:end]

            current_caption_ind = map(lambda cap: map(lambda word: dictionary[word] if word in dictionary else 1, \
                    cap.lower().split(' ')[:-1]), current_captions)

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
            current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 2), current_caption_matrix] ).astype(int)

            # padding을 maxlen+1만큼 해주는 건 마지막에 0을 박기 위해서임.

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))
            # 여기서 +1 해주는 건 캡션 맨 앞에 #START# (2) 를 넣었기 때문임.

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]+1] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                image: current_feats,
                sentence : current_caption_matrix,
                mask : current_mask_matrix
                })

            print "Current Cost: ", loss_value
            step += 1

        print "Epoch ", epoch, " is done. Saving the model ... "
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
        learning_rate *= 0.95

def test(test_feat='./data/test_feat2.npy', model_path='./models/model-77', maxlen=30): # Naive greedy search

    dictionary = pd.read_pickle(dictionary_path)
    inverse_dictionary = pd.Series(dictionary.keys(), index=dictionary.values)
    feat = [np.load(test_feat)]
    sess = tf.InteractiveSession()
    caption_generator = Caption_Generator(
           dim_image=dim_image,
           dim_hidden=dim_hidden,
           dim_embed=dim_embed,
           batch_size=batch_size,
           n_lstm_steps=maxlen,
           n_words=n_words)

    image, generated_words = caption_generator.build_generator(maxlen=maxlen)
    # 이 부분이 존나 중요함. 계속 caption_generator를 가져온 뒤 바로 restore를 했었는데,
    # TensorFlow의 LSTM은 call을 한 뒤에 weight가 만들어지기 때문에 build_generator보다 뒤쪽에서 restore를 해야 함.
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    generated_word_index= sess.run(generated_words, feed_dict={image:feat})
    generated_word_index = np.hstack(generated_word_index)

    generated_sentence = inverse_dictionary[generated_word_index]

    ipdb.set_trace()

def test_v2(test_feat='./data/test_feat.npy', model_path='./models/model-77', maxlen=30): # Beam Search
    k = 8

    dictionary = pd.read_pickle(dictionary_path)
    inverse_dictionary = pd.Series(dictionary.keys(), index=dictionary.values)
    feat = [np.load(test_feat)]
    sess = tf.InteractiveSession()
    caption_generator = Caption_Generator(
           dim_image=dim_image,
           dim_hidden=dim_hidden,
           dim_embed=dim_embed,
           batch_size=batch_size,
           n_lstm_steps=maxlen,
           n_words=n_words)

    image, state, output, top_k_words, top_k_scores = caption_generator.generate_from_image(k=k)
    last_state, current_word, state_, output_, top_k_words_, top_k_scores_ = caption_generator.generate_from_word(k=k)

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    last_state_val = sess.run(state, feed_dict={image:feat}).repeat(k, axis=0)
    top_k_words_val = sess.run(top_k_words, feed_dict={image:feat})[0]

    ipdb.set_trace()

    last_sents = top_k_words_val#.repeat(k)[:,None]

    for i in range(maxlen):

        all_state = []
        all_words = []
        all_scores = []
        last_sents = last_sents.repeat(k, axis=0)
        if len(last_sents.shape ) == 1:
            last_sents = last_sents[:,None]

        for k_ in range(k): # 매 번 k*k개의 sub-sentence가 생성됨. 이 중에서 k개를 골라야 함.
            current_state_val = sess.run(state_, feed_dict={
                last_state:[last_state_val[k_]],
                current_word: [top_k_words_val[k_]]
                })

            current_top_k_words_val = sess.run(top_k_words_, feed_dict={
                last_state:[last_state_val[k_]],
                current_word: [top_k_words_val[k_]]
                })

            current_top_k_scores_val = sess.run(top_k_scores_, feed_dict={
                last_state:[last_state_val[k_]],
                current_word: [top_k_words_val[k_]]
                })

            all_state.append(current_state_val)
            all_words.append(current_top_k_words_val[0])
            all_scores.append(current_top_k_scores_val[0])

        all_state = np.vstack(all_state)
        all_words = np.hstack(all_words)
        all_scores = np.hstack(all_scores)

        current_top_k_scores_ind = np.hstack(all_scores).argsort()[::-1][:k]
        last_top_k_subsents = last_sents[current_top_k_scores_ind]
        current_top_k_subsents = all_words[current_top_k_scores_ind]

        last_state_val = all_state[current_top_k_scores_ind / k]

        last_sents = map(lambda (x,y): x.tolist() + [y], zip(last_top_k_subsents, current_top_k_subsents))
        last_sents = np.vstack(last_sents)

        ipdb.set_trace()







