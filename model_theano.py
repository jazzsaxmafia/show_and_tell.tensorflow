#-*- coding: utf-8 -*-
import theano
import theano.tensor as T

import numpy as np
import pandas as pd

import os
import ipdb
import cPickle

from keras import initializations
from keras.preprocessing import sequence

class Caption_Generator():
    def __init__(self, n_words, dim_embed, dim_hidden, dim_image, bias_init_vector=None):
        self.n_words = n_words
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.dim_image = dim_image

        self.Wemb = initializations.uniform((n_words, dim_embed), scale=0.1)
        self.bemb = initializations.zero((dim_embed))

        self.lstm_W = initializations.uniform((1 + dim_embed + dim_hidden, dim_hidden*4), scale=0.1)

        self.encode_img_W = initializations.uniform((dim_image, dim_hidden), scale=0.1)
        self.encode_img_b = initializations.zero((dim_hidden))

        self.emb_word_W = initializations.uniform((dim_hidden, n_words), scale=0.1)
        if bias_init_vector is None:
            self.emb_word_b = initializations.uniform((n_words))
        else:
            self.emb_word_b = theano.shared(bias_init_vector, borrow=True)

        self.params = [
                self.Wemb, self.bemb,
                self.lstm_W,
                self.encode_img_W, self.encode_img_b,
                self.emb_word_W, self.emb_word_b
            ]

    def forward_lstm(self, x): # x: (n_timestep, n_samples, dim_embed)

        def _step(x_t, h_tm_1, c_tm_1):
            ones = T.ones([x_t.shape[0], 1])
            input_concat = T.concatenate([ones, x_t, h_tm_1], 1)
            lstm_preactive = T.dot(input_concat, self.lstm_W)

            i = T.nnet.sigmoid(lstm_preactive[:, 0*self.dim_hidden:1*self.dim_hidden])
            f = T.nnet.sigmoid(lstm_preactive[:, 1*self.dim_hidden:2*self.dim_hidden])
            o = T.nnet.sigmoid(lstm_preactive[:, 2*self.dim_hidden:3*self.dim_hidden])
            c = T.tanh(lstm_preactive[:, 3*self.dim_hidden:4*self.dim_hidden])

            c = f*c_tm_1 + i*c
            h = o*T.tanh(c)

            return [h,c]

        h0 = T.alloc(0., x.shape[1], self.dim_hidden)
        c0 = T.alloc(0., x.shape[1], self.dim_hidden)

        rval, updates = theano.scan(
                fn=_step,
                sequences=[x],
                outputs_info=[h0,c0]
                )

        h_list, c_list = rval
        return h_list

    def generate_lstm(self, x, maxlen):
        h0 = T.alloc(0., x.shape[0], self.dim_hidden)
        c0 = T.alloc(0., x.shape[0], self.dim_hidden)

        ones = T.ones([x.shape[0], 1])
        input_concat = T.concatenate([ones, x, h0], 1)
        lstm_preactive = T.dot(input_concat, self.lstm_W)

        i = T.nnet.sigmoid(lstm_preactive[:, 0*self.dim_hidden:1*self.dim_hidden])
        f = T.nnet.sigmoid(lstm_preactive[:, 1*self.dim_hidden:2*self.dim_hidden])
        o = T.nnet.sigmoid(lstm_preactive[:, 2*self.dim_hidden:3*self.dim_hidden])
        c = T.tanh(lstm_preactive[:, 3*self.dim_hidden:4*self.dim_hidden])

        c = f*c0 + i*c
        h = o*T.tanh(c)

        start_tag = self.Wemb[0][None,:] + self.bemb

        def _step(x_t, h_tm_1, c_tm_1):
            ones = T.ones([x_t.shape[0], 1])
            input_concat = T.concatenate([ones, x_t, h_tm_1], 1)
            lstm_preactive = T.dot(input_concat, self.lstm_W)

            i = T.nnet.sigmoid(lstm_preactive[:, 0*self.dim_hidden:1*self.dim_hidden])
            f = T.nnet.sigmoid(lstm_preactive[:, 1*self.dim_hidden:2*self.dim_hidden])
            o = T.nnet.sigmoid(lstm_preactive[:, 2*self.dim_hidden:3*self.dim_hidden])
            c = T.tanh(lstm_preactive[:, 3*self.dim_hidden:4*self.dim_hidden])

            c = f*c_tm_1 + i*c
            h = o*T.tanh(c)

            output_word = T.dot(h, self.emb_word_W) + self.emb_word_b #(1, 1, n_words)
            max_index = output_word.flatten().argmax()

            next_x = self.Wemb[max_index][None,:] + self.bemb

            return [h,c, next_x]

        rval, updates = theano.scan(
                fn=_step,
                outputs_info=[h,c,start_tag],
                maxlen=maxlen)

        h_list, c_list, x_list = rval
        return x_list

    def build_model(self):
        image = T.matrix('image')
        sentence = T.imatrix('sentence')
        mask = T.matrix('mask')

        n_samples, n_timestep = sentence.shape
        emb_dimshuffle = self.Wemb[sentence.flatten()].reshape((n_timestep, n_samples, -1)) + self.bemb

        image_enc = T.dot(image, self.encode_img_W) + self.encode_img_b # (n_samples, dim_hidden)
        X = T.concatenate([image_enc[None,:,:], emb_dimshuffle], axis=0) # (n_timestep + 1, n_samples, dim_hidden)

        h_list = self.forward_lstm(X)#, mask_dimshuffle) # (n_timestep, n_samples, dim_hidden)

        output_word = T.dot(h_list, self.emb_word_W) + self.emb_word_b #(n_timestep, n_samples, n_words)
        output_word = output_word.dimshuffle(1,0,2) # (n_samples, n_timestep, n_words)

        output_word_shape = output_word.shape

        probs = T.nnet.softmax(output_word.reshape((output_word_shape[0]*output_word_shape[1], output_word_shape[2])))

        sentence_flat = sentence.flatten()
        probs_flat = probs.flatten()
        cost = -T.log(probs_flat[T.arange(sentence_flat.shape[0])*probs.shape[1] + sentence_flat])

        cost = cost.reshape((sentence.shape[0], sentence.shape[1]))
        masked_cost = (cost * mask)[:, 1:]

        cost_mean = (masked_cost).sum() / mask[:,1:].sum()

        return image, sentence, mask, masked_cost, cost_mean, output_word, h_list

    def build_generator(self, maxlen):
        image = T.matrix('image')
        image_enc = T.dot(image, self.encode_img_W) + self.encode_img_b

        generated_sents = self.generate_lstm(image_enc)
        return theano.function(inputs=[image], outputs=generated_sents, allow_input_downcast=True)

def RMSprop(cost, params, lr=0.001, rho=0.999, epsilon=1e-8, grad_clip=2.):
    grads = T.grad(cost=cost, wrt=params)
    updates = []

    #grads = T.clip(grads, -grad_clip, grad_clip)

    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = T.clip(g, -grad_clip,grad_clip) / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))

    return updates

def get_caption_data(annotation_path, feat_path):
    feats = np.load(feat_path)
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
    captions = annotations['caption'].values

    return feats, captions

def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # borrowed this function from NeuralTalk
       # count up all word counts so that we can threshold
       # this shouldnt be too expensive of an operation
       print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, )
       word_counts = {}
       nsents = 0
       for sent in sentence_iterator:
         nsents += 1
         for w in sent.lower().split(' '):
           word_counts[w] = word_counts.get(w, 0) + 1
       vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
       print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

       # with K distinct words:
       # - there are K+1 possible inputs (START token and all the words)
       # - there are K+1 possible outputs (END token and all the words)
       # we use ixtoword to take predicted indeces and map them to words for output visualization
       # we use wordtoix to take raw words and get their index in word vector matrix
       ixtoword = {}
       ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
       wordtoix = {}
       wordtoix['#START#'] = 0 # make first vector be the start token
       ix = 1
       for w in vocab:
         wordtoix[w] = ix
         ixtoword[ix] = w
         ix += 1
       # compute bias vector, which is related to the log probability of the distribution
       # of the labels (words) and how often they occur. We will use this vector to initialize
       # the decoder weights, so that the loss function doesnt show a huge increase in performance
       # very quickly (which is just the network learning this anyway, for the most part). This makes
       # the visualizations of the cost function nicer because it doesn't look like a hockey stick.
       # for example on Flickr8K, doing this brings down initial perplexity from ~2500 to ~170.
       word_counts['.'] = nsents
       bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
       bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
       bias_init_vector = np.log(bias_init_vector)
       bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
       return wordtoix, ixtoword, bias_init_vector


#################### 잡다한 Parameters ########################
model_path = './models'
data_path = './data'
feat_path = './data/feats.npy'
annotation_path = os.path.join(data_path, 'results_20130124.token')
dictionary_path = os.path.join(data_path, 'dictionary.pkl')
################################################################


def train():
    n_epochs = 100
    batch_size = 128

    dim_embed = 256
    dim_hidden = 256
    dim_image = 4096

    learning_rate = 0.001

    feats, captions = get_caption_data(annotation_path, feat_path)
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)

    n_words = len(wordtoix)

    index = np.arange(len(feats))
    np.random.shuffle(index)

    feats = feats[index]
    captions = captions[index]

    caption_generator = Caption_Generator(n_words, dim_embed, dim_hidden, dim_image, bias_init_vector=bias_init_vector)
    theano_image, theano_sentence, theano_mask, theano_cost_arr, theano_cost, output_word, hid = caption_generator.build_model()

    for epoch in range(n_epochs):

        updates = RMSprop(cost=theano_cost, params=caption_generator.params, lr=learning_rate)
        train_function = theano.function(
                inputs=[theano_image, theano_sentence, theano_mask],
                outputs=theano_cost,
                updates=updates,
                allow_input_downcast=True)

        for start, end in zip(
                range(0, len(feats)+batch_size, batch_size),
                range(batch_size, len(feats)+batch_size, batch_size)
                ):

            current_feats = feats[start:end]
            current_captions = captions[start:end]

            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)

            maxlen = np.max(map(lambda x: len(x), current_caption_ind)) + 1

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)
            current_caption_matrix = np.hstack( [np.full( (len(current_caption_matrix),1), 0), current_caption_matrix] ).astype(int)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array(map(lambda x: (x != 0).sum()+1, current_caption_matrix))

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]+1] = 1

            ipdb.set_trace()

            cost = train_function(current_feats, current_caption_matrix, current_mask_matrix)

            print cost

        with open('./models/theano/iter_'+str(epoch)+'.pickle', 'w') as f:
            cPickle.dump(caption_generator, f)
        learning_rate *= 0.98
