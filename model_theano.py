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

from taeksoo.cnn_util import CNN

class Caption_Generator():
    def __init__(self, n_words, dim_embed, dim_hidden, dim_image):
        self.n_words = n_words
        self.dim_embed = dim_embed
        self.dim_hidden = dim_hidden
        self.dim_image = dim_image

        self.Wemb = initializations.uniform((n_words, dim_embed))

        self.lstm_W = initializations.uniform((dim_embed, dim_hidden*4))
        self.lstm_U = initializations.uniform((dim_hidden, dim_hidden*4))
        self.lstm_b = initializations.zero((dim_hidden*4))

        self.encode_img_W = initializations.uniform((dim_image, dim_hidden))
        self.encode_img_b = initializations.zero((dim_hidden))

        self.hidden_emb_W = initializations.uniform((dim_hidden, dim_embed))
        self.hidden_emb_b = initializations.zero((dim_embed))

        self.emb_word_W = initializations.uniform((dim_embed, n_words))
        self.emb_word_b = initializations.uniform((n_words))

        self.params = [
                self.Wemb,
                self.lstm_W, self.lstm_U, self.lstm_b,
                self.encode_img_W, self.encode_img_b,
                self.hidden_emb_W, self.hidden_emb_b,
                self.emb_word_W, self.emb_word_b
            ]

    def forward_lstm(self, x, mask):

        def _step(m_tm_1, x_t, h_tm_1, c_tm_1):
            lstm_preactive = \
                T.dot(h_tm_1, self.lstm_U) + \
                T.dot(x_t, self.lstm_W) + \
                self.lstm_b

            i = T.nnet.sigmoid(lstm_preactive[:, 0*self.dim_hidden:1*self.dim_hidden])
            f = T.nnet.sigmoid(lstm_preactive[:, 1*self.dim_hidden:2*self.dim_hidden])
            o = T.nnet.sigmoid(lstm_preactive[:, 2*self.dim_hidden:3*self.dim_hidden])
            c = T.tanh(lstm_preactive[:, 3*self.dim_hidden:4*self.dim_hidden])

            c = f*c_tm_1 + i*c
            c = m_tm_1[:,None]*c + (1.-m_tm_1)[:,None]*c_tm_1

            h = o*T.tanh(c)
            h = m_tm_1[:,None]*h + (1.-m_tm_1)[:,None]*h_tm_1

            return [h,c]

        h0 = T.alloc(0., x.shape[1], self.dim_hidden)
        c0 = T.alloc(0., x.shape[1], self.dim_hidden)

        rval, updates = theano.scan(
                fn=_step,
                sequences=[mask,x],
                outputs_info=[h0,c0]
                )

        h_list, c_list = rval
        return h_list

    def build_model(self):
        image = T.matrix('image')
        sentence = T.imatrix('sentence')
        mask = T.matrix('mask')

        n_samples, n_timestep = sentence.shape
        emb_dimshuffle = self.Wemb[sentence.flatten()].reshape((n_timestep, n_samples, -1))
        mask_dimshuffle = mask.dimshuffle(1,0)

        image_enc = T.dot(image, self.encode_img_W) + self.encode_img_b # (n_samples, dim_hidden)

        X = T.concatenate([image_enc[None,:,:], emb_dimshuffle], axis=0) # (n_timestep + 1, n_samples, dim_hidden)
        X = X[:-1] # 마지막 단어 빼고

        h_list = self.forward_lstm(X, mask_dimshuffle) # (n_timestep, n_samples, dim_hidden)

        output_emb = T.dot(h_list, self.hidden_emb_W) + self.hidden_emb_b # (n_timestep, n_samples, dim_embed)
        output_word = T.tanh(T.dot(output_emb, self.emb_word_W) + self.emb_word_b) #(n_timestep, n_samples, n_words)
        output_word = output_word.dimshuffle(1,0,2) # (n_samples, n_timestep, n_words)

        output_word_shape = output_word.shape

        probs = T.nnet.softmax(output_word.reshape((output_word_shape[0]*output_word_shape[1], output_word_shape[2])))

        sentence_flat = sentence.flatten()
        probs_flat = probs.flatten()
        cost = -T.log(probs_flat[T.arange(sentence_flat.shape[0])*probs.shape[1] + sentence_flat])

        cost = cost.reshape((sentence.shape[0], sentence.shape[1]))
        masked_cost = cost * mask

        cost_mean = (masked_cost).sum() / mask.sum()

        return image, sentence, mask, masked_cost, cost_mean, output_word, h_list

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []

    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))

    return updates

def get_caption_data(annotation_path, image_path):
    annotations = pd.read_table(annotation_path, sep='\t', header=None, names=['image', 'caption'])
    annotations['image'] = annotations['image'].map(lambda x: os.path.join(image_path, x.split('#')[0]))
    images = annotations['image'].values
    captions = annotations['caption'].values

    return images, captions

def train():
    n_epochs = 100
    batch_size = 128
    n_words = 10000

    dim_embed = 512
    dim_hidden = 512
    dim_image = 4096

    learning_rate = 0.001

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

    index = np.arange(len(images))
    np.random.shuffle(index)

    images = images[index]
    captions = captions[index]


    caption_generator = Caption_Generator(n_words, dim_embed, dim_hidden, dim_image)

    theano_image, theano_sentence, theano_mask, theano_cost_arr, theano_cost, output_word, hid = caption_generator.build_model()

    f_output_word = theano.function(
            inputs=[theano_image, theano_sentence, theano_mask],
            outputs=output_word,
            allow_input_downcast=True)

    f_hid = theano.function(
            inputs=[theano_image, theano_sentence, theano_mask],
            outputs=hid,
            allow_input_downcast=True)

    f_cost = theano.function(
            inputs=[theano_image, theano_sentence, theano_mask],
            outputs=theano_cost_arr,
            allow_input_downcast=True)

    for epoch in range(n_epochs):

        updates = RMSprop(cost=theano_cost, params=caption_generator.params, lr=learning_rate)
        train_function = theano.function(
                inputs=[theano_image, theano_sentence, theano_mask],
                outputs=theano_cost,
                updates=updates,
                allow_input_downcast=True)

        for start, end in zip(
                range(0, len(images)+batch_size, batch_size),
                range(batch_size, len(images)+batch_size, batch_size)
                ):

            current_images = images[start:end]
            current_captions = captions[start:end]

            image_train = cnn.get_features(current_images, layers='fc7')
            current_caption_ind = map(lambda cap: map(lambda word: dictionary[word] if word in dictionary else 1, cap.lower().split(' ')[:-1]), current_captions)

            maxlen = np.max(map(lambda x: len(x), current_caption_ind)) + 1

            caption_train = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen)
            mask_train = np.zeros_like(caption_train)

            nonzeros = np.array(map(lambda x: (x != 0).sum(), caption_train))

            for ind, row in enumerate(mask_train):
                row[:nonzeros[ind]+1] = 1

            cost = train_function(image_train, caption_train, mask_train)

#            output_word = f_output_word(image_train, caption_train, mask_train)
#            output_hidden = f_hid(image_train, caption_train, mask_train)
#            cost_arr = f_cost(image_train, caption_train, mask_train)
            print cost

        with open('./cv/iter_'+str(epoch)+'.pickle', 'w') as f:
            cPickle.dump(caption_generator, f)
        learning_rate *= 0.98



