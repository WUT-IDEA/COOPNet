import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import warnings
warnings.filterwarnings('ignore')

import re
# import sys
import random as rn
import gc
import argparse
import string
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from keras.models import load_model, Model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Embedding, TimeDistributed, Dropout
from keras.layers import concatenate, maximum
from keras.layers import Bidirectional, GRU
from keras.layers import Add, BatchNormalization, Activation
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.core import Lambda
from keras import regularizers
from sklearn import metrics
from skimage.color import rgb2hsv

from mycode.attention import MultiHeadAttention, MultiHeadSelfAttention
from mycode.vgg16_keras import VGG16
# from mycode.buildModel_2 import han_model,transformer_gru_model, transformer_model, transformer_gru_stepSentiment_model
from mycode.utils import get_fig_path_list, load_image, list2hist
from mycode.text_plus_fig_attention import TextFigAtten, AttLayer
from mycode.Merge import ColumnMaximum, ColumnAverage, CrossMaximum, CrossAverage, WeightedVote, Vote
from keras import backend as K
from mycode.Generator import *
from mycode.my_modules import auc

# np.random.seed(12)

MAX_VOCAB_SIZE = 50000
# MAXLEN = 50
MAXLEN = 30
# LONG_MAXLEN = 300
sentence_timestep = 100
# word_timestep = MAXLEN
EMBED_SIZE = 300
VALID_RATE = 0.2
train_batch_size = 15
test_batch_size = 10
# train_batch_size = 1
# test_batch_size = 1

fig_resize_shape = 150
hist_count = 360

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'input', 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = os.path.join(os.path.dirname(__file__), '..', 'input',
                                   'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')


train_root_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'train')
test_root_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'test')
train_label_path = os.path.join(os.path.dirname(__file__), '..','input','train','en.txt')
test_label_path = os.path.join(os.path.dirname(__file__), '..','input','test','en.txt')

TRAIN_FIGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'input', 'train', 'photo')
TEST_FIGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'input', 'test', 'photo')




def text_preprocess(train_ori, test_ori):
    print('remove patterns...')
    train_ori['text_list'] = train_ori['text_list'].apply(lambda list: _remove_pattern_2(list))
    test_ori['text_list'] = test_ori['text_list'].apply(lambda list: _remove_pattern_2(list))

    print('join text list...')
    train_text = train_ori['text_list'].apply(lambda list: " ".join(list))
    test_text = test_ori['text_list'].apply(lambda list: " ".join(list))

    print('prepare tokenizer')
    tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)#词汇表最多单词数
    tokenizer.fit_on_texts( list(train_text) + list(test_text) )#Updates internal vocabulary based on a list of texts.

    print('texts to sequences...')
    train_ori['seq'] = train_ori['text_list'].apply(lambda list: tokenizer.texts_to_sequences(list))
    test_ori['seq'] = test_ori['text_list'].apply(lambda list: tokenizer.texts_to_sequences(list))

    print('pad sequences...')
    train_ori['seq'] = train_ori['seq'].apply(lambda list: sequence.pad_sequences(list, maxlen=MAXLEN))
    test_ori['seq'] = test_ori['seq'].apply(lambda list: sequence.pad_sequences(list, maxlen=MAXLEN))

    return train_ori, test_ori, tokenizer

def _remove_pattern_2(input_text_list):

    cleaned_text_list = []
    for text in input_text_list:
        text = text.translate(string.punctuation)# Remove puncuation 去除标点
        text = text.lower()# Convert words to lower case and split them

        # text = " ".join(text)

        # Clean the text
        text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)  # 除A-Za-z0-9(),!?'`外的字符，去除
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r",", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"'", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)

        # text = text.split()
        # stemmer = SnowballStemmer('english')
        # stemmed_words = [stemmer.stem(word) for word in text]
        # text = " ".join(stemmed_words)

        cleaned_text_list.append(text)
    return cleaned_text_list


def read_data():#返回label和图像的generator
    # train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train.pd'))
    # test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test.pd'))
    #
    # train, test, tokenizer = text_preprocess(train, test)#修改'text_list'这一列，移除pattern；新增'seq'这一列：变成sequence
    # # 只是为了不要在数据处理上浪费太多时间
    #
    # pd.to_pickle(train, os.path.join(os.path.dirname(__file__), '..', 'output', 'train_rmpattern_seq.pd'))
    # pd.to_pickle(test, os.path.join(os.path.dirname(__file__), '..', 'output', 'test_rmpattern_seq.pd'))


    # train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_rmpattern_seq.pd'))
    # test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_rmpattern_seq.pd'))
    # #shuffle训练集
    # train = train.iloc[np.random.permutation(len(train))]
    # pd.to_pickle(train, path=os.path.join(os.path.dirname(__file__), '..', 'output', 'train_rmpattern_seq_shuffle.pd'))
    # # 注释掉前面这些的原因是为了使每次训练集随机的结果一样，希望能保存下这个随机的结果，以免后面因随机性而有偶然性

    train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_rmpattern_seq_shuffle.pd'))
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_rmpattern_seq.pd'))


    # 将训练集分成训练集和验证集
    train_size = int(len(train)*(1-VALID_RATE))
    valid = train[train_size:]
    train = train[:train_size]

    # 取出pd中的label
    train_label = train['label'].apply(lambda gender: [1] if gender=='male' else [0])
    valid_label = valid['label'].apply(lambda gender: [1] if gender=='male' else [0])
    test_label = test['label'].apply(lambda gender: [1] if gender=='male' else [0])

    return train, valid, test, train_label, valid_label, test_label

def get_generator(train, valid, test, train_label, valid_label, test_label, corp):
    ############## 产生generator ##############

    # 1.产生(fig, label)的generator
    if corp == 'fig':
        train_fig_path_list = train['fig_path_list']
        valid_fig_path_list = valid['fig_path_list']
        test_fig_path_list = test['fig_path_list']
        generator_train = generator_fig_batch(train_fig_path_list, train_label, train_batch_size, fig_resize_shape)
        generator_valid = generator_fig_batch(valid_fig_path_list, valid_label, train_batch_size, fig_resize_shape)
        generator_test = generator_fig_batch(test_fig_path_list, test_label, test_batch_size, fig_resize_shape)
        generator_predict = generator_fig_batch(test_fig_path_list, test_label, test_batch_size, fig_resize_shape, nolabel=True)
        generator_predict_label = generator_label(test_label, test_batch_size)

    # elif corp == 'fig_hue':
    #     train_fig_path_list = train['fig_path_list']
    #     valid_fig_path_list = valid['fig_path_list']
    #     test_fig_path_list = test['fig_path_list']
    #     generator_train = generator_fig_hue_hist_batch(train_fig_path_list, train_label, hist_count, train_batch_size, fig_resize_shape)
    #     generator_valid = generator_fig_hue_hist_batch(valid_fig_path_list, valid_label, hist_count, train_batch_size, fig_resize_shape)
    #     generator_test = generator_fig_hue_hist_batch(test_fig_path_list, test_label, hist_count, test_batch_size, fig_resize_shape)



    # 2.产生(text, fig, label)的generator
    elif corp == 'textfig':
        train_fig_path_list = train['fig_path_list']
        valid_fig_path_list = valid['fig_path_list']
        test_fig_path_list = test['fig_path_list']
        train_text_seq_list = train['seq']
        valid_text_seq_list = valid['seq']
        test_text_seq_list = test['seq']
        generator_train = generator_fig_text_batch(train_text_seq_list, train_fig_path_list, train_label, train_batch_size, fig_resize_shape)
        generator_valid = generator_fig_text_batch(valid_text_seq_list, valid_fig_path_list, valid_label, train_batch_size, fig_resize_shape)
        generator_test = generator_fig_text_batch(test_text_seq_list, test_fig_path_list, test_label, test_batch_size, fig_resize_shape)
        generator_predict = generator_fig_text_batch(test_text_seq_list, test_fig_path_list, test_label, test_batch_size, fig_resize_shape, nolabel=True)
        generator_predict_label = generator_label(test_label, test_batch_size)


    elif corp == 'textfig_multi':
        train_fig_path_list = train['fig_path_list']
        valid_fig_path_list = valid['fig_path_list']
        test_fig_path_list = test['fig_path_list']
        train_text_seq_list = train['seq']
        valid_text_seq_list = valid['seq']
        test_text_seq_list = test['seq']
        generator_train = generator_fig_text_multioutput_batch(train_text_seq_list, train_fig_path_list, train_label, train_batch_size, fig_resize_shape)
        generator_valid = generator_fig_text_multioutput_batch(valid_text_seq_list, valid_fig_path_list, valid_label, train_batch_size, fig_resize_shape)
        generator_test = generator_fig_text_multioutput_batch(test_text_seq_list, test_fig_path_list, test_label, test_batch_size, fig_resize_shape)
        generator_predict = generator_fig_text_multioutput_batch(test_text_seq_list, test_fig_path_list, test_label, test_batch_size, fig_resize_shape, nolabel=True)
        generator_predict_label = generator_label(test_label, test_batch_size)


    # elif corp == 'textfig_multi_4':
    #     train_fig_path_list = train['fig_path_list']
    #     valid_fig_path_list = valid['fig_path_list']
    #     test_fig_path_list = test['fig_path_list']
    #     train_text_seq_list = train['seq']
    #     valid_text_seq_list = valid['seq']
    #     test_text_seq_list = test['seq']
    #     generator_train = generator_fig_text_multioutput_4_batch(train_text_seq_list, train_fig_path_list, train_label, train_batch_size, fig_resize_shape)
    #     generator_valid = generator_fig_text_multioutput_4_batch(valid_text_seq_list, valid_fig_path_list, valid_label, train_batch_size, fig_resize_shape)
    #     generator_test = generator_fig_text_multioutput_4_batch(test_text_seq_list, test_fig_path_list, test_label, test_batch_size, fig_resize_shape)

    # 3.产生(text, label)的generator
    elif corp == 'text':
        train_text_seq_list = train['seq']
        valid_text_seq_list = valid['seq']
        test_text_seq_list = test['seq']
        generator_train = generator_text_batch(train_text_seq_list, train_label, train_batch_size)
        generator_valid = generator_text_batch(valid_text_seq_list, valid_label, train_batch_size)
        generator_test = generator_text_batch(test_text_seq_list, test_label, test_batch_size)
        generator_predict = generator_text_batch(test_text_seq_list, test_label, test_batch_size, nolabel=True)
        generator_predict_label = generator_label(test_label, test_batch_size)



    elif corp =='textsentifig_multi':
        train_fig_path_list = train['fig_path_list']
        valid_fig_path_list = valid['fig_path_list']
        test_fig_path_list = test['fig_path_list']
        train_text_seq_list = train['seq']
        valid_text_seq_list = valid['seq']
        test_text_seq_list = test['seq']
        generator_train = generator_fig_text_senti_multioutput_batch(train_text_seq_list, train_fig_path_list, train_label, train_batch_size, fig_resize_shape)
        generator_valid = generator_fig_text_senti_multioutput_batch(valid_text_seq_list, valid_fig_path_list, valid_label, train_batch_size, fig_resize_shape)
        generator_test = generator_fig_text_senti_multioutput_batch(test_text_seq_list, test_fig_path_list, test_label, test_batch_size, fig_resize_shape)
        generator_predict = generator_fig_text_senti_multioutput_batch(test_text_seq_list, test_fig_path_list, test_label, test_batch_size, fig_resize_shape, nolabel=True)
        generator_predict_label = generator_label(test_label, test_batch_size)



    else:
        print('corp is wrong: text/fig/textfig...')
        exit(0)

    return train_label, valid_label, test_label, generator_train, generator_valid, generator_test, generator_predict, generator_predict_label


def buildModel_fig():
    input = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='input_figs')
    x = TimeDistributed(Conv2D(filters=10, kernel_size=(20, 20), strides=1,
                               # padding='same',
                               activation='relu',
                               data_format='channels_last'))(input)
    x = TimeDistributed(AveragePooling2D())(x)
    x = TimeDistributed(Conv2D(filters=10, kernel_size=(10, 10), strides=1,
                               # padding='same',
                               activation='relu',
                               data_format='channels_last'))(input)
    x = TimeDistributed(AveragePooling2D())(x)
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(1000, activation='relu'))(x)
    x = GRU(500, return_sequences=False)(x)
    x = Dense(100)(x)
    y = Dense(1)(x)
    model = Model(inputs=[input], outputs=[y])
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy', auc])
    return model

def buildModel_fig_vgg16():# 只有图像，vgg16
    input = Input(shape=(10, fig_resize_shape,fig_resize_shape,3), dtype='float32',name='input_fig')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = True

    x = TimeDistributed(Flatten())(x)
    # x = TimeDistributed(Dense(5000, activation='relu'))(x)
    x = TimeDistributed(Dense(1000, activation='sigmoid', kernel_regularizer=regularizers.l1(0.0005)))(x)

    x = ColumnAverage()(x)

    # x = Dense(2000,activation='relu')(x)
    x = Dense(100, activation='relu')(x)
    y = Dense(1, activation='relu')(x)

    model = Model(inputs=[input], outputs=[y])
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model

def taka_fig():
    input = Input(shape=(10, fig_resize_shape,fig_resize_shape,3), dtype='float32',name='input_fig')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = True

    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.00005)))(x)

    x = ColumnAverage()(x)

    x = Dense(100, activation='sigmoid')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy', auc])
    return model




def buildModel_fig_vgg16_mergelast():# 只有图像，vgg16
    input = Input(shape=(10, fig_resize_shape,fig_resize_shape,3), dtype='float32',name='input_fig')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = True

    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(1000, activation='sigmoid', kernel_regularizer=regularizers.l1(0.0005)))(x)
    x = TimeDistributed(Dense(100, activation='sigmoid'))(x)
    x = TimeDistributed(Dense(1, activation='relu'))(x)
    y = ColumnAverage()(x)


    model = Model(inputs=[input], outputs=[y])
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def buildModel_fig_hue_hist():
    # 10张flatten
    # input = Input(shape=(10,hist_count), dtype='float32', name='input_figs')
    # x = TimeDistributed(Dense(100,activation='relu'))(input)
    # x = Flatten()(x)#(1000,)
    # x = Dense(400,activation='relu')(x)
    # x = Dense(15,activation='relu')(x)
    # y = Dense(1,activation='sigmoid')(x)


    # 10张ColumnAverage
    # input = Input(shape=(10,hist_count), dtype='float32', name='input_figs')
    # x = ColumnAverage()(input)
    # x = Dense(100,activation='relu')(x)
    # x = Dense(15,activation='relu')(x)
    # y = Dense(1,activation='sigmoid')(x)


    # 10张ColumnMaximum
    # input = Input(shape=(10,hist_count), dtype='float32', name='input_figs')
    # x = ColumnMaximum()(input)
    # x = Dense(100,activation='relu')(x)
    # x = Dense(15,activation='relu')(x)
    # y = Dense(1,activation='sigmoid')(x)


    input = Input(shape=(10, hist_count), dtype='float32', name='input_figs')
    x = Flatten()(input)
    x = Dense(1000, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    y = Dense(1, activation='sigmoid')(x)



    model = Model(inputs=[input], outputs=[y])
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model




def buildModel_text_fig():# 文本和图像拼接
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True))(wordsEmbedding)
    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  ))(wordAtten)
    wordDense = TimeDistributed(Dense(150))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True)(wordDense)
    sentenceGRU = GRU(64)(sentenceAtten)
    # sentenceDense = Dense(20)(sentenceGRU)
    # y = Dense(1, activation="sigmoid", name="documentOut")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')
    each_fig_conv_1 = TimeDistributed(Conv2D(filters=5, kernel_size=(20, 20), strides=1,
                               # padding='same',
                               activation='relu',
                               data_format='channels_last'))(input_fig)
    each_fig_pool_1 = TimeDistributed(AveragePooling2D())(each_fig_conv_1)
    each_fig_conv_2 = TimeDistributed(Conv2D(filters=10, kernel_size=(10, 10), strides=1,
                               # padding='same',
                               activation='relu',
                               data_format='channels_last'))(each_fig_pool_1)
    each_fig_pool_2 = TimeDistributed(AveragePooling2D())(each_fig_conv_2)
    each_fig_flat = TimeDistributed(Flatten())(each_fig_pool_2)
    each_fig_dense = TimeDistributed(Dense(1000, activation='relu'))(each_fig_flat)
    fig_group_gru = GRU(500, return_sequences=False)(each_fig_dense)
    fig_group_dense = Dense(15)(fig_group_gru)
    # fig_group_dense = Dense(20)(fig_group_dense)


    ############## concatenate #############
    concat = concatenate([sentenceGRU, fig_group_dense], axis=1)


    y = Dense(1, activation='sigmoid')(concat)
    model = Model(inputs=[input_text, input_fig], outputs=[y])
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def buildModel_text_fig_taka():# 文本和图像拼接
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True))(wordsEmbedding)
    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  ))(wordAtten)
    wordDense = TimeDistributed(Dense(150))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True)(wordDense)
    sentenceGRU = GRU(64)(sentenceAtten)
    # sentenceDense = Dense(20)(sentenceGRU)
    # y = Dense(1, activation="sigmoid", name="documentOut")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input_fig)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input_fig, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = False

    fig_flatten = TimeDistributed(Flatten())(x)
    fig_dense = TimeDistributed(Dense(1000, kernel_regularizer=regularizers.l1(0.00005)))(fig_flatten)
    fig_dense = TimeDistributed(Dense(64, kernel_regularizer=regularizers.l1(0.00005)))(fig_dense)
    fig_merge = ColumnAverage()(fig_dense)


    ############## takahash merge #############
    crossMerge = CrossMaximum()([sentenceGRU, fig_merge])


    y = Dense(1, activation='sigmoid', name='merge_y')(crossMerge)
    model = Model(inputs=[input_text, input_fig], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def buildModel_text_fig_taka_multioutput():# 文本和图像拼接
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True))(wordsEmbedding)
    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  ))(wordAtten)
    wordDense = TimeDistributed(Dense(150))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True)(wordDense)
    sentenceGRU = GRU(64)(sentenceAtten)
    sentenceDense = Dense(20)(sentenceGRU)
    y_text = Dense(1, activation="sigmoid", name="y_text")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input_fig)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input_fig, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = False

    fig_flatten = TimeDistributed(Flatten())(x)
    fig_dense = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.00005)))(fig_flatten)
    fig_dense = TimeDistributed(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.00005)))(fig_dense)
    fig_merge = ColumnAverage()(fig_dense)
    fig_dense = Dense(20,activation='sigmoid')(fig_merge)
    y_fig = Dense(1, activation='sigmoid',name='y_fig')(fig_dense)


    ############## takahash merge #############
    crossMerge = CrossMaximum()([sentenceGRU, fig_merge])# 64+64
    crossDense = Dense(20,activation='sigmoid')(crossMerge)


    y_merge = Dense(1, activation='sigmoid', name='y_merge')(crossDense)
    model = Model(inputs=[input_text, input_fig], outputs=[y_text, y_fig, y_merge])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model





def buildModel_text_fig_attention():
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True))(wordsEmbedding)
    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  ))(wordAtten)
    wordDense = TimeDistributed(Dense(150))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True)(wordDense)
    sentenceGRU = GRU(64)(sentenceAtten)
    # sentenceDense = Dense(20)(sentenceGRU)
    # y = Dense(1, activation="sigmoid", name="documentOut")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')
    each_fig_conv_1 = TimeDistributed(Conv2D(filters=16, kernel_size=(5, 5), strides=1,
                               # padding='same',
                               activation='relu',
                               data_format='channels_last'))(input_fig)
    each_fig_pool_1 = TimeDistributed(AveragePooling2D())(each_fig_conv_1)
    each_fig_conv_2 = TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), strides=1,
                               # padding='same',
                               activation='relu',
                               data_format='channels_last'))(each_fig_pool_1)
    each_fig_pool_2 = TimeDistributed(AveragePooling2D())(each_fig_conv_2)
    each_fig_flat = TimeDistributed(Flatten())(each_fig_pool_2)
    each_fig_dense = TimeDistributed(Dense(1000, activation='relu'))(each_fig_flat)

    # text_fig_atten
    text_fig_attention = TextFigAtten()([sentenceGRU, each_fig_dense])
    text_fig_attention = Dense(15)(text_fig_attention)

    # concatenate
    concat = concatenate([sentenceGRU, text_fig_attention])


    y = Dense(1, activation='sigmoid')(concat)
    model = Model(inputs=[input_text, input_fig], outputs=[y])
    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def buildModel_text_fig_regu_attention():
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE, name='word_embedding'))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True, name='word_multihead'))(wordsEmbedding)
    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  name='word_gru'))(wordAtten)
    wordDense = TimeDistributed(Dense(150, name='word_dense'))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True,name='sentence_multihead')(wordDense)
    sentenceGRU = GRU(64,name='sentence_gru')(sentenceAtten)
    sentenceDense = Dense(20)(sentenceGRU)
    y_text = Dense(1, activation="sigmoid", name="y_text")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input_fig)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input_fig, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = False
    fig_flatten = TimeDistributed(Flatten())(x)

    fig_dense = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.00005)))(fig_flatten)
    fig_atten = TextFigAtten()([sentenceGRU, fig_dense])
    fig_dense_1 = Dense(64,activation='sigmoid')(fig_atten)
    fig_dense_2 = Dense(20,activation='sigmoid')(fig_dense_1)
    y_fig = Dense(1, activation='sigmoid',name='y_fig')(fig_dense_2)


    # concatenate
    merge = concatenate([sentenceGRU, fig_dense_1])#64+64
    merge_dense = Dense(20, activation='sigmoid')(merge)


    y_merge = Dense(1, activation='sigmoid', name='y_merge')(merge_dense)
    model = Model(inputs=[input_text, input_fig], outputs=[y_text, y_fig, y_merge])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def buildModel_text_fig_regu_finetune_attention():
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE, name='word_embedding'))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True, name='word_multihead'))(wordsEmbedding)
    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  name='word_gru'))(wordAtten)
    wordDense = TimeDistributed(Dense(150, name='word_dense'))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True,name='sentence_multihead')(wordDense)
    sentenceGRU = GRU(64,name='sentence_gru')(sentenceAtten)
    sentenceDense = Dense(20)(sentenceGRU)
    y_text = Dense(1, activation="sigmoid", name="y_text")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input_fig)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input_fig, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = True
    fig_flatten = TimeDistributed(Flatten())(x)

    fig_dense = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.00005)))(fig_flatten)
    fig_atten = TextFigAtten()([sentenceGRU, fig_dense])
    fig_dense_1 = Dense(64,activation='sigmoid')(fig_atten)
    fig_dense_2 = Dense(20,activation='sigmoid')(fig_dense_1)
    y_fig = Dense(1, activation='sigmoid',name='y_fig')(fig_dense_2)


    # concatenate
    merge = concatenate([sentenceGRU, fig_dense_1])#64+64
    merge_dense = Dense(20, activation='sigmoid')(merge)


    y_merge = Dense(1, activation='sigmoid', name='y_merge')(merge_dense)
    model = Model(inputs=[input_text, input_fig], outputs=[y_text, y_fig, y_merge])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model



def buildModel_text_fig_regu_finetune_readweights_attention():
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE, name='word_embedding'))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True, name='word_multihead'))(wordsEmbedding)
    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  name='word_gru'))(wordAtten)
    wordDense = TimeDistributed(Dense(150, name='word_dense'))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True,name='sentence_multihead')(wordDense)
    sentenceGRU = GRU(64,name='sentence_gru')(sentenceAtten)

    text_model = Model(input_text, sentenceGRU)
    text_model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                         'only_text_ckp', 'model-epoch_08-val_acc_0.80.hdf5'),
                            by_name=True)
    text_model.trainable = True


    sentenceDense = Dense(20)(sentenceGRU)
    y_text = Dense(1, activation="sigmoid", name="y_text")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input_fig)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input_fig, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = True

    fig_flatten = TimeDistributed(Flatten())(x)
    fig_dropout_1 = Dropout(0.4)(fig_flatten)

    fig_dense = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))(fig_dropout_1)
    fig_atten = TextFigAtten()([sentenceGRU, fig_dense])
    fig_dense_1 = Dense(64,activation='sigmoid')(fig_atten)
    fig_dropout_2 = Dropout(0.4)(fig_dense_1)
    fig_dense_2 = Dense(20,activation='sigmoid')(fig_dropout_2)
    y_fig = Dense(1, activation='sigmoid',name='y_fig')(fig_dense_2)


    # concatenate
    merge = concatenate([sentenceGRU, fig_dense_1])#64+64
    merge_dropout = Dropout(0.4)(merge)
    merge_dense = Dense(20, activation='sigmoid')(merge_dropout)


    y_merge = Dense(1, activation='sigmoid', name='y_merge')(merge_dense)
    model = Model(inputs=[input_text, input_fig], outputs=[y_text, y_fig, y_merge])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model





####################### 加senti ########################

def buildModel_text_senti_fig_regu_finetune_readweights_attention():
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE, name='word_embedding'))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True, name='word_multihead'))(wordsEmbedding)


    input_senti = Input(shape=(sentence_timestep, MAXLEN, 1), dtype='float32', name='senti_input')
    wordsSenti = concatenate([wordAtten, input_senti], axis=3)


    wordGRU = TimeDistributed(GRU(301,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  name='word_gru'))(wordsSenti)
    wordDense = TimeDistributed(Dense(150, name='word_dense'))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True,name='sentence_multihead')(wordDense)
    sentenceGRU = GRU(64,name='sentence_gru')(sentenceAtten)

    text_model = Model([input_text, input_senti], sentenceGRU)
    text_model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                         'text_senti', 'model-epoch_02.hdf5'),
                            by_name=True)
    text_model.trainable = True


    sentenceDense = Dense(20)(sentenceGRU)
    y_text = Dense(1, activation="sigmoid", name="y_text")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input_fig)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input_fig, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = True

    fig_flatten = TimeDistributed(Flatten())(x)
    fig_dropout_1 = Dropout(0.4)(fig_flatten)

    fig_dense = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))(fig_dropout_1)
    fig_atten = TextFigAtten()([sentenceGRU, fig_dense])
    fig_dense_1 = Dense(64,activation='sigmoid')(fig_atten)
    fig_dropout_2 = Dropout(0.4)(fig_dense_1)
    fig_dense_2 = Dense(20,activation='sigmoid')(fig_dropout_2)
    y_fig = Dense(1, activation='sigmoid',name='y_fig')(fig_dense_2)


    # concatenate
    merge = concatenate([sentenceGRU, fig_dense_1])#64+64
    merge_dropout = Dropout(0.4)(merge)
    merge_dense = Dense(20, activation='sigmoid')(merge_dropout)


    y_merge = Dense(1, activation='sigmoid', name='y_merge')(merge_dense)
    model = Model(inputs=[input_text, input_senti, input_fig], outputs=[y_text, y_fig, y_merge])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy', auc])
    return model



def buildModel_text_senti_fig_regu_finetune_readweights_attention_again():
    ########### 因为上一个模型（text+senti_fig）还没收敛，继续训练！！！！！！！！！！！！！！！！！！！！！！！！




    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE, name='word_embedding'))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True, name='word_multihead'))(wordsEmbedding)


    input_senti = Input(shape=(sentence_timestep, MAXLEN, 1), dtype='float32', name='senti_input')
    wordsSenti = concatenate([wordAtten, input_senti], axis=3)


    wordGRU = TimeDistributed(GRU(301,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  name='word_gru'))(wordsSenti)
    wordDense = TimeDistributed(Dense(150, name='word_dense'))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True,name='sentence_multihead')(wordDense)
    sentenceGRU = GRU(64,name='sentence_gru')(sentenceAtten)

    # text_model = Model([input_text, input_senti], sentenceGRU)
    # text_model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
    #                                      'text_senti', 'model-epoch_02.hdf5'),
    #                         by_name=True)
    # text_model.trainable = True


    sentenceDense = Dense(20)(sentenceGRU)
    y_text = Dense(1, activation="sigmoid", name="y_text")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input_fig)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input_fig, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = True

    fig_flatten = TimeDistributed(Flatten())(x)
    fig_dropout_1 = Dropout(0.4)(fig_flatten)

    fig_dense = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))(fig_dropout_1)
    fig_atten = TextFigAtten()([sentenceGRU, fig_dense])
    fig_dense_1 = Dense(64,activation='sigmoid')(fig_atten)
    fig_dropout_2 = Dropout(0.4)(fig_dense_1)
    fig_dense_2 = Dense(20,activation='sigmoid')(fig_dropout_2)
    y_fig = Dense(1, activation='sigmoid',name='y_fig')(fig_dense_2)


    # concatenate
    merge = concatenate([sentenceGRU, fig_dense_1])#64+64
    merge_dropout = Dropout(0.4)(merge)
    merge_dense = Dense(20, activation='sigmoid')(merge_dropout)


    y_merge = Dense(1, activation='sigmoid', name='y_merge')(merge_dense)
    model = Model(inputs=[input_text, input_senti, input_fig], outputs=[y_text, y_fig, y_merge])

    ##### load训练了20轮的参数
    model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                         'text_senti_fig', 'model-epoch_20.hdf5'), by_name=True)
    model.trainable = True


    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model




def buildModel_text_fig_regu_finetune_readweights_4output_attention():
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE, name='word_embedding'))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True, name='word_multihead'))(wordsEmbedding)
    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  name='word_gru'))(wordAtten)

    wordDense = TimeDistributed(Dense(150, name='word_dense'))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True,name='sentence_multihead')(wordDense)
    sentenceGRU = GRU(64,name='sentence_gru')(sentenceAtten)
    sentenceDense = Dense(20)(sentenceGRU)

    y_text = Dense(1, activation="sigmoid", name="y_text")(sentenceDense)

    text_model = Model(input_text, y_text)
    text_model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                         'only_text_ckp', 'model-epoch_08-val_acc_0.80.hdf5'),
                            by_name=True)
    text_model.trainable = True



    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input_fig)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input_fig, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = True

    fig_flatten = TimeDistributed(Flatten())(x)
    fig_dropout_1 = Dropout(0.4)(fig_flatten)

    fig_dense = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))(fig_dropout_1)
    fig_atten = TextFigAtten()([sentenceGRU, fig_dense])
    fig_dense_1 = Dense(64,activation='sigmoid')(fig_atten)
    fig_dropout_2 = Dropout(0.4)(fig_dense_1)
    fig_dense_2 = Dense(20,activation='sigmoid')(fig_dropout_2)
    y_fig = Dense(1, activation='sigmoid',name='y_fig')(fig_dense_2)


    # concatenate
    merge = concatenate([sentenceGRU, fig_dense_1])#64+64
    merge_dropout = Dropout(0.4)(merge)
    merge_dense = Dense(20, activation='sigmoid')(merge_dropout)


    y_merge = Dense(1, activation='sigmoid', name='y_merge')(merge_dense)


    # vote
    vote = concatenate([y_text, y_fig, y_merge])
    y_vote = Vote()(vote)

    model = Model(inputs=[input_text, input_fig], outputs=[y_text, y_fig, y_merge, y_vote])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model



def buildModel_text():
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True))(wordsEmbedding)
    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  ))(wordAtten)
    wordDense = TimeDistributed(Dense(150))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True)(wordDense)
    sentenceGRU = GRU(64)(sentenceAtten)
    sentenceDense = Dense(20)(sentenceGRU)
    y = Dense(1, activation="sigmoid", name="documentOut")(sentenceDense)

    model = Model(inputs=[input_text], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', auc])
    return model


def buildModel_supplement_han():
    ############# 纯han模型 ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(input_text)

    wordGRU = TimeDistributed(GRU(300, return_sequences=True))(wordsEmbedding)
    wordAtten = TimeDistributed(AttLayer(64))(wordGRU)

    sentenceGRU = GRU(64, return_sequences=True)(wordAtten)
    sentenceAtten = AttLayer(64)(sentenceGRU)

    sentenceDense = Dense(20, activation='sigmoid')(sentenceAtten)
    y = Dense(1, activation="sigmoid", name="documentOut")(sentenceDense)

    model = Model(inputs=[input_text], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', auc])
    return model



####################### 拼接多层 ########################

def buildModel_text_senti_fig_regu_finetune_readweights_attention_concatmulti():
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE, name='word_embedding'))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True, name='word_multihead'))(wordsEmbedding)


    input_senti = Input(shape=(sentence_timestep, MAXLEN, 1), dtype='float32', name='senti_input')
    wordsSenti = concatenate([wordAtten, input_senti], axis=3)


    wordGRU = TimeDistributed(GRU(301,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  name='word_gru'))(wordsSenti)
    wordDense = TimeDistributed(Dense(150, name='word_dense'))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True,name='sentence_multihead')(wordDense)
    sentenceGRU = GRU(64,name='sentence_gru')(sentenceAtten)

    text_model = Model([input_text, input_senti], sentenceGRU)
    text_model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                         'text_senti', 'model-epoch_02.hdf5'),
                            by_name=True)
    text_model.trainable = True


    sentenceDense = Dense(20)(sentenceGRU)
    y_text = Dense(1, activation="sigmoid", name="y_text")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input_fig)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input_fig, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = True

    fig_flatten = TimeDistributed(Flatten())(x)
    fig_dropout_1 = Dropout(0.4)(fig_flatten)

    fig_dense = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))(fig_dropout_1)
    fig_atten = TextFigAtten()([sentenceGRU, fig_dense])
    fig_dense_1 = Dense(64,activation='sigmoid')(fig_atten)
    fig_dropout_2 = Dropout(0.4)(fig_dense_1)
    fig_dense_2 = Dense(20,activation='sigmoid')(fig_dropout_2)
    y_fig = Dense(1, activation='sigmoid',name='y_fig')(fig_dense_2)


    # concatenate
    merge = concatenate([sentenceGRU, fig_atten, fig_dense_1])#64+64
    merge_dropout = Dropout(0.4)(merge)
    merge_dense = Dense(20, activation='sigmoid')(merge_dropout)
    merge_dense_2 = Dense(20, activation='sigmoid')(merge_dense)


    y_merge = Dense(1, activation='sigmoid', name='y_merge')(merge_dense_2)
    model = Model(inputs=[input_text, input_senti, input_fig], outputs=[y_text, y_fig, y_merge])
    
    
    #### 继续训练 ###########
    # model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp', 'text_senti_fig_concatmulti', 'model-epoch_23+07.hdf5'), by_name=True)
    # model.trainable = True
    
    
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy', auc])
                  
                  ##### load训练了20轮的参数

    return model





def expand_dimension(x):
    return K.expand_dims(x)
    
def squeeze_dim(x):
    return K.squeeze(x, axis=2)

def buildModel_text_senti_fig_regu_finetune_readweights_attention_addcnn():
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE, name='word_embedding'))(input_text)
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True, name='word_multihead'))(wordsEmbedding)


    input_senti = Input(shape=(sentence_timestep, MAXLEN, 1), dtype='float32', name='senti_input')
    wordsSenti = concatenate([wordAtten, input_senti], axis=3)


    wordGRU = TimeDistributed(GRU(301, name='word_gru'))(wordsSenti)
    
    wordExpand = Lambda(expand_dimension)(wordGRU)#####(100,301,1)
    wordCnn = TimeDistributed(Conv1D(128, 301, activation='relu', name='wordcnn_1'))(wordExpand)

    wordCnn = Lambda(squeeze_dim)(wordCnn)
    wordCnn = Lambda(expand_dimension)(wordCnn)
    
    print('wordcnn.shape =>',wordCnn.shape)
    wordPool = TimeDistributed(MaxPooling1D(4, strides=1, name='wordpool_1', padding='same'))(wordCnn)
    print('wordpool.shape', wordPool.shape)
    wordFlat = TimeDistributed(Flatten())(wordPool)
    print('wordFlat.shape =>',wordFlat.shape)
    
    
    
    
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True,name='sentence_multihead')(wordFlat)
    sentenceGRU = GRU(64,name='sentence_gru')(sentenceAtten)

    text_model = Model([input_text, input_senti], sentenceGRU)
    text_model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                         'text_senti_concatmulti', 'model-epoch_02.hdf5'),
                            by_name=True)
    text_model.trainable = True


    sentenceDense = Dense(20)(sentenceGRU)
    y_text = Dense(1, activation="sigmoid", name="y_text")(sentenceDense)


    ############## fig #############
    input_fig = Input(shape=(10,fig_resize_shape,fig_resize_shape,3), dtype='float32', name='fig_input')

    # Block 1
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'))(input_fig)
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))(x)

    # Block 2
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))(x)
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))(x)

    # Block 3
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))(x)
    x = TimeDistributed(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))(x)

    # Block 4
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))(x)

    # Block 5
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))(x)
    x = TimeDistributed(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))(x)
    x = TimeDistributed(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))(x)# (None, 10, 25088)

    model_vgg16 = Model(input_fig, x, name='vgg16')
    model_vgg16.load_weights(WEIGHTS_PATH_NO_TOP)
    model_vgg16.trainable = True

    fig_flatten = TimeDistributed(Flatten())(x)
    fig_dropout_1 = Dropout(0.4)(fig_flatten)

    fig_dense = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.0001)))(fig_dropout_1)
    fig_atten = TextFigAtten()([sentenceGRU, fig_dense])
    fig_dense_1 = Dense(64,activation='sigmoid')(fig_atten)
    fig_dropout_2 = Dropout(0.4)(fig_dense_1)
    fig_dense_2 = Dense(20,activation='sigmoid')(fig_dropout_2)
    y_fig = Dense(1, activation='sigmoid',name='y_fig')(fig_dense_2)


    # concatenate
    merge = concatenate([sentenceGRU, fig_dense_1])#64+64
    merge_dropout = Dropout(0.4)(merge)
    merge_dense = Dense(20, activation='sigmoid')(merge_dropout)


    y_merge = Dense(1, activation='sigmoid', name='y_merge')(merge_dense)
    model = Model(inputs=[input_text, input_senti, input_fig], outputs=[y_text, y_fig, y_merge])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy', auc])
    return model













































def trainModel(model, generator_train, generator_valid, generator_test):
    print('epochs = ', _args.epochs)
    epochs = _args.epochs
    train_steps_per_epoch = int(3000*(1-VALID_RATE)/train_batch_size)
    valid_steps_per_epoch = int(3000*VALID_RATE/train_batch_size)
    test_steps_per_epoch = 1900/test_batch_size
    # train_steps_per_epoch = 1
    # valid_steps_per_epoch = 1
    # test_steps_per_epoch = 1




    # checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'ckp', 'text_senti_fig_rvdropout',
    #                                'model-epoch_{epoch:02d}.hdf5')
    # checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'ckp', 'text_rvmaxlen_fig',
    #                                'model-epoch_{epoch:02d}.hdf5')
    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'ckp', 'text_senti_fig_concatmulti', 'model-epoch_{epoch:02d}.hdf5')


    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1,
                                 save_best_only=False, save_weights_only=True,
                                 mode='max', period=1)
    callback_list = [checkpoint]

    model.fit_generator(generator_train, callbacks=callback_list, steps_per_epoch=train_steps_per_epoch,
                        validation_data=generator_valid, validation_steps=valid_steps_per_epoch,
                        epochs=epochs)


    scores = model.evaluate_generator(generator_test, steps=test_steps_per_epoch)
    print(scores)
    for i in range(len(scores)):
        print(model.metrics_names[i] + ":" + str(scores[i]))



def read_weights_testModel(train, valid, test, train_label, valid_label, test_label):
    corp = _args.corp

    test_steps_per_epoch = 1900/test_batch_size

    checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                        # 'text_senti_fig_rvdropout')
                                        #'text_senti_fig')
                                        'text_senti_fig_concatmulti')
#     list = [
#            'model-epoch_01.hdf5','model-epoch_02.hdf5','model-epoch_03.hdf5','model-epoch_04.hdf5',
#            'model-epoch_05.hdf5','model-epoch_06.hdf5','model-epoch_07.hdf5','model-epoch_08.hdf5',
#            'model-epoch_09.hdf5','model-epoch_10.hdf5','model-epoch_11.hdf5','model-epoch_12.hdf5',
#            'model-epoch_13.hdf5','model-epoch_14.hdf5','model-epoch_15.hdf5','model-epoch_16.hdf5',
#            'model-epoch_17.hdf5','model-epoch_18.hdf5','model-epoch_19.hdf5','model-epoch_20.hdf5',
#            'model-epoch_01+20.hdf5','model-epoch_02+20.hdf5','model-epoch_03+20.hdf5','model-epoch_04+20.hdf5',
#            'model-epoch_05+20.hdf5','model-epoch_06+20.hdf5','model-epoch_07+20.hdf5','model-epoch_08+20.hdf5',
#            'model-epoch_09+20.hdf5','model-epoch_10+20.hdf5','model-epoch_11+20.hdf5','model-epoch_12+20.hdf5',
#            'model-epoch_13+20.hdf5','model-epoch_14+20.hdf5','model-epoch_15+20.hdf5','model-epoch_16+20.hdf5',
#            'model-epoch_17+20.hdf5','model-epoch_18+20.hdf5','model-epoch_19+20.hdf5','model-epoch_20+20.hdf5',
#            ]

    # list = [
    #         'model-epoch_01+20.hdf5','model-epoch_02+20.hdf5','model-epoch_03+20.hdf5','model-epoch_04+20.hdf5',
    #         'model-epoch_05+20.hdf5','model-epoch_06+20.hdf5','model-epoch_07+20.hdf5','model-epoch_08+20.hdf5',
    #         'model-epoch_09+20.hdf5','model-epoch_10+20.hdf5','model-epoch_11+20.hdf5','model-epoch_12+20.hdf5',
    #         'model-epoch_13+20.hdf5','model-epoch_14+20.hdf5','model-epoch_15+20.hdf5','model-epoch_16+20.hdf5',
    #         'model-epoch_17+20.hdf5','model-epoch_18+20.hdf5','model-epoch_19+20.hdf5','model-epoch_20+20.hdf5',
    #         ]
    
    list = [
            'model-epoch_01.hdf5','model-epoch_02.hdf5','model-epoch_03.hdf5','model-epoch_04.hdf5',
            'model-epoch_05.hdf5','model-epoch_06.hdf5','model-epoch_07.hdf5','model-epoch_08.hdf5',
            'model-epoch_09.hdf5','model-epoch_10.hdf5','model-epoch_11.hdf5','model-epoch_12.hdf5',
            'model-epoch_13.hdf5','model-epoch_14.hdf5','model-epoch_15.hdf5','model-epoch_16.hdf5',
            'model-epoch_17.hdf5','model-epoch_18.hdf5','model-epoch_19.hdf5','model-epoch_20.hdf5',
            'model-epoch_21.hdf5','model-epoch_22.hdf5','model-epoch_23.hdf5','model-epoch_24.hdf5',
            'model-epoch_25.hdf5','model-epoch_26.hdf5','model-epoch_27.hdf5','model-epoch_28.hdf5',
            'model-epoch_29.hdf5','model-epoch_30.hdf5','model-epoch_31.hdf5','model-epoch_32.hdf5',
            'model-epoch_33.hdf5','model-epoch_34.hdf5','model-epoch_35.hdf5','model-epoch_36.hdf5',
            'model-epoch_37.hdf5','model-epoch_38.hdf5','model-epoch_39.hdf5','model-epoch_40.hdf5',
            ]
    
    


    # for filename in os.listdir(checkpoint_root_path):
    for filename in list:
        train_label, valid_label, test_label, generator_train, generator_valid, generator_test, generator_predict, generator_predict_label = get_generator(
            train, valid, test, train_label, valid_label, test_label, corp)

        path = checkpoint_root_path+ "/" + filename
        print(path)

        ### 哪个模型测试
        # model = buildModel_fig()
        # model = buildModel_fig_vgg16()
        # model = taka_fig()
        # model = buildModel_fig_vgg16_mergelast()
        # model = buildModel_text_fig()
        # model = buildModel_text()
        # model = buildModel_text_fig_attention()
        # model = buildModel_fig_hue_hist()
        # model = buildModel_text_fig_regu_attention()
        # model = buildModel_text_fig_regu_finetune_attention()
        # model = buildModel_text_fig_regu_finetune_readweights_attention()
        # model = buildModel_text_fig_regu_finetune_readweights_4output_attention()
        # model = buildModel_text_fig_taka()
        # model = buildModel_text_fig_taka_multioutput()

        # model = buildModel_supplement_han()

        # model = buildModel_text_senti_fig_regu_finetune_readweights_attention()

#        model = buildModel_text_senti_fig_regu_finetune_readweights_attention_concatmulti()##拼接多层
        model = buildModel_text_senti_fig_regu_finetune_readweights_attention_addcnn()###加cnn



        model.load_weights(path)


        ###### evaluate
        # scores = model.evaluate_generator(generator_test, steps=test_steps_per_epoch)
        # print(scores)
        # for i in range(len(scores)):
        #     print(model.metrics_names[i] + ":" + str(scores[i]))


        ##用generator获取真实标签
        true_label_list = []
        for i in range(int(test_steps_per_epoch)):
            true_label_list.append(next(generator_predict_label))
        label_true = np.array(true_label_list).reshape(-1, 1)

        # print('label_true.shape=>',label_true.shape)




        # ##### 得到预测结果，适用于1个output的情况
        # prob = model.predict_generator(generator_predict, steps=test_steps_per_epoch)
        #
        # # print('prob.shape', prob.shape)
        # prob = prob[2]
        #
        # print('---------- cal ----------')
        # print('auc:', metrics.roc_auc_score(y_true=label_true, y_score=prob))
        #
        # pred = np.zeros(shape=(len(prob)))
        #
        #
        # for i in range(len(prob)):
        #     if prob[i]>=0.5:
        #         pred[i] = 1
        # print('acc:', metrics.accuracy_score(y_true=label_true, y_pred=pred))
        # print(metrics.confusion_matrix(y_true=label_true, y_pred=pred))
        # print(metrics.classification_report(y_true=label_true, y_pred=pred, digits=6))




        ##### 得到预测结果，适用于多个output的情况
        predict_list = model.predict_generator(generator_predict, steps=test_steps_per_epoch)

        for i in range(len(predict_list)):
            prob = predict_list[i]

            print('---------- cal ----------')
            print('auc:', metrics.roc_auc_score(y_true=label_true, y_score=prob))

            pred = np.zeros(shape=(len(prob)))

            for j in range(len(prob)):
                if prob[j] >= 0.5:
                    pred[j] = 1
            print('acc:', metrics.accuracy_score(y_true=label_true, y_pred=pred))
            print(metrics.confusion_matrix(y_true=label_true, y_pred=pred))
            print(metrics.classification_report(y_true=label_true, y_pred=pred, digits=6))



        prediction_arr = np.array(predict_list).squeeze(axis=2)#squeeze之前为(3, 1900, 1)
        prediction_arr = prediction_arr.T
        save_pred_npy = np.hstack([label_true, prediction_arr])
        file_save_path = path[:-5]+ '_3predictions.npy'
        np.save(file=file_save_path, arr=save_pred_npy)


        del model
        gc.collect()


def read_weights_predict(generator_train, generator_valid, train_label, valid_label):
    train_steps_per_epoch = int(3000*(1-VALID_RATE)/train_batch_size)
    valid_steps_per_epoch = int(3000*VALID_RATE/train_batch_size)
    test_steps_per_epoch = 1900/test_batch_size

    checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                        # 'text_senti_fig_rvdropout')
                                        'text_senti_fig')

    list = [
            'model-epoch_16.hdf5',
            ]

    # for filename in os.listdir(checkpoint_root_path):
    for filename in list:
        path = checkpoint_root_path+ "/" + filename
        print(path)

        ### 哪个模型测试
        # model = buildModel_fig()
        # model = buildModel_fig_vgg16()
        # model = buildModel_fig_vgg16_mergelast()
        # model = buildModel_text_fig()
        # model = buildModel_text()
        # model = buildModel_text_fig_attention()
        # model = buildModel_fig_hue_hist()
        # model = buildModel_text_fig_regu_attention()
        # model = buildModel_text_fig_regu_finetune_attention()
        # model = buildModel_text_fig_regu_finetune_readweights_attention()
        # model = buildModel_text_fig_regu_finetune_readweights_4output_attention()
        # model = buildModel_text_fig_taka()
        # model = buildModel_text_fig_taka_multioutput()

        # model = buildModel_supplement_han()

        # model = buildModel_text_senti_fig_regu_finetune_readweights_attention()

        model = buildModel_text_senti_fig_regu_finetune_readweights_attention_concatmulti()



        model.load_weights(path)



        ## 获取真实标签
        print('train_label.shape',train_label.shape)
        print('valid_label.shape',valid_label.shape)
        label_true = np.vstack([np.array(train_label).reshape(-1,1),
                                np.array(valid_label).reshape(-1,1)])
        print('label_true.shape=>',label_true.shape)


        ##### 得到预测结果
        predict_train = model.predict_generator(generator_train, steps=train_steps_per_epoch, workers=0)
        predict_valid = model.predict_generator(generator_valid, steps=valid_steps_per_epoch, workers=0)
        # predict_train = model.predict_generator(generator_train, steps=5)
        # predict_valid = model.predict_generator(generator_valid, steps=4)
        predict_train_arr = np.array(predict_train).squeeze(axis=2).T
        print('predict_train_arr.shape:', predict_train_arr.shape)
        predict_valid_arr = np.array(predict_valid).squeeze(axis=2).T
        print('predict_valid_arr.shape:', predict_valid_arr.shape)


        predict_arr = np.vstack([predict_train_arr, predict_valid_arr])
        print('predict_list shape:', predict_arr.shape)


        predict_arr = np.hstack([label_true, predict_arr])
        file_save_path = checkpoint_root_path+'/'+filename[:-5]+'-pred_on_trainvalid.npy'
        np.save(file=file_save_path, arr=predict_arr)


        del model
        gc.collect()

def check_pred(dir, index):
    checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                        dir)

    list = [
        # 'model-epoch_01_predictions.npy', 'model-epoch_02_predictions.npy',
        # 'model-epoch_03_predictions.npy', 'model-epoch_04_predictions.npy',
        # 'model-epoch_05_predictions.npy', 'model-epoch_06_predictions.npy',
        # 'model-epoch_07_predictions.npy',
        'model-epoch_08_predictions.npy',
        # 'model-epoch_09_predictions.npy', 'model-epoch_10_predictions.npy',
        # 'model-epoch_11_predictions.npy', 'model-epoch_12_predictions.npy',
        # 'model-epoch_13_predictions.npy', 'model-epoch_14_predictions.npy',
        # 'model-epoch_15_predictions.npy', 'model-epoch_16_predictions.npy',
        # 'model-epoch_17_predictions.npy', 'model-epoch_18_predictions.npy',
        # 'model-epoch_19_predictions.npy', 'model-epoch_20_predictions.npy',
    ]

    # for filename in os.listdir(checkpoint_root_path):
    for filename in list:
        path = checkpoint_root_path + "/" + filename
        print(path)

        data = np.load(path).squeeze(axis=2)
        label = data[:, 0]
        prob = data[:, 1]


        print('---------- cal ----------')
        print('auc:', metrics.roc_auc_score(y_true=label, y_score=prob))

        pred = np.zeros(shape=(len(prob)))


        for i in range(len(prob)):
            if prob[i]>=0.5:
                pred[i] = 1
        print('acc:', metrics.accuracy_score(y_true=label, y_pred=pred))
        print(metrics.confusion_matrix(y_true=label, y_pred=pred))
        print(metrics.classification_report(y_true=label, y_pred=pred, digits=6))



def main():
    mode = _args.mode
    corp = _args.corp
    dir = _args.dir
    index = _args.index
    train, valid, test, train_label, valid_label, test_label = read_data()
    train_label, valid_label, test_label, generator_train, generator_valid, generator_test, generator_predict, generator_predict_label = get_generator(train, valid, test, train_label, valid_label, test_label, corp)

    if mode == 'train':
        # model = buildModel_fig()
        # model = buildModel_fig_vgg16()
        # model = buildModel_fig_vgg16_mergelast()
        # model = buildModel_text_fig()
        # model = buildModel_text()
        # model = buildModel_text_fig_attention()
        # model = buildModel_fig_hue_hist()
        # model = buildModel_text_fig_regu_attention()
        # model = buildModel_text_fig_regu_finetune_attention()
        # model = buildModel_text_fig_regu_finetune_readweights_attention()
        # model = buildModel_text_fig_regu_finetune_readweights_4output_attention()
        # model = buildModel_text_fig_taka()
        # model = buildModel_text_fig_taka_multioutput()

        # model = buildModel_supplement_han()

        # model = buildModel_text_senti_fig_regu_finetune_readweights_attention()

        model = buildModel_text_senti_fig_regu_finetune_readweights_attention_addcnn()##拼接多层



        ###### 训练了20轮之后继续训练
        # model = buildModel_text_senti_fig_regu_finetune_readweights_attention_again()

        trainModel(model, generator_train, generator_valid, generator_test)

    elif mode == 'test':
        # testModel(generator_test)
        read_weights_testModel(train, valid, test, train_label, valid_label, test_label)
    elif mode == 'predict':
        read_weights_predict(generator_train, generator_valid, train_label, valid_label)
    elif mode == 'check_pred':
        check_pred(dir, index)




if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='run model...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument('--epochs', type=int, required=True, metavar='INTEGER')
    _argparser.add_argument('--mode', type=str, required=True)
    _argparser.add_argument('--corp', type=str, required=True)
    _argparser.add_argument('--dir', type=str, required=True)
    _argparser.add_argument('--index', type=int, required=True)

    _args = _argparser.parse_args()
    main()
