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
import json
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
from keras import backend as K
from skimage.color import rgb2hsv

from mycode.attention import MultiHeadAttention, MultiHeadSelfAttention
from mycode.vgg16_keras import VGG16
# from mycode.buildModel_2 import han_model,transformer_gru_model, transformer_model, transformer_gru_stepSentiment_model
from mycode.utils import get_fig_path_list, load_image, list2hist
from mycode.text_plus_fig_attention import TextFigAtten, AttLayer
from mycode.Merge import ColumnMaximum, ColumnAverage, CrossMaximum, CrossAverage, WeightedVote, Vote
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

# wordindex_polarity_dic = {}
SENTI_DIC_PATH = os.path.join(os.path.dirname(__file__), '..', 'output', 'wordindex_polarity_dic.json')


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

def buildSentiDict(word_index):
    wordindex_polarity_dic = {}# 字典的key是word_index中的数字，value是对应的情感极性

    with open(os.path.join(os.path.dirname(__file__), '..', 'input', 'subjclueslen1-HLTEMNLP05.txt')) as f:
        for line in f.readlines():
            split = line.split(" ")
            word = split[2]
            word = word[6:]
            wordindex = -1
            if word_index.__contains__(word):# 如果语料库中出现了这个词，才将这个词放到情感字典中
                wordindex = word_index[word]

                polarity = split[len(split) - 1]
                if polarity.find('neg') != -1:
                    polarity = -1.0
                elif polarity.find('pos') != -1:
                    polarity = 1.0
                elif polarity.find('both') != -1:
                    polarity = 0.0
                elif polarity.find('neu') != -1:
                    polarity = 0.0
                else:
                    print('sentiment polarity error')

                ## 填充 word_polarity_dic
                if not wordindex_polarity_dic.__contains__(wordindex):
                    wordindex_polarity_dic[wordindex] = polarity

            else:# 如果语料库中没有出现这个词，则不管
                pass

    string = json.dumps(wordindex_polarity_dic)
    with open(SENTI_DIC_PATH, 'w') as f:
        f.write(string)
        f.flush()


def loadSentiDict():
    dic = json.load(open(SENTI_DIC_PATH))
    return dic


def read_data(corp):#返回label和图像的generator
    # train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train.pd'))
    # test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test.pd'))
    #
    # train, test, tokenizer = text_preprocess(train, test)#修改'text_list'这一列，移除pattern；新增'seq'这一列：变成sequence
    #
    # # buildSentiDict(tokenizer.word_index)#填充到wordindex_polarity_dic中，全局变量
    #
    # pd.to_pickle(train, os.path.join(os.path.dirname(__file__), '..', 'output', 'train_rmpattern_seq.pd'))
    # pd.to_pickle(test, os.path.join(os.path.dirname(__file__), '..', 'output', 'test_rmpattern_seq.pd'))
    #
    #
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

    # 4.产生(text, senti, label)的generator
    elif corp == 'textsenti':
        train_text_seq_list = train['seq']
        valid_text_seq_list = valid['seq']
        test_text_seq_list = test['seq']
        generator_train = generator_text_senti_batch(train_text_seq_list, train_label, train_batch_size)
        generator_valid = generator_text_senti_batch(valid_text_seq_list, valid_label, train_batch_size)
        generator_test = generator_text_senti_batch(test_text_seq_list, test_label, test_batch_size)
        generator_predict = generator_text_senti_batch(test_text_seq_list, test_label, test_batch_size, nolabel=True)
        generator_predict_label = generator_label(test_label, test_batch_size)

    else:
        print('corp is wrong: text/fig/textfig...')
        exit(0)

    return train_label, valid_label, test_label, generator_train, generator_valid, generator_test, generator_predict, generator_predict_label



def buildModel_text_senti():

    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(input_text)




    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True))(wordsEmbedding)

    input_senti = Input(shape=(sentence_timestep, MAXLEN, 1), dtype='float32', name='senti_input')
    wordsSenti = concatenate([wordAtten, input_senti], axis=3)


    wordGRU = TimeDistributed(GRU(301,
                                  # kernel_regularizer=regularizers.l1(0.1)
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  ))(wordsSenti)


    wordDense = TimeDistributed(Dense(150))(wordGRU)
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True)(wordDense)
    sentenceGRU = GRU(64)(sentenceAtten)
    sentenceDense = Dense(20)(sentenceGRU)
    y = Dense(1, activation="sigmoid", name="documentOut")(sentenceDense)

    model = Model(inputs=[input_text, input_senti], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', auc])
    return model


def expand_dimension(x):
    return K.expand_dims(x)
    
def squeeze_dim(x):
    return K.squeeze(x, axis=2)


def buildModel_text_senti_addcnn():
    
    ############# text ############
    input_text = Input(shape=(sentence_timestep, MAXLEN,), dtype='int32', name='text_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(input_text)
    
    
    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True))(wordsEmbedding)
    
    input_senti = Input(shape=(sentence_timestep, MAXLEN, 1), dtype='float32', name='senti_input')
    wordsSenti = concatenate([wordAtten, input_senti], axis=3)
    
    
    wordGRU = TimeDistributed(GRU(301,))(wordsSenti)
    
#    wordDense = TimeDistributed(Dense(150))(wordGRU)
    wordExpand = Lambda(expand_dimension)(wordGRU)#####(100,301,1)
    wordCnn = TimeDistributed(Conv1D(128, 301, activation='relu', name='wordcnn_1'))(wordExpand)

    wordCnn = Lambda(squeeze_dim)(wordCnn)
    wordCnn = Lambda(expand_dimension)(wordCnn)
    
    print('wordcnn.shape =>',wordCnn.shape)
    wordPool = TimeDistributed(MaxPooling1D(4, strides=1, name='wordpool_1', padding='same'))(wordCnn)
    print('wordpool.shape', wordPool.shape)
    wordFlat = TimeDistributed(Flatten())(wordPool)
    print('wordFlat.shape =>',wordFlat.shape)
    
    
    
    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True)(wordFlat)
    print('sentenceatten.shape =>',sentenceAtten.shape)
    sentenceGRU = GRU(64)(sentenceAtten)
    print('sentencegru.shape =>',sentenceGRU.shape)
    
    
#    sentenceExpand = Lambda(expand_dimension)(sentenceGRU)#####
#    print('sentenceExpand.shape =>',sentenceExpand.shape)
#    sentenceCnn = Conv1D(5, 40, activation='relu', padding='same', name='sentencecnn_1')(sentenceExpand)
#    sentencePool = MaxPooling1D(2, strides=1, name='wordpool_1')(sentenceCnn)
#    print('sentencePool.shape =>', sentencePool.shape)
#    sentenceFlat = Flatten()(sentencePool)
#    print('sentenceFlat.shape',sentenceFlat.shape)

    sentenceDense = Dense(20)(sentenceGRU)
#    sentenceDense = Dense(20)(sentenceDense)
    y = Dense(1, activation="sigmoid", name="documentOut")(sentenceDense)
                                  
    model = Model(inputs=[input_text, input_senti], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy', auc])
    return model


































def trainModel(model, generator_train, generator_valid, generator_test):
    print('epochs = ', _args.epochs)
    epochs = _args.epochs
    train_steps_per_epoch = int(3000 * (1 - VALID_RATE) / train_batch_size)
    valid_steps_per_epoch = int(3000 * VALID_RATE / train_batch_size)
    test_steps_per_epoch = 1900 / test_batch_size
    # train_steps_per_epoch = 1
    # valid_steps_per_epoch = 1
    # test_steps_per_epoch = 1




    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
#    'text_senti_concatmulti',
    'test_dir',
    'model-epoch_{epoch:02d}.hdf5')

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


def read_weights_testModel(generator_test, generator_predict, generator_predict_label, corp):
    test_steps_per_epoch = 1900 / test_batch_size

    checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                        # 'text_fig_norm_atten_finetune_readweights_drop_multi_ckp_2')
#                                        'text_senti')
                                        'text_senti_concatmulti')

    list = [
         'model-epoch_01.hdf5','model-epoch_02.hdf5','model-epoch_03.hdf5','model-epoch_04.hdf5',
         'model-epoch_05.hdf5','model-epoch_06.hdf5','model-epoch_07.hdf5','model-epoch_08.hdf5',
         'model-epoch_09.hdf5','model-epoch_10.hdf5','model-epoch_11.hdf5','model-epoch_12.hdf5',
         'model-epoch_13.hdf5','model-epoch_14.hdf5','model-epoch_15.hdf5','model-epoch_16.hdf5',
         'model-epoch_17.hdf5','model-epoch_18.hdf5','model-epoch_19.hdf5','model-epoch_20.hdf5',
    ]

    # for filename in os.listdir(checkpoint_root_path):
    for filename in list:
        ##### 重新生成generator
        train_label, valid_label, test_label, generator_train, generator_valid, generator_test, generator_predict, generator_predict_label = read_data(corp)
        
        
        path = checkpoint_root_path + "/" + filename
        print(path)

#        model = buildModel_text_senti()
        model = buildModel_text_senti_addcnn()
        model.load_weights(path)

        scores = model.evaluate_generator(generator_test, steps=test_steps_per_epoch)
        print(scores)
        for i in range(len(scores)):
            print(model.metrics_names[i] + ":" + str(scores[i]))



        prob = model.predict_generator(generator_predict, steps=test_steps_per_epoch)

        true_label_list = []
        for i in range(int(test_steps_per_epoch)):
            true_label_list.append(next(generator_predict_label))
        label_true = np.array(true_label_list).reshape(-1, 1)



        ##### 得到预测结果，适用于1个output的情况
        print('---------- cal ----------')
        print('auc:', metrics.roc_auc_score(y_true=label_true, y_score=prob))

        pred = np.zeros(shape=(len(prob)))


        for i in range(len(prob)):
            if prob[i]>=0.5:
                pred[i] = 1
        print('acc:', metrics.accuracy_score(y_true=label_true, y_pred=pred))
        print(metrics.confusion_matrix(y_true=label_true, y_pred=pred))
        print(metrics.classification_report(y_true=label_true, y_pred=pred, digits=6))



        save_pred_npy = np.stack(arrays=[label_true, prob], axis=1)
        file_save_path = path[:-5]+'_predictions.npy'
        np.save(file=file_save_path, arr=save_pred_npy)



        del model
        gc.collect()


def check_pred(dir, index):
    checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                        dir)

    # list = [
    #     'model-epoch_01_predictions.npy', 'model-epoch_02_predictions.npy',
    #     'model-epoch_03_predictions.npy', 'model-epoch_04_predictions.npy',
    #     'model-epoch_05_predictions.npy', 'model-epoch_06_predictions.npy',
    #     'model-epoch_07_predictions.npy', 'model-epoch_08_predictions.npy',
    #     'model-epoch_09_predictions.npy', 'model-epoch_10_predictions.npy',
    #     'model-epoch_11_predictions.npy', 'model-epoch_12_predictions.npy',
    #     'model-epoch_13_predictions.npy', 'model-epoch_14_predictions.npy',
    #     'model-epoch_15_predictions.npy', 'model-epoch_16_predictions.npy',
    #     'model-epoch_17_predictions.npy', 'model-epoch_18_predictions.npy',
    #     'model-epoch_19_predictions.npy', 'model-epoch_20_predictions.npy',
    # ]

    list = [
        'model-epoch_01_3predictions.npy', 'model-epoch_02_3predictions.npy',
        'model-epoch_03_3predictions.npy', 'model-epoch_04_3predictions.npy',
        'model-epoch_05_3predictions.npy', 'model-epoch_06_3predictions.npy',
        'model-epoch_07_3predictions.npy', 'model-epoch_08_3predictions.npy',
        'model-epoch_09_3predictions.npy', 'model-epoch_10_3predictions.npy',
        'model-epoch_11_3predictions.npy', 'model-epoch_12_3predictions.npy',
        'model-epoch_13_3predictions.npy', 'model-epoch_14_3predictions.npy',
        'model-epoch_15_3predictions.npy', 'model-epoch_16_3predictions.npy',
        'model-epoch_17_3predictions.npy', 'model-epoch_18_3predictions.npy',
        'model-epoch_19_3predictions.npy', 'model-epoch_20_3predictions.npy',
    ]

    # list = [
    #     'model-epoch_01+20_3predictions.npy', 'model-epoch_02+20_3predictions.npy',
    #     'model-epoch_03+20_3predictions.npy', 'model-epoch_04+20_3predictions.npy',
    #     'model-epoch_05+20_3predictions.npy', 'model-epoch_06+20_3predictions.npy',
    #     'model-epoch_07+20_3predictions.npy', 'model-epoch_08+20_3predictions.npy',
    #     'model-epoch_09+20_3predictions.npy', 'model-epoch_10+20_3predictions.npy',
    #     'model-epoch_11+20_3predictions.npy', 'model-epoch_12+20_3predictions.npy',
    #     'model-epoch_13+20_3predictions.npy', 'model-epoch_14+20_3predictions.npy',
    #     'model-epoch_15+20_3predictions.npy', 'model-epoch_16+20_3predictions.npy',
    #     'model-epoch_17+20_3predictions.npy', 'model-epoch_18+20_3predictions.npy',
    #     'model-epoch_19+20_3predictions.npy', 'model-epoch_20+20_3predictions.npy',
    # ]

    # for filename in os.listdir(checkpoint_root_path):
    for filename in list:
        path = checkpoint_root_path + "/" + filename
        print(path)

        data = np.load(path)
        label = data[:, 0]
        prob = data[:, index]


        print('---------- cal ----------')
        print('auc:', metrics.roc_auc_score(y_true=label, y_score=prob))

        pred = np.zeros(shape=(len(prob)))

        for i in range(len(prob)):
            if prob[i] >= 0.5:
                pred[i] = 1
        print('acc:', metrics.accuracy_score(y_true=label, y_pred=pred))
        print(metrics.confusion_matrix(y_true=label, y_pred=pred))
        print(metrics.classification_report(y_true=label, y_pred=pred, digits=6))







def main():
    mode = _args.mode
    corp = _args.corp
    dir = _args.dir
    index = _args.index
    train_label, valid_label, test_label, generator_train, generator_valid, generator_test, generator_predict, generator_predict_label = read_data(corp)




    if mode == 'train':
#        model = buildModel_text_senti()
        model = buildModel_text_senti_addcnn()

        trainModel(model, generator_train, generator_valid, generator_test)

    elif mode == 'test':
        # testModel(generator_test)
        read_weights_testModel(generator_test, generator_predict, generator_predict_label, corp)
    elif mode =='check_pred':
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
