import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings('ignore')
import re
import gc

import argparse
import string
import pandas as pd
import numpy as np

from keras.preprocessing import text, sequence
from keras.models import load_model
from sklearn import metrics
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate, TimeDistributed, Dropout
from keras.layers import Bidirectional, GRU, LSTM
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers import Conv1D, MaxPool1D
from keras.layers import Add, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers

from mycode.text_plus_fig_attention import TextFigAtten, AttLayer
from mycode.Merge import ColumnMaximum, ColumnAverage, CrossMaximum, CrossAverage
from mycode.utils import get_text_list, get_fig_path_list
from mycode.Generator import taka_generator_text_batch, generator_fig_batch
from mycode.Generator import taka_generator_fig_text_batch, generator_label
from mycode.my_modules import auc


WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), '..', 'input', 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = os.path.join(os.path.dirname(__file__), '..', 'input',
                                   'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

train_root_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'train')
test_root_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'test')
train_label_path = os.path.join(os.path.dirname(__file__), '..','input','train','en.txt')
test_label_path = os.path.join(os.path.dirname(__file__), '..','input','test','en.txt')


MAX_VOCAB_SIZE = 50000
TAKA_GRU_MAXLEN = 1000
EMBED_SIZE = 300
VALID_RATE = 0.2
train_batch_size = 15
test_batch_size = 10
fig_resize_shape = 150

np.random.seed(12)


def save_data():
    train = pd.read_table(train_label_path, sep=':::', header=None, encoding='utf-8', names=['id', 'label'])
    test = pd.read_table(test_label_path, sep=':::', header=None, encoding='utf-8', names=['id', 'label'])
    train['text_list'] = [get_text_list(id, os.path.join(train_root_path, 'text')) for id in train['id']]
    test['text_list'] = [get_text_list(id, os.path.join(test_root_path, 'text')) for id in test['id']]
    train['fig_path_list'] = [get_fig_path_list(id, os.path.join(train_root_path, 'photo')) for id in train['id']]
    test['fig_path_list'] = [get_fig_path_list(id, os.path.join(test_root_path, 'photo')) for id in test['id']]
    pd.to_pickle(train, os.path.join(os.path.dirname(__file__), '..', 'output', 'train.pd'))
    pd.to_pickle(test, os.path.join(os.path.dirname(__file__), '..', 'output', 'test.pd'))

def _read_data(train_file_name, test_file_name):
    train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', train_file_name))
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', test_file_name))
    return train, test


def _remove_pattern_2(input_text_list):

    cleaned_text_list = []
    for text in input_text_list:
        text = text.translate(string.punctuation)# Remove puncuation 去除标点
        text = text.lower()# Convert words to lower case and split them

        # Remove stop words
        # text = text.split()
        # stops = set(stopwords.words("english"))
        # text = [w for w in text if not w in stops and len(w) >= 3]

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

def text_preprocess():

    print('read data...')
    train_ori, test_ori = _read_data('train.pd', 'test.pd')

    print('remove patterns...')
    train_ori['text_list'] = train_ori['text_list'].apply(lambda list: _remove_pattern_2(list))
    test_ori['text_list'] = test_ori['text_list'].apply(lambda list: _remove_pattern_2(list))

    print('shuffle...')
    train_ori = train_ori.iloc[np.random.permutation(len(train_ori))]# 手动shuffle
    test_ori = test_ori.iloc[np.random.permutation(len(test_ori))]

    print('join text list...')
    train_text = train_ori['text_list'].apply(lambda list: " ".join(list))
    test_text = test_ori['text_list'].apply(lambda list: " ".join(list))
    # train_text = train_ori['text_list']
    # test_text = test_ori['text_list']

    print('prepare labels...')
    Y_train = train_ori['label'].apply(lambda gender: 1 if gender=='male' else 0)
    Y_test = test_ori['label'].apply(lambda gender: 1 if gender=='male' else 0)

    print('prepare tokenizer')
    tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)#词汇表最多单词数
    tokenizer.fit_on_texts( list(train_text) + list(test_text) )#Updates internal vocabulary based on a list of texts.



    ############ 因为takahashi用的是gru，所以就不分单词级和句子级了，直接将每个用户的大文本变成序列
    train_text_seq = tokenizer.texts_to_sequences(train_text)
    test_text_seq = tokenizer.texts_to_sequences(test_text)
    train_text_pad_seq = sequence.pad_sequences(train_text_seq, maxlen=TAKA_GRU_MAXLEN)
    test_text_pad_seq = sequence.pad_sequences(test_text_seq, maxlen=TAKA_GRU_MAXLEN)


    print('fit to numpy...')
    X_train = np.array(list(train_text_pad_seq))
    X_test = np.array(list(test_text_pad_seq))
    print(X_train.shape)
    print(X_test.shape)



    train_size = int(len(train_ori)*(1-VALID_RATE))
    X_valid = X_train[train_size:]
    X_train = X_train[:train_size]
    Y_valid = Y_train[train_size:]
    Y_train = Y_train[:train_size]

    valid_ori = train_ori[train_size:]
    train_ori = train_ori[:train_size]


    ############## 产生generator
    train_fig_path_list = train_ori['fig_path_list']
    valid_fig_path_list = valid_ori['fig_path_list']
    test_fig_path_list = test_ori['fig_path_list']
    train_text_seq_list = X_train
    valid_text_seq_list = X_valid
    test_text_seq_list = X_test

    return train_text_seq_list, valid_text_seq_list, test_text_seq_list, \
           train_fig_path_list, valid_fig_path_list, test_fig_path_list, \
           Y_train, Y_valid, Y_test



def get_generator(train_text_seq_list, valid_text_seq_list, test_text_seq_list,
                  train_fig_path_list, valid_fig_path_list, test_fig_path_list,
                  Y_train, Y_valid, Y_test, corp):
    ############ text
    if corp =='text':
        generator_train = taka_generator_text_batch(train_text_seq_list, Y_train, train_batch_size)
        generator_valid = taka_generator_text_batch(valid_text_seq_list, Y_valid, train_batch_size)
        generator_test = taka_generator_text_batch(test_text_seq_list, Y_test, test_batch_size)
        generator_predict = taka_generator_text_batch(test_text_seq_list, Y_test, test_batch_size, nolabel=True)
        generator_predict_label = generator_label(Y_test, test_batch_size)

    ########### fig
    elif corp == 'fig':
        generator_train = generator_fig_batch(train_fig_path_list, Y_train, train_batch_size, fig_resize_shape)
        generator_valid = generator_fig_batch(valid_fig_path_list, Y_valid, train_batch_size, fig_resize_shape)
        generator_test = generator_fig_batch(test_fig_path_list, Y_test, test_batch_size, fig_resize_shape)
        generator_predict = generator_fig_batch(test_fig_path_list, Y_test, test_batch_size, fig_resize_shape, nolabel=True)
        generator_predict_label = generator_label(Y_test, test_batch_size)

    ########## textfig
    elif corp == 'textfig':
        generator_train = taka_generator_fig_text_batch(train_text_seq_list, train_fig_path_list, Y_train, train_batch_size, fig_resize_shape)
        generator_valid = taka_generator_fig_text_batch(valid_text_seq_list, valid_fig_path_list, Y_valid, train_batch_size, fig_resize_shape)
        generator_test = taka_generator_fig_text_batch(test_text_seq_list, test_fig_path_list, Y_test, test_batch_size, fig_resize_shape)
        generator_predict = taka_generator_fig_text_batch(test_text_seq_list, test_fig_path_list, Y_test, test_batch_size, fig_resize_shape, nolabel=True)
        generator_predict_label = generator_label(Y_test, test_batch_size)

    else:
        exit(1)

    return generator_train, generator_valid, generator_test, generator_predict, generator_predict_label



def taka_text():
    input = Input(shape=(TAKA_GRU_MAXLEN,), dtype='int32', name='input_text')
    x = Embedding(MAX_VOCAB_SIZE, EMBED_SIZE)(input)
    x = Bidirectional(GRU(300))(x)
    x = Dropout(0.4)(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(100, activation='sigmoid')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', auc])
    return model


def taka_text_gru():
    input = Input(shape=(TAKA_GRU_MAXLEN,), dtype='int32', name='input_text')
    x = Embedding(MAX_VOCAB_SIZE, EMBED_SIZE)(input)
    x = GRU(600)(x)
    x = Dropout(0.4)(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(100, activation='sigmoid')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def taka_text_bilstm():
    input = Input(shape=(TAKA_GRU_MAXLEN,), dtype='int32', name='input_text')
    x = Embedding(MAX_VOCAB_SIZE, EMBED_SIZE)(input)
    x = Bidirectional(LSTM(300))(x)
    x = Dropout(0.4)(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(100, activation='sigmoid')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', auc])
    return model


def taka_text_lstm():
    input = Input(shape=(TAKA_GRU_MAXLEN,), dtype='int32', name='input_text')
    x = Embedding(MAX_VOCAB_SIZE, EMBED_SIZE)(input)
    x = LSTM(600)(x)
    x = Dropout(0.4)(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(100, activation='sigmoid')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', auc])
    return model


def taka_text_cnn():
    input = Input(shape=(TAKA_GRU_MAXLEN,), dtype='int32', name='input_text')
    x = Embedding(MAX_VOCAB_SIZE, EMBED_SIZE)(input)
    x = Conv1D(8, 5, activation='relu', padding='same')(x)
    x = Conv1D(8, 5, activation='relu', padding='same')(x)
    x = MaxPool1D()(x)
    x = Conv1D(16, 5, activation='relu', padding='same')(x)
    x = Conv1D(16, 5, activation='relu', padding='same')(x)
    x = MaxPool1D()(x)
    x = Flatten()(x)
    x = Dense(400, activation='relu')(x)
    x = Dense(100, activation='sigmoid')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', auc])
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


def taka_textfig():
    ########### text
    input_text = Input(shape=(TAKA_GRU_MAXLEN,), dtype='int32', name='input_text')
    x_text = Embedding(MAX_VOCAB_SIZE, EMBED_SIZE)(input_text)
    x_text = Bidirectional(GRU(300))(x_text)
    x_text = Dropout(0.4)(x_text)
    x_text = Dense(400, activation='relu')(x_text)
    x_text = Dense(100, activation='sigmoid')(x_text)
    # y = Dense(1, activation='sigmoid')(x_text)


    ########### fig
    input_fig = Input(shape=(10, fig_resize_shape,fig_resize_shape,3), dtype='float32',name='input_fig')
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

    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(1000, activation='relu', kernel_regularizer=regularizers.l1(0.00005)))(x)
    x = ColumnAverage()(x)

    x_fig = Dense(100, activation='sigmoid')(x)

    x_merge = CrossMaximum()([x_text, x_fig])
    x_merge = Dense(20, activation='sigmoid')(x_merge)
    y_merge = Dense(1, activation='sigmoid')(x_merge)

    model = Model(inputs=[input_text, input_fig], outputs=[y_merge])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model



def trainModel(generator_train, generator_valid, generator_test, generator_predict, generator_predict_label):
    print('epochs = ', _args.epochs)
    epochs = _args.epochs

    train_steps_per_epoch = int(3000*(1-VALID_RATE)/train_batch_size)
    valid_steps_per_epoch = int(3000*VALID_RATE/train_batch_size)
    test_steps_per_epoch = 1900/test_batch_size


    # model = taka_text()### 哪个模型
    model = taka_fig()
    # model = taka_textfig()
    # model = taka_text_gru()
    # model = taka_text_lstm()
    # model = taka_text_bilstm()
    # model = taka_text_cnn()


    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                   'taka_fig_2',
                                   # 'expe_taka_text_bilstm',
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



    # #### true_label_list中装入真实标签
    # true_label_list = []
    # for i in range(int(test_steps_per_epoch)):
    #     true_label_list.append(next(generator_predict_label))
    # label_true = np.array(true_label_list).reshape(-1, 1)
    #
    # #### prob为预测结果
    # prob = model.predict_generator(generator_predict, steps=test_steps_per_epoch)
    # print(prob.shape)
    #
    # save_pred_npy = np.stack(arrays=[label_true, prob], axis=1)
    # file_save_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
    #                        'expe_taka_text_bilstm', 'pred_%d.npy'%round)
    #
    #
    # print('auc:', metrics.roc_auc_score(y_true=label_true, y_score=prob))
    #
    # pred = prob
    # pred[pred >= 0.5] = 1
    # pred[pred < 0.5] = 0
    #
    # print('acc:', metrics.accuracy_score(y_true=label_true, y_pred=pred))
    # print('auc:', metrics.roc_auc_score(y_true=label_true, y_score=prob))
    # print(metrics.confusion_matrix(y_true=label_true, y_pred=pred))
    # print(metrics.classification_report(y_true=label_true, y_pred=pred, digits=6))
    #
    # np.save(file=file_save_path, arr=save_pred_npy)


def read_weights_testModel(train_text_seq_list, valid_text_seq_list, test_text_seq_list,
                           train_fig_path_list, valid_fig_path_list, test_fig_path_list,
                           Y_train, Y_valid, Y_test):
    corp = _args.corp

    test_steps_per_epoch = 1900 / test_batch_size

    checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                        'taka_fig_2')

    list = [
            # 'model-epoch_01.hdf5','model-epoch_02.hdf5','model-epoch_03.hdf5','model-epoch_04.hdf5',
            # 'model-epoch_05.hdf5','model-epoch_06.hdf5','model-epoch_07.hdf5','model-epoch_08.hdf5',
            # 'model-epoch_09.hdf5','model-epoch_10.hdf5','model-epoch_11.hdf5',
        'model-epoch_12.hdf5',
            'model-epoch_13.hdf5','model-epoch_14.hdf5','model-epoch_15.hdf5','model-epoch_16.hdf5',
            'model-epoch_17.hdf5','model-epoch_18.hdf5','model-epoch_19.hdf5','model-epoch_20.hdf5',
            ]

    # for filename in os.listdir(checkpoint_root_path):
    for filename in list:
        generator_train, generator_valid, generator_test, generator_predict, generator_predict_label = get_generator(
            train_text_seq_list, valid_text_seq_list, test_text_seq_list,
            train_fig_path_list, valid_fig_path_list, test_fig_path_list,
            Y_train, Y_valid, Y_test, corp)


        path = checkpoint_root_path + "/" + filename
        print(path)

        # model = taka_text()
        model = taka_fig()
        # model = taka_textfig()
        # model = taka_text_gru()
        # model = taka_text_lstm()
        # model = taka_text_bilstm()
        # model = taka_text_cnn()

        model.load_weights(path)

        scores = model.evaluate_generator(generator_test, steps=test_steps_per_epoch)
        print(scores)
        for i in range(len(scores)):
            print(model.metrics_names[i] + ":" + str(scores[i]))


        ##用generator获取真实标签
        true_label_list = []
        for i in range(int(test_steps_per_epoch)):
            true_label_list.append(next(generator_predict_label))
        label_true = np.array(true_label_list).reshape(-1, 1)



        ##### 得到预测结果，适用于1个output的情况
        prob = model.predict_generator(generator_predict, steps=test_steps_per_epoch)

        # prob = prob[2]

        print('---------- cal ----------')
        print('auc:', metrics.roc_auc_score(y_true=label_true, y_score=prob))

        pred = np.zeros(shape=(len(prob)))


        for i in range(len(prob)):
            if prob[i]>=0.5:
                pred[i] = 1
        print('acc:', metrics.accuracy_score(y_true=label_true, y_pred=pred))
        print(metrics.confusion_matrix(y_true=label_true, y_pred=pred))
        print(metrics.classification_report(y_true=label_true, y_pred=pred, digits=6))


        del model
        gc.collect()


def read_weights_predict(generator_predict, generator_predict_label, dir):
    test_steps_per_epoch = 1900 / test_batch_size

    checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                        dir)

    list = [
        'model-epoch_01.hdf5',
        'model-epoch_02.hdf5',
        'model-epoch_03.hdf5',
        'model-epoch_04.hdf5',
        'model-epoch_05.hdf5',
        'model-epoch_06.hdf5',
        'model-epoch_07.hdf5',
        'model-epoch_08.hdf5',
        'model-epoch_09.hdf5',
        'model-epoch_10.hdf5',
        'model-epoch_11.hdf5',
        'model-epoch_12.hdf5',
        'model-epoch_13.hdf5',
        'model-epoch_14.hdf5',
        'model-epoch_15.hdf5',
        'model-epoch_16.hdf5',
        'model-epoch_17.hdf5',
        'model-epoch_18.hdf5',
        'model-epoch_19.hdf5',
        'model-epoch_20.hdf5',
    ]

    # for filename in os.listdir(checkpoint_root_path):
    for filename in list:
        path = checkpoint_root_path + "/" + filename
        print(path)

        # model = taka_text()
        model = taka_fig()
        # model = taka_textfig()
        # model = taka_text_gru()
        # model = taka_text_lstm()
        # model = taka_text_bilstm()
        # model = taka_text_cnn()

        model.load_weights(path)


        #### prob为预测结果
        prob = model.predict_generator(generator_predict, steps=test_steps_per_epoch)
        #### true_label_list中装入真实标签
        true_label_list = []
        for i in range(int(test_steps_per_epoch)):
            true_label_list.append(next(generator_predict_label))
        label_true = np.array(true_label_list).reshape(-1, 1)


        print('auc:',metrics.roc_auc_score(y_true=label_true, y_score=prob))

        pred = prob
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        print('acc:',metrics.accuracy_score(y_true=label_true, y_pred=pred))
        # print('auc:',metrics.roc_auc_score(y_true=label_true, y_score=prob))
        print(metrics.confusion_matrix(y_true=label_true, y_pred=pred))
        print(metrics.classification_report(y_true=label_true, y_pred=pred, digits=6))



        save_pred_npy = np.stack(arrays=[label_true, prob], axis=1)
        file_save_path = path[:-5]+'_predictions.npy'
        np.save(file=file_save_path, arr=save_pred_npy)


        del model
        gc.collect()


def check_pred(dir):
    checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                        dir)

    list = [
        'model-epoch_01_predictions.npy', 'model-epoch_02_predictions.npy',
        'model-epoch_03_predictions.npy', 'model-epoch_04_predictions.npy',
        'model-epoch_05_predictions.npy', 'model-epoch_06_predictions.npy',
        'model-epoch_07_predictions.npy', 'model-epoch_08_predictions.npy',
        'model-epoch_09_predictions.npy', 'model-epoch_10_predictions.npy',
        'model-epoch_11_predictions.npy', 'model-epoch_12_predictions.npy',
        'model-epoch_13_predictions.npy', 'model-epoch_14_predictions.npy',
        'model-epoch_15_predictions.npy', 'model-epoch_16_predictions.npy',
        'model-epoch_17_predictions.npy', 'model-epoch_18_predictions.npy',
        'model-epoch_19_predictions.npy', 'model-epoch_20_predictions.npy',
    ]

    # for filename in os.listdir(checkpoint_root_path):
    for filename in list:
        path = checkpoint_root_path + "/" + filename
        print(path)

        data = np.load(path).squeeze(axis=2)
        label = data[:, 0]
        prob = data[:, 1]

        pred = prob
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        print('acc:',metrics.accuracy_score(y_true=label, y_pred=pred))
        print('auc:',metrics.roc_auc_score(y_true=label, y_score=prob))
        print(metrics.confusion_matrix(y_true=label, y_pred=pred))
        print(metrics.classification_report(y_true=label, y_pred=pred, digits=6))


def main(_args):
    corp = _args.corp
    mode = _args.mode
    dir = _args.dir

    # save_data()


    train_text_seq_list, valid_text_seq_list, test_text_seq_list, \
    train_fig_path_list, valid_fig_path_list, test_fig_path_list, \
    Y_train, Y_valid, Y_test = text_preprocess()

    generator_train, generator_valid, generator_test, generator_predict, generator_predict_label = get_generator(
        train_text_seq_list, valid_text_seq_list, test_text_seq_list,
        train_fig_path_list, valid_fig_path_list, test_fig_path_list,
        Y_train, Y_valid, Y_test, corp)
    if mode == 'train':
        trainModel(generator_train, generator_valid, generator_test, generator_predict, generator_predict_label)
    elif mode == 'test':
        read_weights_testModel(train_text_seq_list, valid_text_seq_list, test_text_seq_list,
                               train_fig_path_list, valid_fig_path_list, test_fig_path_list,
                               Y_train, Y_valid, Y_test)
    elif mode == 'predict':
        read_weights_predict(generator_predict, generator_predict_label, dir)
    elif mode == 'check_pred':
        check_pred(dir)




if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='run model...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument('--epochs', type=int, required=True, metavar='INTEGER')
    _argparser.add_argument('--mode', type=str, required=True)
    _argparser.add_argument('--corp', type=str, required=True)
    _argparser.add_argument('--dir', type=str, required=True)

    _args = _argparser.parse_args()

    main(_args)

