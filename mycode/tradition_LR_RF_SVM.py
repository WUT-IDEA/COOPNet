import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import warnings

warnings.filterwarnings('ignore')

import re
# import sys
import gc
import argparse
import string
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from keras.models import load_model, Model
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Embedding, TimeDistributed, Dropout
from keras.layers import concatenate, maximum
from keras.layers import Bidirectional, GRU
from keras.layers import Add, BatchNormalization, Activation
from keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers
from sklearn import metrics
from gensim.models import word2vec, KeyedVectors
from nltk import SnowballStemmer
from nltk.corpus import stopwords

# from mycode.attention import MultiHeadAttention, MultiHeadSelfAttention
# from mycode.vgg16_keras import VGG16
# from mycode.buildModel_2 import han_model,transformer_gru_model, transformer_model, transformer_gru_stepSentiment_model
# from mycode.utils import get_fig_path_list, load_image, list2hist
# from mycode.text_plus_fig_attention import TextFigAtten, AttLayer
# from mycode.Merge import ColumnMaximum, ColumnAverage, CrossMaximum, CrossAverage, WeightedVote, Vote
# from mycode.Generator import *


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(12)

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


WORD_VECTOR_DIC_PATH = os.path.join(os.path.dirname(__file__), '..', 'input', 'glove.6B.300d.txt')


def read_data():  # 返回label和图像的generator
    train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train.pd'))
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test.pd'))

    # shuffle
    random_shuffle_index = np.random.permutation(len(train))
    train = train.iloc[random_shuffle_index]
    # test = test.iloc[np.random.permutation(len(test))]## test不需要random shuffle

    train, test, Y_train, Y_test = text_preprocess(train, test)
    #修改'text_list'这一列，移除pattern；新增'seq'这一列：变成sequence

    w2v = read_w2v_dic()
    train_arr = buildMatrix(train, w2v)
    test_arr = buildMatrix(test, w2v)


    Y_train = np.array(Y_train).reshape((len(Y_train), 1))
    Y_test = np.array(Y_test).reshape((len(Y_test), 1))
    print(Y_train.shape)
    print(Y_test.shape)


    train = np.concatenate((train_arr, Y_train), axis=1)
    test = np.concatenate((test_arr, Y_test), axis=1)

    print(train[:5, :3])
    print(test[:5, :3])

    # np.save(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_mlp'), train)
    # np.save(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_mlp'), test)
    #
    # train = np.load(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_mlp.npy'))
    # test = np.load(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_mlp.npy'))

    return train, test



def text_preprocess(train_ori, test_ori):
    print('remove patterns...')
    train_ori['text_list'] = train_ori['text_list'].apply(lambda list: _remove_pattern_2(list))
    test_ori['text_list'] = test_ori['text_list'].apply(lambda list: _remove_pattern_2(list))

    print('join text list...')
    train_text = train_ori['text_list'].apply(lambda list: " ".join(list))
    test_text = test_ori['text_list'].apply(lambda list: " ".join(list))

    print('prepare labels...')
    Y_train = train_ori['label'].apply(lambda gender: 1 if gender == 'male' else 0)
    Y_test = test_ori['label'].apply(lambda gender: 1 if gender == 'male' else 0)

    return train_text, test_text, Y_train, Y_test


def _remove_pattern_2(input_text_list):
    stoplist = read_stopwords()

    cleaned_text_list = []
    for text in input_text_list:
        text = text.translate(string.punctuation)  # Remove puncuation 去除标点
        text = text.lower()  # Convert words to lower case and split them

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
        text = re.sub(r"https://t.co/[A-Za-z]{10}", " ", text)

        text = text.split()

        text = [word for word in text if word not in stoplist]## 在提取词根前清除一次停用词

        stemmer = SnowballStemmer('english')
        stemmed_words = [stemmer.stem(word) for word in text]

        cleanwordlist = [word for word in stemmed_words if word not in stoplist]## 提取词根后，再清除

        text = " ".join(cleanwordlist)

        cleaned_text_list.append(text)
    return cleaned_text_list

def read_stopwords():
    stopwords_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'english')
    with open(stopwords_path) as f:
        list = f.readlines()
        list_rv = [word[:-1] for word in list]## 去掉末尾的\n
    return list_rv


def read_w2v_dic():
    dic = {}

    with open(WORD_VECTOR_DIC_PATH, encoding='utf8') as f:
        for line in f.readlines():
            split = line.split(' ')
            if len(split) != 301:
                print(line)
            else:
                word = split[0]
                # vec = np.zeros((300), dtype='float32')
                vec_list = []
                for i in range(300):
                    vec_list.append(float(split[i + 1]))
                dic[word] = np.asarray(list(vec_list))
    print('w2v dic finished, len:', len(dic))
    # print(dic['door'])
    return dic


def buildMatrix(data, w2v_dic):
    arr = []
    for i in range(len(data)):
        sample = data[i]

        split = sample.split(' ')

        eff_word_count = 0
        word_vec_sum = np.zeros([300, ])
        for word in split:
            if w2v_dic.__contains__(word):
                word_vec_sum += w2v_dic[word]
                eff_word_count += 1
            else:
                pass
        if eff_word_count != 0:
            word_vec_sum /= eff_word_count
            arr.append(word_vec_sum)

        # print('hit rate:%d'%eff_word_count, '/%d'%len(split),'=',eff_word_count/len(split))

    arr = np.array(list(arr))
    print('arr.shape=', arr.shape)
    # arr = arr.reshape((sample_count, 300))

    return arr



def buildModel_MLP():
    input = Input(shape=(300,), dtype='float32')
    x = Dense(150, activation='sigmoid')(input)
    x = Dropout(0.4)(x)
    x = Dense(30, activation='sigmoid')(x)
    y = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input], outputs=[y])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def trainModel(train, test):
    #####MLP
    print('epochs = ', _args.epochs)
    epochs = _args.epochs

    model = buildModel_MLP()### 哪个模型

    # checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'ckp', 'expe_mlp',
    #                                'model-epoch_{epoch:02d}.hdf5')
    #
    # checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1,
    #                              save_best_only=False, save_weights_only=True,
    #                              mode='max', period=1)
    # callback_list = [checkpoint]

    for round in range(epochs):
        model.fit(train[:, :300], train[:, 300], validation_split=0.2,
                  epochs=1, batch_size=64, verbose=2)

        scores = model.evaluate(test[:, :300], test[:, 300])
        print(scores)
        for i in range(len(scores)):
            print(model.metrics_names[i] + ":" + str(scores[i]))

        prob = model.predict(test[:, :300])
        print('auc:', metrics.roc_auc_score(y_true=test[:, 300], y_score=prob))

        # pred = prob
        # pred[pred >= 0.5] = 1
        # pred[pred < 0.5] = 0
        # print('acc:', metrics.accuracy_score(y_true=test[:, 300], y_pred=pred))
        # print(metrics.confusion_matrix(y_true=test[:, 300], y_pred=pred))
        # print(metrics.classification_report(y_true=test[:, 300], y_pred=pred, digits=6))



        fpr, tpr, thresholds = metrics.roc_curve(test[:, 300], prob)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
        print('optimal_threshold:', optimal_threshold)
        pred = prob
        pred[pred >= optimal_threshold] = 1
        pred[pred < optimal_threshold] = 0
        print('acc:', metrics.accuracy_score(y_true=test[:, 300], y_pred=pred))
        print(metrics.confusion_matrix(y_true=test[:, 300], y_pred=pred))
        print(metrics.classification_report(y_true=test[:, 300], y_pred=pred, digits=6))



    # ########## LR
    # model = LogisticRegression(penalty='l2', C=1)
    # model.fit(train[:, :300], train[:, 300])
    # print('Accuarcy of LR Classifier:', model.score(test[:, :300], test[:, 300]))
    #
    # prob = model.predict_proba(test[:, :300])
    # auc = metrics.roc_auc_score(y_true=test[:, 300], y_score=prob[:, 1])
    # print('AUC of LR Classifier:', auc)
    #
    # pred = prob[:, 1]
    # pred[pred >= 0.5] = 1
    # pred[pred < 0.5] = 0
    # print(metrics.accuracy_score(y_true=test[:, 300], y_pred=pred))
    # print(metrics.confusion_matrix(y_true=test[:, 300], y_pred=pred))
    # print(metrics.classification_report(y_true=test[:, 300], y_pred=pred, digits=6))
    #
    # ########## RF
    # model = RandomForestClassifier(max_depth=20, min_samples_split=10, min_samples_leaf=5,
    #                                max_features=210, max_leaf_nodes=150
    #                                )
    # model.fit(train[:, :300], train[:, 300])
    # print('Accuarcy of RF Classifier:', model.score(test[:, :300], test[:, 300]))
    #
    # prob = model.predict_proba(test[:, :300])
    # auc = metrics.roc_auc_score(y_true=test[:, 300], y_score=prob[:, 1])
    # print('AUC of RF Classifier:', auc)
    #
    # pred = prob[:, 1]
    # pred[pred >= 0.5] = 1
    # pred[pred < 0.5] = 0
    # print(metrics.accuracy_score(y_true=test[:, 300], y_pred=pred))
    # print(metrics.confusion_matrix(y_true=test[:, 300], y_pred=pred))
    # print(metrics.classification_report(y_true=test[:, 300], y_pred=pred, digits=6))
    #
    # ######### SVM
    # model = SVC(kernel='linear')
    # model.fit(train[:, :300], train[:, 300])
    # print('Accuarcy of SVM Classifier:', model.score(test[:, :300], test[:, 300]))
    #
    # prob = model.predict(test[:, :300])
    # auc = metrics.roc_auc_score(y_true=test[:, 300], y_score=prob)
    # print('AUC of SVM Classifier:', auc)
    #
    # pred = prob
    # pred[pred >= 0.5] = 1
    # pred[pred < 0.5] = 0
    # print(metrics.accuracy_score(y_true=test[:, 300], y_pred=pred))
    # print(metrics.confusion_matrix(y_true=test[:, 300], y_pred=pred))
    # print(metrics.classification_report(y_true=test[:, 300], y_pred=pred, digits=6))


def read_weights_testModel(test):
    checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                        'expe_mlp')

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

    for filename in os.listdir(checkpoint_root_path):
        # for filename in list:
        path = checkpoint_root_path + "/" + filename
        print(path)

        model = buildModel_MLP()

        model.load_weights(path)

        scores = model.evaluate(test[:, :300], test[:, 300])
        print(scores)
        for i in range(len(scores)):
            print(model.metrics_names[i] + ":" + str(scores[i]))

        prob = model.predict(test[:, :300])
        auc = metrics.roc_auc_score(y_true=test[:, 300], y_score=prob)
        print('AUC of MLP Classifier:', auc)

        del model
        gc.collect()


def main(_args):
    mode = _args.mode

    train, test = read_data()

    if mode == 'train':
        trainModel(train, test)
    elif mode == 'test':
        read_weights_testModel(test)


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='run model...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument('--epochs', type=int, required=True, metavar='INTEGER')
    _argparser.add_argument('--mode', type=str, required=True)
    _args = _argparser.parse_args()

    main(_args)
