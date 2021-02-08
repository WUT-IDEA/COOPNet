# -*- coding:utf-8 -*-
import re
import os
import pandas as pd
import numpy as np




from keras.models import Sequential,Model
from keras.layers import Embedding,Dense,Dropout,Input,LSTM,Conv2D,AveragePooling2D,Flatten,GRU,Bidirectional
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.callbacks import EarlyStopping

# MAX_LEN = 500
# W2V_DIM = 300
# FIG_DIM = 1000
# LABEL_CATEGORIES = 2


class MLP_text():
    def __init__(self):
        train_text, train_fig, train_label, test_text, test_fig, test_label = load_data()
        self.X_train = train_text
        self.Y_train = train_label
        self.X_test = test_text
        self.Y_test = test_label
        self.MAX_LEN = 500
        self.W2V_DIM = 300
        self.FIG_DIM = 1000
        self.LABEL_CATEGORIES = 2
        self.model = None

    def fit(self):
        print('fitting model...')
        inputs = Input(shape=(self.MAX_LEN, self.W2V_DIM))
        x = Bidirectional(GRU(128), merge_mode='sum')(inputs)
        x = Dropout(0.4)(x)
        x = Dense(50, activation='relu')(x)
        y = Dense(self.LABEL_CATEGORIES, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=y)
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.fit(self.X_train, self.Y_train, shuffle=True, epochs=3, batch_size=32,
                  validation_split=0.2, verbose=2)

    def eval(self):
        print('evaluating model...')
        scores = self.model.evaluate(self.X_test, self.Y_test, batch_size=32)

        print(self.model.metrics_names[0] + ":" + str(scores[0]) + "  "
              + self.model.metrics_names[1] + ":" + str(scores[1]) + "  ")


class MLP_fig():
    def __init__(self):
        train_text, train_fig, train_label, test_text, test_fig, test_label = load_data()
        self.X_train = train_fig.reshape([len(train_fig), -1])
        self.Y_train = train_label
        self.X_test = test_fig.reshape([len(test_fig), -1])
        self.Y_test = test_label
        self.MAX_LEN = 500
        self.W2V_DIM = 300
        self.FIG_DIM = 1000*10
        self.LABEL_CATEGORIES = 2
        self.model = None

    def fit(self):
        print('fitting model...')
        inputs = Input(shape=(self.FIG_DIM,))
        x = Dense(200, activation='relu')(inputs)
        x = Dropout(0.4)(x)
        x = Dense(50, activation='relu')(x)
        y = Dense(self.LABEL_CATEGORIES, activation='softmax')(x)
        self.model = Model(inputs=inputs, outputs=y)
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.fit(self.X_train, self.Y_train, shuffle=True, epochs=30, batch_size=32,
                  validation_split=0.2, verbose=2)

    def eval(self):
        print('evaluating model...')
        scores = self.model.evaluate(self.X_test, self.Y_test, batch_size=32)

        print(self.model.metrics_names[0] + ":" + str(scores[0]) + "  "
              + self.model.metrics_names[1] + ":" + str(scores[1]) + "  ")


class MLP_text_embedding():
    def __init__(self):
        train_text, train_fig, train_label, test_text, test_fig, test_label = load_data(embedding=True)
        self.X_train = train_text
        self.Y_train = train_label
        self.X_test = test_text
        self.Y_test = test_label
        self.MAX_LEN = 500
        self.W2V_DIM = 300
        self.FIG_DIM = 1000
        self.LABEL_CATEGORIES = 2
        self.model = None
        self.VOCAB = 26521+1

    def fit(self):
        print('fitting model...')
        inputs = Input(shape=(self.MAX_LEN,))
        x = Embedding(self.VOCAB, 256, input_length=self.MAX_LEN, mask_zero=True)(inputs)
        x = Bidirectional(GRU(128), merge_mode='sum')(x)
        x = Dense(100, activation='sigmoid')(x)
        x = Dropout(0.4)(x)
        x = Dense(50, activation='sigmoid')(x)
        y = Dense(self.LABEL_CATEGORIES, activation='sigmoid')(x)
        self.model = Model(inputs=inputs, outputs=y)
        self.model.summary()

        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        self.model.fit(self.X_train, self.Y_train, shuffle=True, epochs=5, batch_size=32,
                  validation_split=0.2, verbose=2)

    def eval(self):
        print('evaluating model...')
        scores = self.model.evaluate(self.X_test, self.Y_test, batch_size=32)

        print(self.model.metrics_names[0] + ":" + str(scores[0]) + "  "
              + self.model.metrics_names[1] + ":" + str(scores[1]) + "  ")


def load_data(embedding=False):
    print('load data...')
    # train
    if embedding==False:
        train_text = np.load(os.path.join(os.path.dirname(__file__), '..', 'output','train_textvec.npy'))#已经向量化的文本
    else:
        train_text = np.load(os.path.join(os.path.dirname(__file__), '..', 'output','train_docseq.npy'))

    train_fig = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output','train_figfeas.pd'))#
    train_fig['filled_figfeas'] = train_fig['fig_feas'].apply(lambda figfeas: _fill_figfeas(figfeas))
    train_fig_ = np.array(list(train_fig['filled_figfeas']))

    train_label_1 = np.array(train_fig['label'].apply(lambda gender: 1 if gender == 'male' else 0))
    train_label_2 = np.array(train_fig['label'].apply(lambda gender: 1 if gender == 'female' else 0))
    train_label = np.concatenate([train_label_1.reshape(-1,1), train_label_2.reshape(-1,1)], axis=1)
    #label index is (ismale, isfemale), if this user belongs to male, its label is (1,0)

    # print train shape
    print('train_text shape: ', train_text.shape)
    print('train_fig shape: ', train_fig_.shape)
    print('train_label shape: ', train_label.shape)

    # test
    if embedding==False:
        test_text = np.load(os.path.join(os.path.dirname(__file__), '..', 'output','test_textvec.npy'))
    else:
        test_text = np.load(os.path.join(os.path.dirname(__file__), '..', 'output','test_docseq.npy'))

    test_fig = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output','test_figfeas.pd'))
    test_fig['filled_figfeas'] = test_fig['fig_feas'].apply(lambda figfeas: _fill_figfeas(figfeas))
    test_fig_ = np.array(list(test_fig['filled_figfeas']))

    test_label_1 = np.array(test_fig['label'].apply(lambda gender: 1 if gender == 'male' else 0))
    test_label_2 = np.array(test_fig['label'].apply(lambda gender: 1 if gender == 'female' else 0))
    test_label = np.concatenate([test_label_1.reshape(-1,1), test_label_2.reshape(-1,1)], axis=1)

    print('test_text shape: ', test_text.shape)
    print('test_fig shape: ', test_fig_.shape)
    print('test_label shape: ', test_label.shape)
    return train_text, train_fig_, train_label, test_text, test_fig_, test_label

def _fill_figfeas(figfeas):
    if len(figfeas)==10:
        return figfeas
    else:
        length = len(figfeas)
        fill_mat = np.zeros(((10-length), 1000))
        conc = np.concatenate([figfeas, fill_mat], axis=0)
        return conc


def main():
    mlp_model = MLP_text_embedding()
    mlp_model.fit()
    mlp_model.eval()

if __name__ == '__main__':
    main()