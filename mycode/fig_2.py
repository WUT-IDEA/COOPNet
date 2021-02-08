# -*- coding:utf-8 -*-
import os
import re
import sys
import argparse
import string
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from keras.models import load_model, Model
from keras.layers import Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn import metrics

from mycode.buildModel_2 import fig_GRU, fig_VGG16_GRU
from mycode.vgg16_keras import VGG16

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"


np.random.seed(42)
# rn.seed(12345)
# tf.set_random_seed(1234)

train_root_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'train')
test_root_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'test')
train_label_path = os.path.join(os.path.dirname(__file__), '..','input','train','en.txt')
test_label_path = os.path.join(os.path.dirname(__file__), '..','input','test','en.txt')

MAXLEN = 50
MAX_VOCAB_SIZE = 50000
EMBED_DIZE = 300

def _read_data(train_file_name, test_file_name):
    train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', train_file_name))
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', test_file_name))
    return train, test

def _fill_figfeas(figfeas):
    if len(figfeas)==10:
        return figfeas
    else:
        length = len(figfeas)
        fill_mat = np.zeros(((10-length), 1000))
        conc = np.concatenate([fill_mat, figfeas], axis=0)
        return conc

def get_data():
    train_fig = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output','train_figfeas.pd'))#
    test_fig = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_figfeas.pd'))  #
    train_fig = train_fig.iloc[np.random.permutation(len(train_fig))]#shuffle
    test_fig = test_fig.iloc[np.random.permutation(len(test_fig))]

    train_fig['filled_figfeas'] = train_fig['fig_feas'].apply(lambda figfeas: _fill_figfeas(figfeas))
    train_fig_ = np.array(list(train_fig['filled_figfeas']))

    train_label = np.array(train_fig['label'].apply(lambda gender: 1 if gender == 'male' else 0))



    test_fig['filled_figfeas'] = test_fig['fig_feas'].apply(lambda figfeas: _fill_figfeas(figfeas))
    test_fig_ = np.array(list(test_fig['filled_figfeas']))

    test_label = np.array(test_fig['label'].apply(lambda gender: 1 if gender == 'male' else 0))


    return train_fig_, train_label, test_fig_, test_label

# 这个是将之前提取出的vgg16的最后一层，1000维的特征，10张图片做gru的模型，前面的函数都是为这个模型做准备
def run_fig_GRU(_args):
    X_train, Y_train, X_test, Y_test = get_data()

    transformer = fig_GRU()
    transformer.summary()

    batch_size = 64
    print('epochs = ', _args.epochs)
    epochs = _args.epochs

    hist = transformer.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                    verbose=True, shuffle=True)
    transformer.save('./lstm.h5')

    scores = transformer.evaluate(X_test, Y_test, batch_size=batch_size)
    print(scores)
    print(transformer.metrics_names[0] + ":" + str(scores[0]) + "  "
          + transformer.metrics_names[1] + ":" + str(scores[1]) + "  ")


# 跑10个vgg16
def get_orifig_data():
    train_fig = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output','train_reshapedfig.pd'))#
    test_fig = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_reshapedfig.pd'))#

    # train_fig = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output','train.pd'))#
    # test_fig = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test.pd'))#
    # train_fig['figs_reshaped_arr'] = 0
    # test_fig['figs_reshaped_arr'] = 0


    train_fig = train_fig.iloc[np.random.permutation(len(train_fig))]#shuffle
    test_fig = test_fig.iloc[np.random.permutation(len(test_fig))]

    train_label = np.array(train_fig['label'].apply(lambda gender: 1 if gender == 'male' else 0))
    test_label = np.array(test_fig['label'].apply(lambda gender: 1 if gender == 'male' else 0))

    return train_fig, train_label, test_fig, test_label

def _fill_orifigfeas(figfeas):
    if type(figfeas)==float:
        return np.zeros((10,224,224,3))
    elif len(figfeas)==10:
        return figfeas
    else:
        length = len(figfeas)
        fill_mat = np.zeros(((10-length), 224, 224, 3))
        conc = np.concatenate([fill_mat, figfeas], axis=0)
        return conc

def get_fig_generator(batchsize):
    X_train, Y_train, X_test, Y_test = get_orifig_data()# train和test都已经shuffle过了

    VALIDATION_SPLIT_RATE = 0.2
    TRAIN_SIZE = int(len(X_train)*(1-VALIDATION_SPLIT_RATE))

    train_generator = _get_generator(X_train[:TRAIN_SIZE], Y_train[:TRAIN_SIZE], batchsize)
    valid_generator = _get_generator(X_train[TRAIN_SIZE:], Y_train[TRAIN_SIZE:], batchsize)
    test_generator = _get_generator(X_test, Y_test, batchsize)
    return train_generator, valid_generator, test_generator

def _get_generator(data, label, batchsize):
    count = 0
    while count < len(data):
        batch_data = np.array(list(data['figs_reshaped_arr'][count: count+batchsize].apply(lambda arr: _fill_orifigfeas(arr))))
        batch_label = label[count: count+batchsize]

        count = (count + batchsize) % len(data)
        yield (batch_data, batch_label)


# 这个网络是，10个vgg16，再加上一个gru。前面的函数都是为下面的generator做准备的
def run_fig_vgg16_GRU(_args):
    user_onetime = 2#每次传入2个用户的数据，尽量设置为2400，600，1900都能除的数！
    train_generator, valid_generator, test_generator = get_fig_generator(batchsize=user_onetime)

    vgg16_gru = fig_VGG16_GRU()
    vgg16_gru.summary()


    print('epochs = ', _args.epochs)
    epochs = _args.epochs

    hist = vgg16_gru.fit_generator(train_generator,
                                   steps_per_epoch=3000*0.8/user_onetime,
                                   validation_data=valid_generator,
                                   validation_steps=3000*0.2/user_onetime,
                                   epochs=epochs, verbose=True, shuffle=True)
    vgg16_gru.save('./lstm.h5')

    scores = vgg16_gru.evaluate_generator(test_generator, steps=1900/user_onetime)
    print(scores)
    print(vgg16_gru.metrics_names[0] + ":" + str(scores[0]) + "  "
          + vgg16_gru.metrics_names[1] + ":" + str(scores[1]) + "  ")




# ###############################  finetune vgg16  #####################################
# finetune vgg16时候的生成器。
def get_finetune_train_generator(data, batchsize):
    index = 0
    while index < len(data):
        data_slice = data.iloc[index:index+batchsize]
        data_arr = np.vstack(data_slice['figs_reshaped_arr'].apply(lambda x: x if type(x)!=float else np.random.randn(0,224,224,3)))

        expand_label = []
        label_count = list(data_slice['figs_reshaped_arr'].apply(lambda x: len(x) if type(x)!=float else 0))
        label = list(data_slice['label'].apply(lambda gender: 1 if gender == 'male' else 0))
        for i in range(len(label_count)):
            count = label_count[i]
            expand_label.extend([label[i]] * count)

        index = index + batchsize
        if index >= len(data):
            index = 0
        yield (data_arr, expand_label)


# 准备finetune vgg16的网络
def finetune_vgg16(_args):
    train_fig = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output','train_reshapedfig.pd'))#
    test_fig = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_reshapedfig.pd'))#
    print('read data finished')

    train_fig = train_fig.iloc[np.random.permutation(len(train_fig))]
    test_fig = test_fig.iloc[np.random.permutation(len(test_fig))]
    print('shuffle data finished')

    VALID_SPILIT_RATE = 0.2

    train_generator = get_finetune_train_generator(train_fig[:int(len(train_fig)*VALID_SPILIT_RATE)], batchsize = 3)
    valid_generator = get_finetune_train_generator(train_fig[int(len(train_fig)*VALID_SPILIT_RATE):], batchsize = 3)
    test_generator = get_finetune_train_generator(test_fig, batchsize = 5)
    print('generator finished')

    print('epochs = ', _args.epochs)
    epochs = _args.epochs

    # keras vgg16，载入imagenet的权重
    vgg16 = VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))
    print('load vgg16 finished')

    x = Flatten()(vgg16.output)
    x = Dense(3000, activation='tanh')(x)
    x = Dense(1024, activation='tanh')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[vgg16.input], outputs=[output])
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    # 用数据generator训练模型
    model.fit_generator(train_generator,steps_per_epoch=800, validation_data=valid_generator,
                        validation_steps=200, epochs=epochs)
    model.save_weights('finetune_vgg16.h5')

    scores = model.evaluate_generator(test_generator, steps=380)
    print(scores)
    print(model.metrics_names[0] + ":" + str(scores[0]) + "  "
          + model.metrics_names[1] + ":" + str(scores[1]) + "  ")


if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='run model...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument('--epochs', type=int, required=True, metavar='INTEGER')
    _args = _argparser.parse_args()

    # run_fig_GRU(_args)
    run_fig_vgg16_GRU(_args)
    # finetune_vgg16(_args)