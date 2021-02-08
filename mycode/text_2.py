# -*- coding:utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import os
import re
import sys
import random as rn
import argparse
import string
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import metrics
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate, TimeDistributed, Dropout
from keras.layers import Bidirectional, GRU
from keras.layers import Add, BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint

from mycode.attention import MultiHeadAttention, MultiHeadSelfAttention
from mycode.my_modules import auc
from mycode.buildModel_2 import han_model,transformer_gru_model, transformer_model, transformer_gru_stepSentiment_model


np.random.seed(123)
# rn.seed(12345)
# tf.set_random_seed(1234)

train_root_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'train')
test_root_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'test')
train_label_path = os.path.join(os.path.dirname(__file__), '..','input','train','en.txt')
test_label_path = os.path.join(os.path.dirname(__file__), '..','input','test','en.txt')

MAXLEN = 30
LONG_MAXLEN = 300
MAX_VOCAB_SIZE = 50000
EMBED_SIZE = 300

def _read_data(train_file_name, test_file_name):
    train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', train_file_name))
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', test_file_name))
    return train, test

def _remove_pattern(input_text_list):
    PATTERN_AT = "@[\w]*"
    HTTPS_LINKS = "https://[\w]*\.[\w]*/[\w\d]{,12}"
    NUMBER_SIGN = "#"
    USELESS_CHAR = "[^\w\s0-9\.,:;?!-_'\"\(\)\*]"
    # EMOJI_CHAR = "[\U00010000 -\U0010ffff\uD800 -\uDBFF\uDC00 -\uDFFF]"

    cleaned_text_list = []
    for text in input_text_list:
        text = re.sub(PATTERN_AT, "", text)# 去掉@user
        text = re.sub(HTTPS_LINKS, "", text)# 去掉https链接
        text = re.sub(NUMBER_SIGN, "", text)# 井号的英文是number_sign，去掉井号#
        text = re.sub(USELESS_CHAR, "", text)#去掉mian_char以外的字符，包括奇怪的字符、emoji等，留下字母、数字、主要标点符号

        text = re.sub("\.", " .", text)# 将标点符号和单词分离，单独作为一个符号
        text = re.sub(",", " ,", text)
        text = re.sub(":", " :", text)
        text = re.sub(";", " ;", text)
        text = re.sub("\?", " ?", text)
        text = re.sub("!", " !", text)
        text = re.sub("-", " -", text)
        text = re.sub("_", " _", text)
        text = re.sub("'", " '", text)
        text = re.sub("\"", " \" ", text)
        text = re.sub("\(", "", text)
        text = re.sub("\)", "", text)
        text = re.sub("\*", " * ", text)

        text = text.strip()
        text = text.lower()
        cleaned_text_list.append(text)
    return cleaned_text_list


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
    # train_ori.sample(frac=1).reset_index(drop=True)
    # test_ori.sample(frac=1).reset_index(drop=True)
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

    # train_seq = tokenizer.texts_to_sequences(train_text)#Transforms each text in texts in a sequence of integers
    # test_seq = tokenizer.texts_to_sequences(test_text)
    #
    # X_train = sequence.pad_sequences(train_seq, maxlen=MAXLEN)#Pads each sequence to the same length (length of the longest sequence)
    # X_test = sequence.pad_sequences(test_seq, maxlen=MAXLEN)
    # print(X_train.shape)
    # print(X_test.shape)

    print('texts to sequences...')
    train_ori['seq'] = train_ori['text_list'].apply(lambda list: tokenizer.texts_to_sequences(list))
    test_ori['seq'] = test_ori['text_list'].apply(lambda list: tokenizer.texts_to_sequences(list))

    print('pad sequences...')
    train_ori['seq'] = train_ori['seq'].apply(lambda list: sequence.pad_sequences(list, maxlen=MAXLEN))
    test_ori['seq'] = test_ori['seq'].apply(lambda list: sequence.pad_sequences(list, maxlen=MAXLEN))

    # X_train = sequence.pad_sequences(train_ori['seq'], maxlen=MAXLEN)#Pads each sequence to the same length (length of the longest sequence)
    # X_test = sequence.pad_sequences(test_ori['seq'], maxlen=MAXLEN)

    print('fit to numpy...')
    X_train = np.array(list(train_ori['seq']))
    X_test = np.array(list(test_ori['seq']))

    print(X_train.shape)
    print(X_test.shape)

    return X_train, Y_train, X_test, Y_test, tokenizer

def senti_preprocess():
    train_ori, test_ori = _read_data('train.pd', 'test.pd')

    train_ori['text_list'] = train_ori['text_list'].apply(lambda list: _remove_pattern_2(list))
    test_ori['text_list'] = test_ori['text_list'].apply(lambda list: _remove_pattern_2(list))

    # 把数据的随机shuffle的顺序保存下来，后面senti的数据也做同样的shuffle
    train_random = np.random.permutation(len(train_ori))
    test_random = np.random.permutation(len(test_ori))
    train_ori = train_ori.iloc[train_random]# 手动shuffle
    test_ori = test_ori.iloc[test_random]

    # 将每个用户的推文连起来，主要是为了后面tokenizer.fit_on_texts
    train_text = train_ori['text_list'].apply(lambda list: " ".join(list))
    test_text = test_ori['text_list'].apply(lambda list: " ".join(list))

    # 变换label，构造Y标签数据集
    Y_train = train_ori['label'].apply(lambda gender: 1 if gender=='male' else 0)
    Y_test = test_ori['label'].apply(lambda gender: 1 if gender=='male' else 0)

    # fit tokenizer
    tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)#词汇表最多单词数
    tokenizer.fit_on_texts( list(train_text) + list(test_text) )#Updates internal vocabulary based on a list of texts.

    # 将每个用户的text_list，变成序列，然后变成等长的序列
    train_ori['seq'] = train_ori['text_list'].apply(lambda list: tokenizer.texts_to_sequences(list))
    test_ori['seq'] = test_ori['text_list'].apply(lambda list: tokenizer.texts_to_sequences(list))
    train_ori['seq'] = train_ori['seq'].apply(lambda list: sequence.pad_sequences(list, maxlen=MAXLEN))
    test_ori['seq'] = test_ori['seq'].apply(lambda list: sequence.pad_sequences(list, maxlen=MAXLEN))

    # 将等长的序列变为numpy数据
    X_train = np.array(list(train_ori['seq']))
    X_test = np.array(list(test_ori['seq']))

    print(X_train.shape)
    print(X_test.shape)

    #################### 下面准备senti的数据
    senti = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'senti_ori.pd'))
    senti['text'] = _remove_pattern_2(list(senti['text']))

    senti = senti.iloc[np.random.permutation(len(senti))]  # 手动shuffle

    senti_text_list = list(senti['text'])
    senti_tokenizer = text.Tokenizer(num_words=MAX_VOCAB_SIZE)  # 词汇表最多单词数
    senti_tokenizer.fit_on_texts(senti_text_list + list(train_text) + list(test_text))  # Updates internal vocabulary based on a list of texts.

    senti_text_seq = senti_tokenizer.texts_to_sequences(senti_text_list)  # Transforms each text in texts in a sequence of integers
    senti_train_seq = senti_tokenizer.texts_to_sequences(list(train_text))
    senti_test_seq = senti_tokenizer.texts_to_sequences(list(test_text))

    X_senti = sequence.pad_sequences(senti_text_seq, maxlen=LONG_MAXLEN)  # Pads each sequence to the same length (length of the longest sequence)
    X_senti = np.array(list(X_senti))
    Y_senti = senti['label']
    print(X_senti.shape)
    print(Y_senti.shape)

    senti_train = sequence.pad_sequences(senti_train_seq, maxlen=LONG_MAXLEN)
    senti_test = sequence.pad_sequences(senti_test_seq, maxlen=LONG_MAXLEN)

    senti_train = np.array(list(senti_train))
    senti_test = np.array(list(senti_test))
    print(senti_train.shape)
    print(senti_test.shape)

    return X_train, Y_train, X_test, Y_test, tokenizer, X_senti, Y_senti, senti_train, senti_test


def run_han(_args):
    X_train, Y_train, X_test, Y_test, tokenizer = text_preprocess()

    han = han_model()
    # lstm = load_model('lstm.h5')
    han.summary()

    batch_size = 64
    epochs = _args.epochs

    # early_stopping = EarlyStopping(patience=3, verbose=1, monitor='val_loss', mode='min')
    # model_checkpoint = ModelCheckpoint('./lstm.model', save_best_only=True, verbose=1, monitor='val_loss', mode='min')
    # reduce_lr = ReduceLROnPlateau(factor=0.5, patience=3, min_lr=0.0001, verbose=1)

    hist = han.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                    verbose=True, shuffle=True)
    han.save('./lstm.h5')

    scores = han.evaluate(X_test, Y_test, batch_size=batch_size)
    print(scores)
    print(han.metrics_names[0] + ":" + str(scores[0]) + "  "
          + han.metrics_names[1] + ":" + str(scores[1]) + "  ")

def run_transformer_gru(_args):
    X_train, Y_train, X_test, Y_test, tokenizer = text_preprocess()

    transformer = transformer_gru_model()
    transformer.summary()

    batch_size = 1
    print('epochs = ', _args.epochs)

    epochs = _args.epochs

    hist = transformer.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                    verbose=True, shuffle=False)
    transformer.save('./lstm.h5')

    scores = transformer.evaluate(X_test, Y_test, batch_size=batch_size)
    print(scores)
    print(transformer.metrics_names[0] + ":" + str(scores[0]) + "  "
          + transformer.metrics_names[1] + ":" + str(scores[1]) + "  ")


def run_transformer(_args):
    X_train, Y_train, X_test, Y_test, tokenizer = text_preprocess()

    transformer = transformer_model()
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

def run_transformer_gru_withsenti(_args):
    X_train, Y_train, X_test, Y_test, tokenizer, X_senti, Y_senti, senti_train, senti_test = senti_preprocess()

    batch_size = 32
    print('epochs = ', _args.epochs)
    epochs = _args.epochs


    # 先构造情感模型
    print('train senti model')
    inputs = Input(shape=(LONG_MAXLEN,))
    x = Embedding(MAX_VOCAB_SIZE, EMBED_SIZE)(inputs)
    x = GRU(200)(x)
    x = Dropout(0.4)(x)
    x = Dense(100, activation='sigmoid')(x)
    x = Dense(50, activation='sigmoid', name='mid_out')(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(input=[inputs], output=[y])
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # model.fit(X_senti, Y_senti, batch_size=32, epochs=3, validation_split=0.2)
    # model.save_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
    #                                 'twitter_HSAN_staticsenti','sentiment_lstm_weights.hdf5'))

    model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                    'twitter_HSAN_staticsenti','sentiment_lstm_weights.hdf5'))


    # 获取中间层输出mid_out
    print('get mid model')
    mid_model = Model(inputs=model.input, outputs=model.get_layer('mid_out').output)
    mid_out_train = mid_model.predict(senti_train)
    mid_out_test = mid_model.predict(senti_test)


    # 下面是主要模型
    print('train main model')
    sentence_timestep = 100
    word_timestep = MAXLEN
    wordsInputs = Input(shape=(sentence_timestep, word_timestep,), dtype='int32', name='words_input')
    print('wordsInputs shape, ', wordsInputs.shape)

    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE), input_shape=(10, 16))(wordsInputs)
    print('emb shape, ',wordsEmbedding.shape)


    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True))(wordsEmbedding)
    print('attention shape, ', wordAtten.shape)

    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  ))(wordAtten)
    print('wordGRU shape, ', wordGRU.shape)

    wordDense = TimeDistributed(Dense(150))(wordGRU)
    print('wordDense shape, ', wordDense.shape)


    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True)(wordDense)
    print('documentEmb shape, ',sentenceAtten.shape)

    sentenceGRU = GRU(64)(sentenceAtten)
    print('sentenceGRU shape, ', sentenceGRU.shape)

    sentenceDense = Dense(50)(sentenceGRU)
    print('sentenceDense shape, ',sentenceDense)

    sentiInputs = Input(shape=(50,))# 加入senti的中间层
    concaLayer = concatenate([sentenceDense, sentiInputs])
    # concaLayer = Dense(20, activation='sigmoid')(concaLayer)

    documentOut = Dense(1, activation="sigmoid", name="documentOut")(concaLayer)

    model = Model(input=[wordsInputs, sentiInputs], output=[documentOut])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', auc])# 目前在epochs为10时可以达到80.94%


    ######## 保存模型参数
    # checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'ckp', 'twitter_HSAN_staticsenti',
    #                                'model-epoch_{epoch:02d}.hdf5')
    #
    #
    # checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1,
    #                              save_best_only=False, save_weights_only=True,
    #                              mode='max', period=1)
    #
    #
    # model.fit([X_train, mid_out_train], Y_train, batch_size=batch_size, epochs=epochs,
    #                  validation_split=0.2, verbose=True, shuffle=True, callbacks=[checkpoint])
    #
    # scores = model.evaluate([X_test, mid_out_test], Y_test, batch_size=batch_size)
    # print(scores)
    # for i in range(len(scores)):
    #     print(model.metrics_names[i] + ":" + str(scores[i]))



    ##### evaluate 和 save 预测结果
    checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                        'twitter_HSAN_staticsenti')

    for filename in os.listdir(checkpoint_root_path):
        path = checkpoint_root_path+ "/" + filename
        print(path)

        model.load_weights(path)

        #### evaluate
        scores = model.evaluate([X_test, mid_out_test], Y_test, batch_size=batch_size)
        print(scores)
        for i in range(len(scores)):
            print(model.metrics_names[i] + ":" + str(scores[i]))

        #### predict
        prob = model.predict([X_test, mid_out_test])

        roc_auc = metrics.roc_auc_score(y_true=Y_test, y_score=prob)
        print('AUC of Classifier:', roc_auc)

        save_pred_npy = np.hstack([np.array(list(Y_test)).reshape(-1,1), prob])
        file_save_path = path[:-5] + '_predictions.npy'
        np.save(file=file_save_path, arr=save_pred_npy)


def run_transformer_gru_withsenti_dynamic(_args):
    X_train, Y_train, X_test, Y_test, tokenizer, X_senti, Y_senti, senti_train, senti_test = senti_preprocess()

    batch_size = 32
    print('epochs = ', _args.epochs)
    epochs = _args.epochs


    # 先构造情感模型
    print('train senti model')
    senti_inputs = Input(shape=(LONG_MAXLEN,), name='senti_input')
    x = Embedding(MAX_VOCAB_SIZE, EMBED_SIZE, name='senti_embedding')(senti_inputs)
    x = GRU(200, name='senti_gru')(x)
    x = Dropout(0.4, name='senti_dropout')(x)
    mid_out = Dense(100, activation='sigmoid', name='mid_out')(x)
    x = Dense(50, activation='sigmoid', name='senti_dense_1')(mid_out)
    y = Dense(1, activation='sigmoid', name='senti_output')(x)
    senti_model = Model(input=[senti_inputs], output=[y])
    senti_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # senti_model.fit(X_senti, Y_senti, batch_size=32, epochs=3, validation_split=0.2)
    # senti_model.save_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
    #                                 'twitter_HSAN_dynamicsenti','sentiment_lstm_weights.hdf5'))

    senti_model.load_weights(os.path.join(os.path.dirname(__file__), '..', 'ckp',
                                    'twitter_HSAN_dynamicsenti','sentiment_lstm_weights.hdf5'))
    print('=================>senti_model.weights')
    print(senti_model.get_weights())



    # 下面是主要模型
    print('train main model')
    sentence_timestep = 100
    word_timestep = MAXLEN
    wordsInputs = Input(shape=(sentence_timestep, word_timestep,), dtype='int32', name='words_input')
    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE), input_shape=(10, 16))(wordsInputs)

    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True))(wordsEmbedding)
    wordGRU = TimeDistributed(GRU(300))(wordAtten)
    wordDense = TimeDistributed(Dense(150))(wordGRU)

    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True)(wordDense)
    sentenceGRU = GRU(64)(sentenceAtten)
    sentenceDense = Dense(50)(sentenceGRU)

    concaLayer = concatenate([sentenceDense, mid_out])

    documentOut = Dense(1, activation="sigmoid", name="documentOut")(concaLayer)

    model = Model(input=[wordsInputs, senti_inputs], output=[documentOut])
    # model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy', auc])# 目前在epochs为10时可以达到80.94%


    ######## 保存模型参数
    # checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'ckp', 'twitter_HSAN_dynamicsenti',
    #                                'model-epoch_{epoch:02d}.hdf5')
    #
    # checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=1,
    #                              save_best_only=False, save_weights_only=True,
    #                              mode='max', period=1)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=12, verbose=0, mode='min')

    for round in range(20):
        model.fit([X_train, senti_train], Y_train, batch_size=batch_size, epochs=1,
                  validation_split=0.2, verbose=True, shuffle=True,
                  # callbacks=[early_stopping]
                  )

        ### evaluate
        scores = model.evaluate([X_test, senti_test], Y_test, batch_size=batch_size)
        print(scores)
        for i in range(len(scores)):
            print(model.metrics_names[i] + ":" + str(scores[i]))


        ### predict
        prob = model.predict([X_test, senti_test])

        roc_auc = metrics.roc_auc_score(y_true=Y_test, y_score=prob)
        print('AUC of Classifier:', roc_auc)

        save_pred_npy = np.hstack([np.array(list(Y_test)).reshape(-1,1), prob])
        file_save_path = os.path.join(os.path.dirname(__file__), '..', 'ckp', 'twitter_HSAN_dynamicsenti',
                                   'predictions_%d.npy'%round)
        np.save(file=file_save_path, arr=save_pred_npy)



    ##### evaluate 和 save 预测结果
    # checkpoint_root_path = os.path.join(os.path.dirname(__file__), '..', 'ckp',
    #                                     'twitter_HSAN_dynamicsenti')
    #
    # for filename in os.listdir(checkpoint_root_path):
    #     path = checkpoint_root_path+ "/" + filename
    #     print(path)
    #
    #     model.load_weights(path)
    #     print('=================>model.weights')
    #     print(model.get_weights())
    #
    #     #### evaluate
    #     scores = model.evaluate([X_test, senti_test], Y_test, batch_size=batch_size)
    #     print(scores)
    #     for i in range(len(scores)):
    #         print(model.metrics_names[i] + ":" + str(scores[i]))
    #
    #     #### predict
    #     prob = model.predict([X_test, senti_test])
    #
    #     roc_auc = metrics.roc_auc_score(y_true=Y_test, y_score=prob)
    #     print('AUC of Classifier:', roc_auc)
    #
    #     save_pred_npy = np.hstack([np.array(list(Y_test)).reshape(-1,1), prob])
    #     file_save_path = path[:-5] + '_predictions.npy'
    #     np.save(file=file_save_path, arr=save_pred_npy)



def run_transformer_gru_stepSentiment_model(_args):
    X_train, Y_train, X_test, Y_test, tokenizer = text_preprocess()

    transformer = transformer_gru_stepSentiment_model()
    transformer.summary()

    batch_size = 64
    print('epochs = ', _args.epochs)

    epochs = _args.epochs

    hist = transformer.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2,
                           verbose=True, shuffle=False)
    transformer.save('./lstm.h5')

    scores = transformer.evaluate(X_test, Y_test, batch_size=batch_size)
    print(scores)
    print(transformer.metrics_names[0] + ":" + str(scores[0]) + "  "
          + transformer.metrics_names[1] + ":" + str(scores[1]) + "  ")







if __name__ == '__main__':
    _argparser = argparse.ArgumentParser(
        description='run model...',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _argparser.add_argument('--epochs', type=int, required=True, metavar='INTEGER')
    _args = _argparser.parse_args()

    # run_han(_args)
    # run_transformer_gru(_args)
    # run_transformer(_args)
    # run_transformer_gru_withsenti(_args)
    run_transformer_gru_withsenti_dynamic(_args)
    # run_transformer_gru_stepSentiment_model(_args)