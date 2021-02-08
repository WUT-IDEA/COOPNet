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
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Embedding, TimeDistributed, Dropout
from keras.layers import concatenate, maximum
from keras.layers import Bidirectional, GRU
from keras.layers import Add, BatchNormalization, Activation
from keras.layers import Flatten, Dense
from keras import regularizers
from sklearn import metrics

from mycode.attention import MultiHeadAttention, MultiHeadSelfAttention
from mycode.vgg16_keras import VGG16
# from mycode.buildModel_2 import han_model,transformer_gru_model, transformer_model, transformer_gru_stepSentiment_model
from mycode.utils import get_fig_path_list, load_image, list2hist
from mycode.text_plus_fig_attention import TextFigAtten, AttLayer
from mycode.Merge import ColumnMaximum, ColumnAverage, CrossMaximum, CrossAverage, WeightedVote, Vote
from mycode.Generator import *

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
    train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train.pd'))
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test.pd'))

    train, test, tokenizer = text_preprocess(train, test)#修改'text_list'这一列，移除pattern；新增'seq'这一列：变成sequence

    print(train)






    # # 将训练集分成训练集和验证集
    # train_size = int(len(train)*(1-VALID_RATE))
    # valid = train[train_size:]
    # train = train[:train_size]
    #
    # # 取出pd中的label
    # train_label = train['label'].apply(lambda gender: [1] if gender=='male' else [0])
    # valid_label = valid['label'].apply(lambda gender: [1] if gender=='male' else [0])
    # test_label = test['label'].apply(lambda gender: [1] if gender=='male' else [0])



def main():
    read_data()


if __name__ == '__main__':
    # _argparser = argparse.ArgumentParser(
    #     description='run model...',
    #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # _argparser.add_argument('--epochs', type=int, required=True, metavar='INTEGER')
    # _argparser.add_argument('--mode', type=str, required=True)
    # _argparser.add_argument('--corp', type=str, required=True)
    # _args = _argparser.parse_args()
    main()