from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import os
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
import pandas as pd
# from bs4 import BeautifulSoup
import itertools
import more_itertools
import numpy as np
import pickle
import time
import math

from gensim.models import Word2Vec
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import string
from collections import defaultdict
from mycode.HAN_pytorch import WordRNN, SentenceRNN

## mark the columns which contains text for classification and target class
col_text = 'text_list'
col_target = 'label'
max_sent_len = 12
max_seq_len = 25
hid_size = 100
embedsize = 200
epoch = 1



def load_data():
    df = pd.read_csv('yelp.csv')
    return df

## creates a 3D list of format paragraph[sentence[word]]
def create3DList(df, col, max_sent_len,max_seq_len):
    # x=[]
    # for docs in df[col].as_matrix():
    #     x1=[]
    #     idx = 0
    #     for seq in "|||".join(re.split("[.?!]", docs)).split("|||"):
    #         x1.append(clean_str(seq,max_sent_len))
    #         if(idx>=max_seq_len-1):
    #             break
    #         idx= idx+1
    #     x.append(x1)

    x = []
    for i in range(len(df)):
        text_list = df[col].iloc[i]
        # print(i,':',text_list)
        split_list = []
        for text in text_list:
            split = clean_str(text, max_sent_len)
            split_list.append(split)
        x.append(split_list)
    return x

def clean_str(string, max_seq_len):
    """
    adapted from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
#     string = BeautifulSoup(string, "lxml").text
    string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
    string = re.sub(r"\"s", " \"s", string)
    string = re.sub(r"\"ve", " \"ve", string)
    string = re.sub(r"n\"t", " n\"t", string)
    string = re.sub(r"\"re", " \"re", string)
    string = re.sub(r"\"d", " \"d", string)
    string = re.sub(r"\"ll", " \"ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    s =string.strip().lower().split(" ")
    if len(s) > max_seq_len:
        return s[0:max_seq_len]
    return s

def read_stopwords():
    stoplist = []
    with open('/users/zhengyunpei/nltk_data/stopwords/english') as f:
        for line in f.readlines():
            stoplist.append(line[:-2])
    return stoplist

## Make the the multiple attention with word vectors.
def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i]
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0).unsqueeze(0)

def train_data(batch_size, review, targets, sent_attn_model, sent_optimizer, criterion):
    state_word = sent_attn_model.init_hidden_word()
    state_sent = sent_attn_model.init_hidden_sent()
    sent_optimizer.zero_grad()

    y_pred, state_sent = sent_attn_model(review, state_sent, state_word)

    #     loss = criterion(y_pred.cuda(), torch.cuda.LongTensor(targets))
    loss = criterion(y_pred, torch.LongTensor(targets))

    max_index = y_pred.max(dim=1)[1]
    #     correct = (max_index == torch.cuda.LongTensor(targets)).sum()
    correct = (max_index == torch.LongTensor(targets)).sum()
    acc = float(correct) / batch_size

    loss.backward()

    sent_optimizer.step()

    #     return loss.data[0],acc
    return loss.item(), acc

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def gen_batch(x,y,batch_size):
    k = random.sample(range(len(x)-1),batch_size)
    x_batch=[]
    y_batch=[]

    for t in k:
        x_batch.append(x[t])
        y_batch.append(y[t])

    return [x_batch,y_batch]


def validation_accuracy(batch_size, x_val, y_val, sent_attn_model):
    acc = []
    val_length = len(x_val)
    for j in range(int(val_length / batch_size)):
        x, y = gen_batch(x_val, y_val, batch_size)
        state_word = sent_attn_model.init_hidden_word()
        state_sent = sent_attn_model.init_hidden_sent()

        y_pred, state_sent = sent_attn_model(x, state_sent, state_word)
        max_index = y_pred.max(dim=1)[1]
        #         correct = (max_index == torch.cuda.LongTensor(y)).sum()
        correct = (max_index == torch.LongTensor(y)).sum()
        acc.append(float(correct) / batch_size)
    return np.mean(acc)


def train_early_stopping(batch_size, x_train, y_train, x_val, y_val, sent_attn_model,
                         sent_attn_optimiser, loss_criterion, num_epoch,
                         print_loss_every=50, code_test=True):
    start = time.time()
    loss_full = []
    loss_epoch = []
    acc_epoch = []
    acc_full = []
    val_acc = []
    epoch_counter = 0
    train_length = len(x_train)
    for i in range(1, num_epoch + 1):
        loss_epoch = []
        acc_epoch = []
        for j in range(int(train_length / batch_size)):
            x, y = gen_batch(x_train, y_train, batch_size)
            loss, acc = train_data(batch_size, x, y, sent_attn_model, sent_attn_optimiser, loss_criterion)
            loss_epoch.append(loss)
            acc_epoch.append(acc)
            if (code_test and j % int(print_loss_every / batch_size) == 0):
                print('Loss at %d paragraphs, %d epoch,(%s) is %f' % (
                j * batch_size, i, timeSince(start), np.mean(loss_epoch)))
                print('Accuracy at %d paragraphs, %d epoch,(%s) is %f' % (
                j * batch_size, i, timeSince(start), np.mean(acc_epoch)))

        loss_full.append(np.mean(loss_epoch))
        acc_full.append(np.mean(acc_epoch))
        torch.save(sent_attn_model.state_dict(), 'sent_attn_model_yelp.pth')
        print('Loss after %d epoch,(%s) is %f' % (i, timeSince(start), np.mean(loss_epoch)))
        print('Train Accuracy after %d epoch,(%s) is %f' % (i, timeSince(start), np.mean(acc_epoch)))

        val_acc.append(validation_accuracy(batch_size, x_val, y_val, sent_attn_model))
        print('Validation Accuracy after %d epoch,(%s) is %f' % (i, timeSince(start), val_acc[-1]))
    return loss_full, acc_full, val_acc

def test_accuracy(batch_size, x_test, y_test, sent_attn_model):
    acc = []
    test_length = len(x_test)
    for j in range(int(test_length / batch_size)):
        x, y = gen_batch(x_test, y_test, batch_size)
        state_word = sent_attn_model.init_hidden_word()
        state_sent = sent_attn_model.init_hidden_sent()

        y_pred, state_sent = sent_attn_model(x, state_sent, state_word)
        max_index = y_pred.max(dim=1)[1]
        # correct = (max_index == torch.cuda.LongTensor(y)).sum()
        correct = (max_index == torch.LongTensor(y)).sum()
        acc.append(float(correct) / batch_size)
    return np.mean(acc)

def divide_data(df):
    length = df.shape[0]
    train_len = int(0.8*length)
    val_len = int(0.1*length)

    train = df[:train_len]
    val = df[train_len:train_len+val_len]
    test = df[train_len+val_len:]

    return train, val, test







if __name__ == '__main__':
    train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train.pd'))
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test.pd'))
    # print(train.head)
    # print(train.columns)

    # shuffle训练集
    train = train.iloc[np.random.permutation(len(train))]

    # 将训练集分成训练集和验证集
    train_size = int(len(train) * 0.9)
    valid = train[train_size:]
    train = train[:train_size]

    # 取出pd中的label
    train_label = train[col_target].apply(lambda gender: [1] if gender == 'male' else [0])
    valid_label = valid[col_target].apply(lambda gender: [1] if gender == 'male' else [0])
    test_label = test[col_target].apply(lambda gender: [1] if gender == 'male' else [0])

    # label的种类
    # cls_arr = np.sort(train_label.unique()).tolist()
    cls_arr = [0,1]
    classes = len(cls_arr)


    ## 构造［batch，sent，word］的三维list
    x_train = create3DList(train, col_text, max_sent_len, max_seq_len)
    x_val = create3DList(valid, col_text, max_sent_len, max_seq_len)
    x_test = create3DList(test, col_text, max_sent_len, max_seq_len)

    print("x_train: {}".format(len(x_train)))
    print("x_val: {}".format(len(x_val)))
    print("x_test: {}".format(len(x_test)))

    print('stem words...')
    stoplist = read_stopwords()
    stemmer = SnowballStemmer('english')
    x_train_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist]
                      for sent in para]
                     for para in x_train]
    x_test_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist]
                     for sent in para]
                    for para in x_test]
    x_val_texts = [[[stemmer.stem(word.lower()) for word in sent if word not in stoplist]
                    for sent in para]
                   for para in x_val]

    ## calculate frequency of words
    print('calculate frequency...')
    frequency1 = defaultdict(int)
    for texts in x_train_texts:
        for text in texts:
            for token in text:
                frequency1[token] += 1
    for texts in x_test_texts:
        for text in texts:
            for token in text:
                frequency1[token] += 1
    for texts in x_val_texts:
        for text in texts:
            for token in text:
                frequency1[token] += 1

    ## remove  words with frequency less than 5.
    print('remove  words with frequency less than 5...')
    x_train_texts = [[[token for token in text if frequency1[token] > 5]
                      for text in texts]
                     for texts in x_train_texts]

    x_test_texts = [[[token for token in text if frequency1[token] > 5]
                     for text in texts]
                    for texts in x_test_texts]

    x_val_texts = [[[token for token in text if frequency1[token] > 5]
                    for text in texts]
                   for texts in x_val_texts]


    texts = list(more_itertools.collapse(x_train_texts[:] + x_test_texts[:] + x_val_texts[:],levels=1))

    ## train word2vec model on all the words
    print('train word2vec model...')
    word2vec = Word2Vec(texts, size=200, min_count=5)
    word2vec.save("dictonary_yelp")


    ## convert 3D text list to 3D list of index
    print('convert 3D text list to 3D list of index...')
    x_train_vec = [[[word2vec.wv.vocab[token].index for token in text]
                    for text in texts]
                   for texts in x_train_texts]

    x_test_vec = [[[word2vec.wv.vocab[token].index for token in text]
                   for text in texts]
                  for texts in x_test_texts]

    x_val_vec = [[[word2vec.wv.vocab[token].index for token in text]
                  for text in texts]
                 for texts in x_val_texts]


    print('build weights...')
    # weights = torch.FloatTensor(word2vec.wv.syn0).cuda()
    weights = torch.FloatTensor(word2vec.wv.syn0)
    vocab_size = len(word2vec.wv.vocab)

    print('label to list..')
    y_train = train_label.tolist()
    y_test = test_label.tolist()
    y_val = valid_label.tolist()

    ## converting list to tensor
    print('converting list to tensor...')
    # y_train_tensor =  [torch.cuda.FloatTensor([cls_arr.index(label)]) for label in y_train]
    # y_val_tensor =  [torch.cuda.FloatTensor([cls_arr.index(label)]) for label in y_val]
    # y_test_tensor =  [torch.cuda.FloatTensor([cls_arr.index(label)]) for label in y_test]

    y_train_tensor =  [torch.FloatTensor(label) for label in y_train]
    y_val_tensor =  [torch.FloatTensor(label) for label in y_val]
    y_test_tensor =  [torch.FloatTensor(label) for label in y_test]


    print('calculate max_seq_len, max_sent_len...')
    max_seq_len = max([len(seq) for seq in itertools.chain.from_iterable(x_train_vec +x_val_vec + x_test_vec)])
    max_sent_len = max([len(sent) for sent in (x_train_vec + x_val_vec + x_test_vec)])

    np.percentile(np.array([len(seq) for seq in itertools.chain.from_iterable(x_train_vec +x_val_vec + x_test_vec)]),90)
    np.percentile(np.array([len(sent) for sent in (x_train_vec +x_val_vec + x_test_vec)]),90)

    ## Padding the input
    print('padding the input')
    X_train_pad = [sub_list + [[0]] * (max_sent_len - len(sub_list)) for sub_list in x_train_vec]
    X_val_pad = [sub_list + [[0]] * (max_sent_len - len(sub_list)) for sub_list in x_val_vec]
    X_test_pad = [sub_list + [[0]] * (max_sent_len - len(sub_list)) for sub_list in x_test_vec]

    print('build model...')
    batch_size = 64
    sent_attn = SentenceRNN(vocab_size, embedsize, batch_size, hid_size, classes)
    # sent_attn.cuda()
    sent_attn.wordRNN.embed.from_pretrained(weights)
    torch.backends.cudnn.benchmark=True


    learning_rate = 1e-3
    momentum = 0.9

    sent_optimizer = torch.optim.SGD(sent_attn.parameters(), lr=learning_rate, momentum= momentum)

    criterion = nn.NLLLoss()

    print('train model...')
    loss_full, acc_full, val_acc = train_early_stopping(batch_size, X_train_pad, y_train_tensor, X_val_pad,
                                    y_val_tensor, sent_attn, sent_optimizer, criterion, epoch, 10000, False)
    acc = test_accuracy(batch_size, X_test_pad, y_test_tensor, sent_attn)
    print('test_acc =',acc)