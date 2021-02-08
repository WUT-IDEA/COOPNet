import re
import os
import pandas as pd
import numpy as np
import nltk
from nltk import SnowballStemmer
import gensim
from gensim.models import word2vec, KeyedVectors
from mycode.utils import get_text_list, get_fig_path_list

MIN_COUNT = 5
W2V_DIM = 300

train_root_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'train')
test_root_path = os.path.join(os.path.dirname(__file__), '..', 'input', 'test')
train_label_path = os.path.join(os.path.dirname(__file__), '..','input','train','en.txt')
test_label_path = os.path.join(os.path.dirname(__file__), '..','input','test','en.txt')

# ELMO_OPTIONS_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'input', 'elmo_2x4096_512_2048cnn_2xhighway_options.json')
# ELMO_WEIGHT_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'input', 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5')

def save_data():
    train = pd.read_table(train_label_path, sep=':::', header=None, encoding='utf-8', names=['id', 'label'])
    test = pd.read_table(test_label_path, sep=':::', header=None, encoding='utf-8', names=['id', 'label'])
    train['text_list'] = [get_text_list(id, os.path.join(train_root_path, 'text')) for id in train['id']]
    test['text_list'] = [get_text_list(id, os.path.join(test_root_path, 'text')) for id in test['id']]
    train['fig_path_list'] = [get_fig_path_list(id, os.path.join(train_root_path, 'photo')) for id in train['id']]
    test['fig_path_list'] = [get_fig_path_list(id, os.path.join(test_root_path, 'photo')) for id in test['id']]
    pd.to_pickle(train, os.path.join(os.path.dirname(__file__), '..', 'output', 'train.pd'))
    pd.to_pickle(test, os.path.join(os.path.dirname(__file__), '..', 'output', 'test.pd'))

def read_data(train_file_name, test_file_name):
    train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', train_file_name))
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', test_file_name))
    return train, test

def text_preprocess():
    train, test = read_data('train.pd', 'test.pd')
    # train
    train['tidy_text_list'] = train['text_list'].apply(lambda text_list: _remove_pattern(text_list))# 去掉@user,https_link,#
    # train['tidy_text_list'] = train['tidy_text_list'].apply(
    #     lambda text_list: [' '.join([w for w in text.split() if len(w)>3]) for text in text_list])# 去掉少于3个单词的tweets
    # train['tidy_text_list'] = train['tidy_text_list'].apply(lambda text_list: [text.split() for text in text_list])
    train = train.drop(['text_list'], axis=1)

    # test
    test['tidy_text_list'] = test['text_list'].apply(lambda text_list: _remove_pattern(text_list))
    # test['tidy_text_list'] = test['tidy_text_list'].apply(
    #     lambda text_list: [' '.join([w for w in text.split() if len(w)>3]) for text in text_list])
    # test['tidy_text_list'] = test['tidy_text_list'].apply(lambda text_list: [text.split() for text in text_list])
    test = test.drop(['text_list'], axis=1)

    print(train)
    pd.to_pickle(train, os.path.join(os.path.dirname(__file__), '..', 'output', 'train_cleanText.pd'))
    pd.to_pickle(test, os.path.join(os.path.dirname(__file__), '..', 'output', 'test_cleanText.pd'))

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

# def _remove_pattern_char(input_text_list):# 去掉标点符号，数字和特殊字符等，只留下字母内容
#     cleaned_text_list = []
#     for text in input_text_list:
#         text = re.sub(PATTERN_NONCHAR, " ", text)
#         cleaned_text_list.append(text)
#     return cleaned_text_list
#
# def _stemming(input_text_list):# 提取词干
#     cleaned_text_list = []
#     stemmer = nltk.stem.SnowballStemmer('english')
#     for text in input_text_list:
#         stemmed_word_list = []
#         for word in text:
#             stemmed_word_list.append(stemmer.stem(word))
#         cleaned_text_list.append(stemmed_word_list)
#     return cleaned_text_list



# def ELMo_vectorize():
#     from allennlp.commands.elmo import ElmoEmbedder
#
#     train, test = read_data('train_cleanText.pd', 'test_cleanText.pd')
#     elmo = ElmoEmbedder(ELMO_OPTIONS_FILE_PATH, ELMO_WEIGHT_FILE_PATH)
#     '''
#     ELMo使用例子
#     context_tokens = [['I', 'love', 'you', '.'], ['Sorry', ',', 'I', 'don', "'t", 'love', 'you','.']]  # references
#     elmo_embedding = elmo.embed_batch(context_tokens)
#     print(type(elmo_embedding))
#     print(elmo_embedding[1].shape)
#     print(elmo_embedding[1])
#     '''
#
#     为避免train和test的句子词数不一样，将train和test先合并，一起向量化，再分开
    # train_len = len(train)
    # train = pd.concat([train, test], axis=0)
    #
    # words_list_list = train['tidy_text_list'].tolist()
    # new_word_list = []
    # for i in range(len(words_list_list)):
    #     words_list = words_list_list[i]
    #     all_words = []
    #     for list in words_list:
    #         for w in list:
    #             all_words.append(w)
    #     new_word_list.append(all_words)
    # print('finish combination')
    # elmo_embedding = elmo.embed_batch(new_word_list)
    # print(elmo_embedding)
    # print(elmo_embedding.shape)
    #
    # train['elmo_embedding'] = elmo_embedding
    # train = train.drop(['tidy_text_list'], axis=1)
    # print(train)
    # test = train.iloc[train_len:]
    # train = train.iloc[:train_len]
    # pd.to_pickle(train, os.path.join(os.path.dirname(__file__), '..', 'output', 'train_elmo.pd'))
    # pd.to_pickle(test, os.path.join(os.path.dirname(__file__), '..', 'output', 'test_elmo.pd'))

def Word2Vec_vectorization():
    train, test = read_data('train_cleanText.pd', 'test_cleanText.pd')
    model = KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__),'..','input','GoogleNews-vectors-negative300.bin'),
                                              binary=True,unicode_errors='ignore')
    print(model['better'])
    ## 将train和test合并在一起，一起向量化，再分开
    train_len = len(train)
    train = pd.concat([train, test], axis=0)

    words_list_list = train['tidy_text_list'].tolist()
    new_word_list = []
    for i in range(len(words_list_list)):
        words_list = words_list_list[i]
        all_words = []
        for list in words_list:
            for w in list:
                all_words.append(w)
        # print(len(all_words))
        new_word_list.append(all_words)
    print('finish combination')

    MAXLEN = 500
    W2V_DIM = 300
    USER_SUM = len(train)
    vecs = np.zeros([USER_SUM, MAXLEN, W2V_DIM])
    for i in range(len(new_word_list)):
        word_list = new_word_list[i]
        effec_word_count = 0
        for word in word_list:
            # print(word)
            if effec_word_count < MAXLEN:
                try:
                    vec = model[word]
                    vecs[i, effec_word_count, :] = vec
                    effec_word_count = effec_word_count + 1
                except BaseException as e:
                    # print('not found', word)
                    pass
            else:
                break
    np_train = vecs[:train_len, :, :]
    np_test = vecs[train_len:, :, :]
    np.save(os.path.join(os.path.dirname(__file__),'..','output','train_textvec'), np_train)
    np.save(os.path.join(os.path.dirname(__file__),'..','output','test_textvec'), np_test)

def build_word2index_dic():
    train, test = read_data('train_cleanText.pd', 'test_cleanText.pd')

    # 建立 word->index, word->vector的字典，并将文章转变为单词索引的列表，仅限train
    word_count_dic = {}
    for text_list in train['tidy_text_list']:
        for sentence in text_list:
            word_list = sentence.split(" ")
            for word in word_list:
                if word not in word_count_dic:
                    word_count_dic[word] = 1
                else:
                    word_count_dic[word] = word_count_dic[word] + 1

    temp_list = []
    for key, value in word_count_dic.items():
        print(key, value)
        temp_list.append([key, value])
    temp_list.sort(key=lambda pair: pair[1], reverse=True)
    df = pd.DataFrame(temp_list, columns=['word', 'count'])
    df = df[df['count'] > MIN_COUNT]# 除去出现次数少于MIN_COUNT的词
    df['index'] = range(1, len(df)+1)
    print(df)
    pd.to_pickle(df, os.path.join(os.path.dirname(__file__), '..', 'output', 'word2index.pd'))

def build_embedding_matrix():
    df = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'word2index.pd'))

    embedding_matrix = np.zeros([len(df)+1, W2V_DIM])
    w2v_model = KeyedVectors.load_word2vec_format(os.path.join(os.path.dirname(__file__),'..','input','GoogleNews-vectors-negative300.bin'),
                                              binary=True,unicode_errors='ignore')

    for i in range(3):
        word = df['word'][i]
        # word_index_dic[word] = i+1
        try:
            vec = w2v_model[word]
            embedding_matrix[i+1, :] = vec[:]
            print(embedding_matrix)
            print(word, vec)
        except BaseException as e:
            print(word,'NOT FOUND')
            pass
    print(embedding_matrix)
    np.save(os.path.join(os.path.dirname(__file__), '..', 'output', 'embedding_maxtrix'), embedding_matrix)

def transfer_document():
    train, test = read_data('train_cleanText.pd', 'test_cleanText.pd')

    df = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'word2index.pd'))
    word2index_dic = {}
    for i in range(len(df)):
        word = df['word'][i]
        word2index_dic[word] = df['index'][i]
    word2index_dic[' '] = 0

    train['trans_document'] = train['tidy_text_list'].apply(lambda text_list: _text2seq(text_list, word2index_dic))
    test['trans_document'] = test['tidy_text_list'].apply(lambda text_list: _text2seq(text_list, word2index_dic))
    train_seq = np.array(list(train['trans_document']))
    print('train_seq shape: ', train_seq.shape)
    np.save(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_docseq'), train_seq)
    test_seq = np.array(list(test['trans_document']))
    print('test_seq shape: ', test_seq.shape)
    np.save(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_docseq'), test_seq)

def _text2seq(text_list, word2index_dic):
    MAXLEN = 500
    seq_list = []
    for sentence in text_list:
        if len(seq_list)>=MAXLEN:
            break
        for word in sentence.split(" "):
            if len(seq_list) < MAXLEN and word in word2index_dic:
                seq_list.append(word2index_dic[word])
    if len(seq_list) < MAXLEN:
        for _ in range(MAXLEN-len(seq_list)):
            seq_list.append(0)
    return seq_list

def main():
    print('read data...')
    save_data()
    text_preprocess()
    # Word2Vec_vectorization()
    # build_word2index_dic()
    # build_embedding_matrix()
    # transfer_document()


if __name__ == '__main__':
    main()