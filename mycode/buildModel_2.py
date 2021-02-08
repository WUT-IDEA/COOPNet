from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from keras.layers import Add, BatchNormalization, Activation, Dropout
# from keras.layers import *
from keras.layers import Permute,Lambda,RepeatVector,Multiply,GRU,TimeDistributed,Flatten,Concatenate
# from keras.models import *
import numpy as np
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras import regularizers
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
# from keras.applications import VGG16
import gc
from sklearn import metrics
import tensorflow as tf
import random as rn
import keras
from mycode.attention import MultiHeadAttention, MultiHeadSelfAttention
from mycode.vgg16_keras import VGG16

np.random.seed(123)
# rn.seed(12345)
# tf.set_random_seed(1234)


MAXLEN = 300
MAX_VOCAB_SIZE = 50000
EMBED_SIZE = 300


def attention_3d_block(inputs):  # inputs shape=[?, ? , 256]
    # inputs.shape = (batch_size, time_steps, input_dim)
    print(inputs.shape)
    TIME_STEPS = inputs.shape[1]  # None
    SINGLE_ATTENTION_VECTOR = False

    input_dim = int(inputs.shape[2])  # 256
    a = Permute((2, 1))(inputs)  # shape=[?, 256, 70]
    # a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    print(a.shape)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    print(a.shape)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class reset(Layer):
    def __init__(self, rep=32):
        self.rep = rep
        super(reset, self).__init__()

    def call(self, x):
        x = keras.backend.reshape(x, (self.rep, 100, 50, 300))
        return x

    def compute_output_shape(self, input_shape):
        return (self.rep, 100, 50, 300)

def get_embedding_matrix(tokenizer):
    EMBEDDING_FILE = '../input/glove.6B.300d.txt'
    def get_coefs(word,*arr):
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='utf-8'))

    all_embs = np.stack(embeddings_index.values())
    emb_mean= all_embs.mean()  # 总体的平均值，一个数值
    emb_std = all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index  # X_train与X_test中所有单词的编号,编号按频率由高到低
    nb_words = min(MAX_VOCAB_SIZE, len(word_index))
    embedding_matrix_1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= MAX_VOCAB_SIZE:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix_1[i] = embedding_vector

    del embeddings_index; gc.collect()  # 两句结合清理内存

    embedding_matrix = embedding_matrix_1

    return embedding_matrix


class WeightLayer(Layer):
    def __init__(self, rep=32, **kwargs):
        self.rep = rep
        super(WeightLayer, self).__init__(**kwargs)

    def call(self, x):
        x = Permute([2, 1])(x)
        k = K.variable(K.truncated_normal(shape=(20, 50, 300)))
        #返回具有截尾正太分布值的向量，在距离均值两个标准差之外的数据将会被截断并重新生成
        x = K.conv1d(x, k, padding='same')
        x = K.max(x, axis=1)


        # x = keras.backend.permute_dimensions(x, [0, 3, 2, 1])
        return x

    def compute_output_shape(self, input_shape):
        return (self.rep, 1, 300)



class Concate(Layer):
    def __init__(self, rep=32 ,**kwargs):
        self.rep = rep
        super(Concate, self).__init__(**kwargs)

    def call(self, x):
        h0 = K.expand_dims(x[1], axis=2)
        x = K.concatenate([x[0], h0], axis=2)

        return x

    def compute_output_shape(self, input_shape):
        return (self.rep, 100, 51, 300)



def han_model():
    sentence_timestep = 100# 100句
    word_timestep = 30# 每句50个词
    wordsInputs = Input(shape=(sentence_timestep, word_timestep,), dtype='int32', name='words_input')
    print('wordsInputs shape, ', wordsInputs.shape)

    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(wordsInputs)
    print('emb shape, ',wordsEmbedding.shape)

    #
    wordGRU = TimeDistributed(Bidirectional(GRU(128, unroll=True, return_sequences=True), merge_mode='concat'))(wordsEmbedding)
    print('wordRnn shape, ', wordGRU.shape)

    wordAtten = TimeDistributed(AttLayer(64))(wordGRU)
    print('attention shape, ', wordAtten.shape)


    sentenceGRU = Bidirectional(GRU(128, unroll=True, return_sequences=True), merge_mode='concat')(wordAtten)
    print('sentenceRnn shape, ',sentenceGRU.shape)

    sentenceAtten = AttLayer(64)(sentenceGRU)
    print('documentEmb shape, ',sentenceAtten.shape)

    documentOut = Dense(1, activation="sigmoid", name="documentOut")(sentenceAtten)

    model = Model(input=[wordsInputs], output=[documentOut])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])# 这个model在epochs为6时最好，为77.10%

    return model


def transformer_gru_model():
    sentence_timestep = 100
    word_timestep = 50
    wordsInputs = Input(shape=(sentence_timestep, word_timestep,), dtype='int32', name='words_input')
    print('wordsInputs shape, ', wordsInputs.shape)

    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(wordsInputs)
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

    sentenceDense = Dense(20)(sentenceGRU)
    print('sentenceDense shape, ',sentenceDense)


    documentOut = Dense(1, activation="sigmoid", name="documentOut")(sentenceDense)

    model = Model(input=[wordsInputs], output=[documentOut])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])# 目前在epochs为10时可以达到80.94%

    return model


def transformer_gru_stepSentiment_model():
    sentence_timestep = 100
    word_timestep = 50
    wordsInputs = Input(shape=(sentence_timestep, word_timestep,), dtype='int32', name='words_input')
    print('wordsInputs shape:', wordsInputs.shape, type(wordsInputs))

    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(wordsInputs)
    print('wordsEmbedding shape:',wordsEmbedding.shape, type(wordsEmbedding))
    # wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=True))(wordsEmbedding)

    h0 = TimeDistributed(WeightLayer())(wordsEmbedding)
    h0 = keras.layers.Reshape([100,1,300])(h0)
    wordsEmbedding = reset()(wordsEmbedding)
    wordsEmbedding = keras.layers.Concatenate(axis=-2)([wordsEmbedding, h0])
    print('wordsEmbedding shape:',wordsEmbedding.shape, type(wordsEmbedding))

    wordGRU = TimeDistributed(GRU(300,
                                  # kernel_regularizer=regularizers.l2(0.01),
                                  # recurrent_regularizer=regularizers.l2(0.01)
                                  ))(wordsEmbedding)
    print('wordGRU shape:',wordGRU.shape, type(wordGRU))

    wordDense = TimeDistributed(Dense(150))(wordGRU)
    print('wordDense shape:',wordDense.shape, type(wordDense))

    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=True)(wordDense)
    sentenceGRU = GRU(64)(sentenceAtten)
    sentenceDense = Dense(20)(sentenceGRU)
    print('sentenceDense shape',sentenceDense.shape, type(sentenceDense))

    documentOut = Dense(1, activation="sigmoid", name="documentOut")(sentenceDense)
    print('documentOut shape',documentOut.shape, type(documentOut))

    model = Model(inputs=[wordsInputs], outputs=[documentOut])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model



def transformer_model():
    sentence_timestep = 100
    word_timestep = 50
    wordsInputs = Input(shape=(sentence_timestep, word_timestep,), dtype='int32', name='words_input')
    print('wordsInputs shape, ', wordsInputs.shape)

    wordsEmbedding = TimeDistributed(Embedding(MAX_VOCAB_SIZE, EMBED_SIZE))(wordsInputs)
    print('emb shape, ',wordsEmbedding.shape)


    wordAtten = TimeDistributed(MultiHeadSelfAttention(num_heads=2, use_masking=False))(wordsEmbedding)
    print('attention shape, ', wordAtten.shape)

    wordFlatten = TimeDistributed(Flatten())(wordAtten)
    print('wordFlatten shape, ', wordFlatten.shape)

    wordDense = TimeDistributed(Dense(1000))(wordFlatten)
    wordDense = TimeDistributed(Dense(100))(wordDense)
    print('wordDense shape, ', wordDense.shape)


    sentenceAtten = MultiHeadSelfAttention(num_heads=2, use_masking=False)(wordDense)
    print('documentEmb shape, ',sentenceAtten.shape)

    sentenceFlatten = Flatten()(sentenceAtten)
    print('sentenceFlatten shape, ', sentenceFlatten.shape)

    sentenceDense = Dense(1000)(sentenceFlatten)
    sentenceDense = Dense(100)(sentenceDense)
    print('sentenceGRU shape, ', sentenceDense.shape)

    sentenceDense = Dense(20)(sentenceDense)
    print('sentenceDense shape, ',sentenceDense)


    documentOut = Dense(1, activation="sigmoid", name="documentOut")(sentenceDense)

    model = Model(input=[wordsInputs], output=[documentOut])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model


def fig_GRU():
    timestep = 10
    figfeas_dim = 1000

    input = Input(shape=(timestep, figfeas_dim,), dtype='float32', name='fig_input')
    print(input.shape)

    x = GRU(600)(input)
    print(x.shape)

    x = Dense(100)(x)
    print(x.shape)

    x = Dense(20)(x)
    print(x.shape)

    output = Dense(1, activation="sigmoid", name="documentOut")(x)

    model = Model(input=[input], output=[output])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model

# def fig_GRU():
#     timestep = 10
#     figfeas_dim = 1000
#
#     input = Input(shape=(timestep, figfeas_dim,), dtype='float32', name='fig_input')
#     print(input.shape)
#
#     x = GRU(1000)(input)
#     print(x.shape)
#
#     x = Dense(100)(x)
#     print(x.shape)
#
#     x = Dense(20)(x)
#     print(x.shape)
#
#     output = Dense(1, activation="sigmoid", name="documentOut")(x)
#
#     model = Model(input=[input], output=[output])
#     model.compile(loss='binary_crossentropy',
#                   optimizer='rmsprop',
#                   metrics=['accuracy'])
#     return model

def fig_VGG16_GRU():
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
    vgg16.trainable = False

    input = Input(shape=(10,224,224,3))
    x = TimeDistributed(vgg16)(input)
    x = TimeDistributed(Flatten())(x)
    x = GRU(500)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(input=[input], output=[output])
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    return model

# if __name__ == '__main__':
#     transformer_gru_stepSentiment_model()