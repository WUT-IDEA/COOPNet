from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K
import tensorflow as tf



class TextFigAtten(Layer):
    def __init__(self):
        self.init = initializers.get('normal')
        self.supports_masking = True
        super(TextFigAtten, self).__init__()

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape[0]) == 2# 验证text输入的维度
        assert len(input_shape[1]) == 3# 验证fig输入的维度
        self.text_dim = input_shape[0][1]# 文本的维度，例如64
        self.fig_dim = input_shape[1][2]# 10个图像每个图像的维度，例如1000
        self.fig_num = input_shape[1][1]# 图片的张数，例如10

        self.W = K.variable(self.init((self.fig_dim, self.text_dim)))# (1000,64)
        # W是用来改变fig的维度，方便与text进行相似度计算。
        self.b = K.variable(self.init((self.text_dim, )))# (64)
        self.trainable_weights = [self.W, self.b]
        super(TextFigAtten, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        assert isinstance(x, list)
        x_text, x_fig = x

        uit = K.tanh(K.bias_add(K.dot(x_fig, self.W), self.b))# (10,1000)*(1000,64)+b => (10,64)
        x_text = K.expand_dims(x_text)# (64,) => (64,1)

        # ait = K.dot(uit, x_text)# (10,64)*(64,1) => (10,1)
        ait = K.batch_dot(uit, x_text)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x_fig * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[1][2])


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