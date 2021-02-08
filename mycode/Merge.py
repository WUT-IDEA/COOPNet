from keras.engine.topology import Layer
from keras import initializers
from keras import backend as K
from keras.layers import concatenate


class ColumnMaximum(Layer):
    def __init__(self):
        super(ColumnMaximum, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        super(ColumnMaximum, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        output = K.max(x, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class ColumnAverage(Layer):
    def __init__(self):
        super(ColumnAverage, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        super(ColumnAverage, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        output = K.mean(x, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


class CrossMaximum(Layer):
    def __init__(self):
        super(CrossMaximum, self).__init__()

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape[0]) == 2# 验证text输入的维度
        assert len(input_shape[1]) == 2# 验证fig输入的维度
        self.text_dim = input_shape[0][1]# 文本的维度，例如64
        self.fig_dim = input_shape[1][1]# 图像的维度
        super(CrossMaximum, self).build(input_shape)


    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        assert isinstance(x, list)
        x_text, x_fig = x
        x_text = K.expand_dims(x_text, axis=2)
        x_fig = K.expand_dims(x_fig, axis=1)
        cartesian = K.batch_dot(x_text, x_fig)
        row_max_pool = K.max(cartesian, axis=2)
        col_max_pool = K.max(cartesian, axis=1)
        output = concatenate([row_max_pool, col_max_pool])
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1]+input_shape[1][1])



class CrossAverage(Layer):
    def __init__(self):
        super(CrossAverage, self).__init__()

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        assert len(input_shape[0]) == 2# 验证text输入的维度
        assert len(input_shape[1]) == 2# 验证fig输入的维度
        self.text_dim = input_shape[0][1]# 文本的维度，例如64
        self.fig_dim = input_shape[1][1]# 图像的维度
        super(CrossAverage, self).build(input_shape)


    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        assert isinstance(x, list)
        x_text, x_fig = x
        cartesian = K.batch_dot(x_text, x_fig)
        row_max_pool = K.mean(cartesian, axis=1)
        col_max_pool = K.mean(cartesian, axis=0)
        output = concatenate(row_max_pool, col_max_pool)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1]+input_shape[1][1])


class WeightedVote(Layer):
    def __init__(self):
        self.init = initializers.get('normal')
        super(WeightedVote, self).__init__()

    def build(self, input_shape):
        self.inputLength = input_shape[1]# 得到输入的维度

        self.W = K.variable(self.init((self.inputLength,1)))
        self.trainable_weights = [self.W]

        super(WeightedVote, self).build(input_shape)


    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        output = K.dot(x,self.W)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


class Vote(Layer):
    def __init__(self):
        self.init = initializers.get('normal')
        super(Vote, self).__init__()

    def build(self, input_shape):
        super(Vote, self).build(input_shape)

    def call(self, x, mask=None):
        x = K.sum(x, axis=1)
        x = K.expand_dims(x, 1)
        output = x * 1/3
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)