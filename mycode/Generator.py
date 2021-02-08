import os
import numpy as np
import json
from mycode.utils import load_image, list2hist
from skimage.color import rgb2hsv


def generator_fig_batch(all_fig_path_list, label_list, batch_size, fig_resize_shape, nolabel=False):
    fig_mean = np.load(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_figmean.npy'))

    length = len(all_fig_path_list)
    index = 0
    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))


        # 构造图像的输入array
        fig_path_list_batch = all_fig_path_list.iloc[index_list]
        user_fig_arr = np.zeros(shape=(batch_size, 10, fig_resize_shape, fig_resize_shape, 3))

        for batch_index in range(len(fig_path_list_batch)):
            fig_path_list = fig_path_list_batch.iloc[batch_index]
            for i in range(len(fig_path_list)):
                path = fig_path_list[i]
                try:
                    fig_arr = load_image(path)
                    if fig_arr.shape != (fig_resize_shape, fig_resize_shape, 3):
                        print('?:', path)
                    else:
                        fig_arr = fig_arr - fig_mean ##################### fig归一化
                        user_fig_arr[batch_index, i] = fig_arr
                except BaseException as e:
                    pass
        index = (index + batch_size) % length

        if nolabel:
            yield user_fig_arr
        else:
            yield (user_fig_arr, label)


def generator_fig_hue_hist_batch(all_fig_path_list, label_list, hist_count, batch_size, fig_resize_shape, nolabel=False):
    length = len(all_fig_path_list)
    index = 0
    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))


        # 构造图像的输入array
        fig_path_list_batch = all_fig_path_list.iloc[index_list]
        user_fig_arr = np.zeros(shape=(batch_size, 10, hist_count))

        for batch_index in range(len(fig_path_list_batch)):
            fig_path_list = fig_path_list_batch.iloc[batch_index]
            for i in range(len(fig_path_list)):
                path = fig_path_list[i]
                try:
                    fig_arr = load_image(path)
                    if fig_arr.shape != (fig_resize_shape, fig_resize_shape, 3):
                        print('?:', path)
                    else:
                        fig_hsv = rgb2hsv(fig_arr)
                        fig_hue = fig_hsv[:,:,0]
                        hist_arr = list2hist(fig_hue.ravel(), hist_count)

                        user_fig_arr[batch_index, i] = hist_arr
                except BaseException as e:
                    pass
        index = (index + batch_size) % length

        if nolabel:
            yield user_fig_arr
        else:
            yield (user_fig_arr, label)



def generator_text_batch(text_seq_list, label_list, batch_size, nolabel=False):
    length = len(text_seq_list)
    index = 0# index逐渐递增，但是值每次都对length取模

    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))

        #构造文本的输入array
        text_seq = text_seq_list.iloc[index_list]
        text_arr = np.array(list(text_seq), dtype='int32')

        index = (index + batch_size) % length

        if nolabel:
            yield text_arr
        else:
            yield (text_arr, label)


def generator_fig_text_batch(text_seq_list, all_fig_path_list, label_list, batch_size, fig_resize_shape, nolabel=False):
    fig_mean = np.load(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_figmean.npy'))

    length = len(text_seq_list)
    index = 0# index逐渐递增，但是值每次都对length取模

    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))

        #构造文本的输入array
        text_seq = text_seq_list.iloc[index_list]
        text_arr = np.array(list(text_seq), dtype='int32')

        # 构造图像的输入array
        fig_path_list_batch = all_fig_path_list.iloc[index_list]
        user_fig_arr = np.zeros(shape=(batch_size, 10, fig_resize_shape, fig_resize_shape, 3))

        for batch_index in range(len(fig_path_list_batch)):
            fig_path_list = fig_path_list_batch.iloc[batch_index]
            for i in range(len(fig_path_list)):
                path = fig_path_list[i]
                try:
                    fig_arr = load_image(path)
                    if fig_arr.shape != (fig_resize_shape, fig_resize_shape, 3):
                        print('?:', path)
                    else:
                        fig_arr = fig_arr - fig_mean  ##################### fig归一化
                        user_fig_arr[batch_index, i] = fig_arr
                except BaseException as e:
                    pass

        index = (index + batch_size) % length

        if nolabel:
            yield [text_arr, user_fig_arr]
        else:
            yield ([text_arr, user_fig_arr], label)


def generator_fig_text_multioutput_batch(text_seq_list, all_fig_path_list, label_list, batch_size, fig_resize_shape, nolabel=False):
    fig_mean = np.load(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_figmean.npy'))

    length = len(text_seq_list)
    index = 0# index逐渐递增，但是值每次都对length取模

    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))

        #构造文本的输入array
        text_seq = text_seq_list.iloc[index_list]
        text_arr = np.array(list(text_seq), dtype='int32')

        # 构造图像的输入array
        fig_path_list_batch = all_fig_path_list.iloc[index_list]
        user_fig_arr = np.zeros(shape=(batch_size, 10, fig_resize_shape, fig_resize_shape, 3))

        for batch_index in range(len(fig_path_list_batch)):
            fig_path_list = fig_path_list_batch.iloc[batch_index]
            for i in range(len(fig_path_list)):
                path = fig_path_list[i]
                try:
                    fig_arr = load_image(path)
                    if fig_arr.shape != (fig_resize_shape, fig_resize_shape, 3):
                        print('?:', path)
                    else:
                        fig_arr = fig_arr - fig_mean  ##################### fig归一化
                        user_fig_arr[batch_index, i] = fig_arr
                except BaseException as e:
                    pass

        index = (index + batch_size) % length

        if nolabel:
            yield [text_arr, user_fig_arr]
        else:
            yield ([text_arr, user_fig_arr], [label, label, label])



def generator_fig_text_multioutput_4_batch(text_seq_list, all_fig_path_list, label_list, batch_size, fig_resize_shape, nolabel=False):
    fig_mean = np.load(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_figmean.npy'))

    length = len(text_seq_list)
    index = 0# index逐渐递增，但是值每次都对length取模

    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))

        #构造文本的输入array
        text_seq = text_seq_list.iloc[index_list]
        text_arr = np.array(list(text_seq), dtype='int32')

        # 构造图像的输入array
        fig_path_list_batch = all_fig_path_list.iloc[index_list]
        user_fig_arr = np.zeros(shape=(batch_size, 10, fig_resize_shape, fig_resize_shape, 3))

        for batch_index in range(len(fig_path_list_batch)):
            fig_path_list = fig_path_list_batch.iloc[batch_index]
            for i in range(len(fig_path_list)):
                path = fig_path_list[i]
                try:
                    fig_arr = load_image(path)
                    if fig_arr.shape != (fig_resize_shape, fig_resize_shape, 3):
                        print('?:', path)
                    else:
                        fig_arr = fig_arr - fig_mean  ##################### fig归一化
                        user_fig_arr[batch_index, i] = fig_arr
                except BaseException as e:
                    pass

        index = (index + batch_size) % length

        if nolabel:
            yield [text_arr, user_fig_arr]
        else:
            yield ([text_arr, user_fig_arr], [label, label, label, label])



def taka_generator_text_batch(text_seq_list, label_list, batch_size, nolabel=False):
    length = len(text_seq_list)
    index = 0# index逐渐递增，但是值每次都对length取模

    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))

        #构造文本的输入array
        text_seq = text_seq_list[index_list]
        text_arr = np.array(list(text_seq), dtype='int32')

        index = (index + batch_size) % length

        if nolabel:
            yield text_arr
        else:
            yield (text_arr, label)


def taka_generator_fig_text_batch(text_seq_list, all_fig_path_list, label_list, batch_size, fig_resize_shape, nolabel=False):
    fig_mean = np.load(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_figmean.npy'))

    length = len(text_seq_list)
    index = 0# index逐渐递增，但是值每次都对length取模

    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))

        #构造文本的输入array
        text_seq = text_seq_list[index_list]
        text_arr = np.array(list(text_seq), dtype='int32')

        # 构造图像的输入array
        fig_path_list_batch = all_fig_path_list.iloc[index_list]
        user_fig_arr = np.zeros(shape=(batch_size, 10, fig_resize_shape, fig_resize_shape, 3))

        for batch_index in range(len(fig_path_list_batch)):
            fig_path_list = fig_path_list_batch.iloc[batch_index]
            for i in range(len(fig_path_list)):
                path = fig_path_list[i]
                try:
                    fig_arr = load_image(path)
                    if fig_arr.shape != (fig_resize_shape, fig_resize_shape, 3):
                        print('?:', path)
                    else:
                        fig_arr = fig_arr - fig_mean  ##################### fig归一化
                        user_fig_arr[batch_index, i] = fig_arr
                except BaseException as e:
                    pass

        index = (index + batch_size) % length

        if nolabel:
            yield [text_arr, user_fig_arr]
        else:
            yield ([text_arr, user_fig_arr], label)







SENTI_DIC_PATH = os.path.join(os.path.dirname(__file__), '..', 'output', 'wordindex_polarity_dic.json')


def generator_text_senti_batch(text_seq_list, label_list, batch_size, nolabel=False):
    dic = json.load(open(SENTI_DIC_PATH))

    length = len(text_seq_list)
    index = 0# index逐渐递增，但是值每次都对length取模

    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))

        #构造文本的输入array
        text_seq = text_seq_list.iloc[index_list]
        text_arr = np.array(list(text_seq), dtype='int32')

        text_senti = np.zeros((text_arr.shape[0], text_arr.shape[1], text_arr.shape[2]))# (batchsize, 100, 50)
        for k in range(len(text_arr)):
            sample = text_arr[k]
            for i in range(sample.shape[0]):
                for j in range(sample.shape[1]):
                    num = sample[i][j]

                    if dic.__contains__(num):
                        print('contains', num)
                        text_senti[k][i][j] = dic[num]
        text_senti = text_senti.reshape((text_arr.shape[0], text_arr.shape[1], text_arr.shape[2], 1))

        index = (index + batch_size) % length

        if nolabel:
            yield [text_arr, text_senti]
        else:
            yield ([text_arr, text_senti], label)


def generator_fig_text_senti_multioutput_batch(text_seq_list, all_fig_path_list, label_list, batch_size, fig_resize_shape, nolabel=False):
    fig_mean = np.load(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_figmean.npy'))
    dic = json.load(open(SENTI_DIC_PATH))

    length = len(text_seq_list)
    index = 0# index逐渐递增，但是值每次都对length取模

    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))

        #构造文本的输入array
        text_seq = text_seq_list.iloc[index_list]
        text_arr = np.array(list(text_seq), dtype='int32')

        text_senti = np.zeros((text_arr.shape[0], text_arr.shape[1], text_arr.shape[2]))# (batchsize, 100, 50)
        for k in range(len(text_arr)):
            sample = text_arr[k]
            for i in range(sample.shape[0]):
                for j in range(sample.shape[1]):
                    num = sample[i][j]

                    if dic.__contains__(num):
                        print('contains', num)
                        text_senti[k][i][j] = dic[num]
        text_senti = text_senti.reshape((text_arr.shape[0], text_arr.shape[1], text_arr.shape[2], 1))

        # 构造图像的输入array
        fig_path_list_batch = all_fig_path_list.iloc[index_list]
        user_fig_arr = np.zeros(shape=(batch_size, 10, fig_resize_shape, fig_resize_shape, 3))

        for batch_index in range(len(fig_path_list_batch)):
            fig_path_list = fig_path_list_batch.iloc[batch_index]
            for i in range(len(fig_path_list)):
                path = fig_path_list[i]
                try:
                    fig_arr = load_image(path)
                    if fig_arr.shape != (fig_resize_shape, fig_resize_shape, 3):
                        print('?:', path)
                    else:
                        fig_arr = fig_arr - fig_mean  ##################### fig归一化
                        user_fig_arr[batch_index, i] = fig_arr
                except BaseException as e:
                    pass


        index = (index + batch_size) % length

        if nolabel:
            yield [text_arr, text_senti, user_fig_arr]
        else:
            yield ([text_arr, text_senti, user_fig_arr], [label, label, label])





######## 用于只predict的时候
def generator_label(label_list, batch_size):
    length = len(label_list)
    index = 0# index逐渐递增，但是值每次都对length取模

    while True:
        index_list = []
        for i in range(batch_size):
            index_list.append(index+i)

        #label
        label = np.array(list(label_list.iloc[index_list]), dtype='int32')
        label = label.reshape((batch_size, 1))

        index = (index + batch_size) % length

        yield label