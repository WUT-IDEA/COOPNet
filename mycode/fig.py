import numpy as np
import pandas as pd
import tensorflow as tf
from mycode import vgg16, utils
from mycode.Nclasses import labels
import os
import gc
import math
from skimage import io

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def pred_class(img_path):
    img_ready = utils.load_image(img_path)# 读入待判图，并将其处理成[1,224,224,3]的形状
    # fig = plt.figure(u"Top-5 prediction")# 准备一张图，把运行的结果可视化。
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [1,224,224,3])
        # 这里images的占位形状为[一次喂入的图片张数，图片长，图片宽，图片通道信息红绿蓝]
        vgg = vgg16.Vgg16()# 实例化vgg，运行了Vgg16类的初始化函数__ini__
        vgg.forward(images)
        probability = sess.run(vgg.prob, feed_dict={images: img_ready})
        # probability存着1000个概率，对应1000中每个类别的概率，索引key是labels中的健
        # 用probability[0][i]遍历每个概率值
        top5 = np.argsort(probability[0])[-1:-6:-1]
        # np.argsort表示对列表从小到大排序，返回索引值。这里的top5存着probability列表中5个最高概率的索引
        print("top:", top5)
        values = []# 用来存probability中元素的值，存概率
        bar_label = []# 用来存标签字典中对应的值，存名称
        for n, i in enumerate(top5):
            # print("n:", n)
            # print("i:", i)
            values.append(probability[0][i])
            bar_label.append(labels[i])
            print(i, ":", labels[i], "----", utils.percent(probability[0][i]))

        '''
        ax = fig.add_subplot(111)
        # fig.add_subplot(数 数 数)，111表示：包含1行，包含1列，当前是第1个
        ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc='g')
        # ax.bar(bar的个数，bar的值，每个bar的名字，bar宽，bar色)
        ax.set_ylabel(u'probabilityit')
        # ax.set_ylabel(" ")设置第一列的y轴名字
        ax.set_title(u'Top-5')
        # ax.set_title(" ")设置子图名字
        for a, b in zip(range(len(values)), values):
            ax.text(a, b+0.0005, utils.percent(b), ha='center', va='bottom', fontsize=7)
            # ax.text(文字x坐标，文字y坐标，文字内容，ha='center'水平方向位置，va='bottom'垂直方向位置，字号)
        plt.show()
        '''

        return values[0], bar_label[0]

def get_fc8(img_path):
    img_ready = utils.load_image(img_path)# 读入待判图，并将其处理成[None,224,224,3]的形状
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [1,224,224,3])
        # 这里images的占位形状为[一次喂入的图片张数，图片长，图片宽，图片通道信息红绿蓝]
        vgg = vgg16.Vgg16()# 实例化vgg，运行了Vgg16类的初始化函数__ini__
        vgg.forward(images)
        probability = sess.run(vgg.fc8, feed_dict={images: img_ready})
        return probability

def single_get_fc8(arr):
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [1,224,224,3])
        # 这里images的占位形状为[一次喂入的图片张数，图片长，图片宽，图片通道信息红绿蓝]
        vgg = vgg16.Vgg16()# 实例化vgg，运行了Vgg16类的初始化函数__ini__
        vgg.forward(images)
        probability = sess.run(vgg.fc8, feed_dict={images: arr})
        return probability

def batch_get_fc8(img_ready_array):
    # img_ready_list = []
    # for img_path in img_path_list:
    #     img_ready_list.append(utils.load_image(img_path))# 读入待判图，并将其处理成[1,224,224,3]的形状
    # img_ready_array = np.array(img_ready_list)
    print(img_ready_array.shape)
    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [None,224,224,3])
        # 这里images的占位形状为[一次喂入的图片张数，图片长，图片宽，图片通道信息红绿蓝]
        vgg = vgg16.Vgg16()# 实例化vgg，运行了Vgg16类的初始化函数__ini__
        vgg.forward(images)
        fc8_feas = sess.run(vgg.fc8, feed_dict={images: img_ready_array})
        return fc8_feas

def read_data(train_file_name, test_file_name):
    train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', train_file_name))
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', test_file_name))
    return train, test


# 将传入的list中的路径的图片，读入图片，转换成(224,224,3)的形式，
def figs_reshaped_arr(fig_path):# 传入的是某个用户的所有图片的路径
    if len(fig_path)==0:# 如果某个用户拥有的图片数为0
        return np.nan
    reshape_list = []
    for path in fig_path:
        try:
            img_reshaped = utils.load_image(path)
            reshape_list.append(img_reshaped)
        except OSError as e:
            print('img cannot open:',path,e)
            pass
    if len(reshape_list)==0:
        return np.nan
    arr = np.zeros(shape=[len(reshape_list), 224, 224, 3])
    for i,reshaped in enumerate(reshape_list):
        arr[i,:,:,:] = reshaped
    return arr

def fig_vectorize():
    # train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train.pd'))# 读入train
    # train['figs_reshaped_arr'] = train['fig_path_list'].apply(lambda list: figs_reshaped_arr(list))
    # pd.to_pickle(train, os.path.join(os.path.dirname(__file__), '..', 'output', 'train_reshapedfig.pd'))

    # test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test.pd'))
    # test['figs_reshaped_arr'] = test['fig_path_list'].apply(lambda list: figs_reshaped_arr(list))
    # pd.to_pickle(test, os.path.join(os.path.dirname(__file__), '..', 'output', 'test_reshapedfig.pd'))

    with tf.Session() as sess:
        images = tf.placeholder(tf.float32, [None,224,224,3])
        # 这里images的占位形状为[一次喂入的图片张数，图片长，图片宽，图片通道信息红绿蓝]
        vgg = vgg16.Vgg16()# 实例化vgg，运行了Vgg16类的初始化函数__ini__
        vgg.forward(images)

        ## train========
        ## 因为文件太大，所以分开存储
        # chunk_num = 1500
        # for i in range(chunk_num):
        #     train_temp = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_reshapedfig.pd'))
        #     chunk_size_train = math.ceil(len(train_temp) / chunk_num)
        #     train = train_temp.iloc[i*chunk_size_train: (i+1)*chunk_size_train]
        #     del train_temp
        #     gc.collect()
        #
        #     arr_list = []
        #     for arr in train['figs_reshaped_arr']:
        #         if type(arr)!=float:
        #             arr_list.append(arr)
        #     del train
        #     gc.collect()
        #     reshaped_arr_ensemble = np.concatenate(arr_list, axis=0)
        #     print(reshaped_arr_ensemble.shape)
        #     print('train reshaped fig arr ensemble finished')
        #     del arr_list
        #     gc.collect()
        #
        #     fig_feas = sess.run(vgg.fc8, feed_dict={images: reshaped_arr_ensemble})
        #     np.save(os.path.join(os.path.dirname(__file__), '..', 'output', 'midFile', 'train_figFC8_%d.npy'%(i)), fig_feas)
        #     print('train feature finished %d'%(i))
        #     del reshaped_arr_ensemble, fig_feas
        #     gc.collect()

        ## test=========
        chunk_num = 950
        for i in range(chunk_num):
            test_temp = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_reshapedfig.pd'))
            chunk_size_test = math.ceil(len(test_temp) / chunk_num)
            test = test_temp.iloc[i*chunk_size_test: (i+1)*chunk_size_test]
            del test_temp
            gc.collect()

            arr_list = []
            for arr in test['figs_reshaped_arr']:
                if type(arr)!=float:
                    arr_list.append(arr)
            del test
            gc.collect()
            reshaped_arr_ensemble = np.concatenate(arr_list, axis=0)
            print(reshaped_arr_ensemble.shape)
            print('test reshaped fig arr ensemble finished')
            del arr_list
            gc.collect()

            fig_feas = sess.run(vgg.fc8, feed_dict={images: reshaped_arr_ensemble})
            np.save(os.path.join(os.path.dirname(__file__), '..', 'output', 'midFile', 'test_figFC8_%d.npy'%(i)), fig_feas)
            print('test feature finished %d'%(i))
            del reshaped_arr_ensemble, fig_feas
            gc.collect()

# 将得到的FC8输出的所有中间文件读入，然后根据每个用户持有的图片数量slice分离。
def slice_fig_vector():
    ## train
    train = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'train_reshapedfig.pd'))
    train['fig_num'] = train['figs_reshaped_arr'].apply([lambda arr: 0 if type(arr)==float else len(arr)])
    train['fig_num_cumsum'] = train['fig_num'].cumsum()# 每一项累计求和

    fig_num_cumsum = train['fig_num_cumsum']# 得到累计和的Series

    sum_ = fig_num_cumsum[len(fig_num_cumsum) - 1]# 得到累计求和的最后一项，即为所有用的图片数量之和。
    train_new = train.drop(columns=['figs_reshaped_arr'])
    del train
    gc.collect()

    train_list = []# 将得到的图片FC8输出读入，存储到train_list中
    train_len = 1500# train的FC8输出有1500个中间文件
    for i in range(train_len):
        file_arr = np.load(os.path.join(os.path.dirname(__file__), '..', 'output', 'midFile', 'train_figFC8_%d.npy' % (i)))
        # print('train_file_arr_%d_shape ='%(i),file_arr.shape)
        train_list.append(file_arr)
    train_arr = np.concatenate(train_list, axis=0)# 将list中的向量concatenate
    print('concatenate train_arr shape =',train_arr.shape)
    print('sum_ =', sum_)
    del train_list
    gc.collect()

    slices = np.split(train_arr, fig_num_cumsum, axis=0)# 根据累计求和得到分离后的slices
    del train_arr
    gc.collect()
    if slices[-1].shape[0] == 0:
        print('train slice right')
        train_new['fig_feas'] = slices[:len(slices)-1]
        pd.to_pickle(train_new, os.path.join(os.path.dirname(__file__), '..', 'output','train_figfeas.pd'))
        del slices, train_new
        gc.collect()
    else:
        print('slice wrong')

    ## test
    test = pd.read_pickle(os.path.join(os.path.dirname(__file__), '..', 'output', 'test_reshapedfig.pd'))
    test['fig_num'] = test['figs_reshaped_arr'].apply([lambda arr: 0 if type(arr)==float else len(arr)])
    test['fig_num_cumsum'] = test['fig_num'].cumsum()
    fig_num_cumsum = test['fig_num_cumsum']
    sum_ = fig_num_cumsum[len(fig_num_cumsum) - 1]
    test_new = test.drop(columns=['figs_reshaped_arr'])
    del test
    gc.collect()

    test_list = []
    test_len = 950
    for i in range(test_len):
        file_arr = np.load(os.path.join(os.path.dirname(__file__), '..', 'output', 'midFile', 'test_figFC8_%d.npy' % (i)))
        # print('test_file_arr_%d_shape ='%(i),file_arr.shape)
        test_list.append(file_arr)
    test_arr = np.concatenate(test_list, axis=0)
    print('concatenate test_arr shape =', test_arr.shape)
    print('sum_ =', sum_)
    del test_list
    gc.collect()
    slices = np.split(test_arr, fig_num_cumsum, axis=0)
    del test_arr
    gc.collect()
    if slices[-1].shape[0] == 0:
        print('test slice right')
        test_new['fig_feas'] = slices[:len(slices)-1]
        pd.to_pickle(test_new, os.path.join(os.path.dirname(__file__), '..', 'output','test_figfeas.pd'))
        del slices, test_new
        gc.collect()
    else:
        print('slice wrong')



def main():
    # fig_vectorize()
    slice_fig_vector()


if __name__ == '__main__':
    main()