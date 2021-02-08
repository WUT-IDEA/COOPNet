from skimage import io, transform
from skimage.color import rgb2hsv
import numpy as np
import math
# import matplotlib.pyplot as plt
import tensorflow as tf
# from pylab import mpl
import xml.etree.ElementTree
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
# 为了正常显示中文标签和正负号
# mpl.rcParams['font.sans-serif'] = ['SimHei']
# mpl.rcParams['axes.unicode_minus'] = False


fig_resize_shape = 150

def load_image(path_):
    img = np.asarray(Image.open(path_))
    if len(img.shape) == 2:#单通道
        c = []
        for i in range(3):
            c.append(img)
        img = np.asarray(c)
        img = img.transpose([1, 2, 0])
    elif img.shape[2] != 3:#
        # print(img.shape)
        img = np.asarray(Image.open(path_).convert("RGB"))
    else:
        assert img.shape[2] == 3
        img = io.imread(os.path.join(os.path.dirname(__file__),path_))# 读入图片
    img = img / 255.0# 将像素归一化到0-1之间

    # ax0 = fig.add_subplot(131)# fig.add_subplot(数 数 数)，131表示：包含1行，包含3列，当前是第1个
    # ax0.set_xlabel(u'Original Picture')
    # ax.set_xlabel(" ")设置第一列的x轴名字
    # ax.bar(bar的个数，bar的值，每个bar的名字，bar宽，bar色)
    # ax0.imshow(img)

    short_edge = min(img.shape[:2])
    y = int((img.shape[0] - short_edge) / 2)
    x = int((img.shape[1] - short_edge) / 2)
    crop_img = img[y: y+short_edge, x: x+short_edge]

    # ax1 = fig.add_subplot(132)
    # ax1.set_xlabel(u"Centre Picture")
    # ax1.imshow(crop_img)

    re_img = transform.resize(crop_img, (fig_resize_shape, fig_resize_shape))# 224，224分辨率

    # ax2 = fig.add_subplot(133)
    # ax2.set_xlabel(u"Resize Picture")
    # ax2.imshow(re_img)

    # print(re_img.shape)


    if re_img.shape == (fig_resize_shape,fig_resize_shape,3):
        return re_img
    elif re_img.shape == (fig_resize_shape,fig_resize_shape):
        arr = np.zeros([fig_resize_shape,fig_resize_shape,3])
        for i in range(re_img.shape[0]):
            for j in range(re_img.shape[1]):
                arr[i, j, 0] = re_img[i, j]
                arr[i, j, 1] = re_img[i, j]
                arr[i, j, 2] = re_img[i, j]
        return arr
    else:
        print('shape =',re_img.shape)

def percent(value):
    return("%2.f%%" % (value * 100))# 将一个数字表示为百分比的形式

def get_text_list(id, path):
    file_path = os.path.join(path,id+'.xml')
    tree = xml.etree.ElementTree.parse(file_path)
    root = tree.getroot()
    text_list = []
    for node in root.iter('document'):
        text_list.append(node.text)
    return text_list

def get_fig_path_list(id, path):
    fig_path_list = []
    for i in range(10):
        fig_path = os.path.join(path, id, id+'.'+str(i)+'.jpeg')
        try:
            f = open(fig_path, 'rb')
            fig_path_list.append(fig_path)
        except FileNotFoundError:
            pass
    return fig_path_list

def list2hist(list, bins_num):
    count_arr = [0] * bins_num
    for i in list:
        index = math.floor(i * bins_num)
        count_arr[index] = count_arr[index] + 1
    return count_arr