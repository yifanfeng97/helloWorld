import numpy as np
import point_datasets
import os.path as osp
import h5_helper
import point_preprocess
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
# import utils.heatmap
from utils import heatmap

"""
Airplane	02691156
Bag	02773838
Cap	02954340
Car	02958343
Chair	03001627
Earphone	03261776
Guitar	03467517
Knife	03624134
Lamp	03636649
Laptop	03642806
Motorbike	03790512
Mug	03797390
Pistol	03948459
Rocket	04099429
Skateboard	04225987
Table	04379243
"""


def get_theta(a, b, z):
    c = []
    for idx in range(len(a)):

        if b[idx] == 0:
            val = np.arcsin(a[idx]/(b[idx] + 1e-6))
        else:
            val = np.arcsin(a[idx]/b[idx])

        # if z[idx] < 0 and a[idx] != 0:
        #     if a[idx] > 0:
        #         val = np.pi - val
        #     else:
        #         val = -np.pi - val
            # if a[idx] > 0:
            #     val = -(np.pi - val)
            # else:
            #     val = -(-np.pi - val)

        c.append(val)

    return np.array(c)

def prj_sub_3d_to_2d(data, img_sz):

    L = np.sqrt(np.sum(data ** 2, -1))
    print (L.shape)
    alpha = get_theta(data[:, 0], L, data[:, 2])
    beta = get_theta(data[:, 1], L, data[:, 2])

    alpha = alpha + np.pi / 2
    beta = beta + np.pi / 2

    img = np.zeros((img_sz, img_sz), dtype=np.float32)

    x = np.array([int(t) for t in alpha / np.pi * img_sz])
    y = np.array([int(t) for t in beta / np.pi * img_sz])
    img[x, y] = L

    return img


def pjt_3d_to_2d(data, img_sz=200):

    data_up = data[data[:, 2] >= 0, :]
    data_down = data[data[:, 2] < 0, :]
    # data_down[:, 2] = -data_down[:, 2]
    # data up
    img_up = prj_sub_3d_to_2d(data_up, img_sz)
    img_down = prj_sub_3d_to_2d(data_down, img_sz)

    return img_up, img_down



# def pjt_3d_to_map(data, img_size=200):
#     '''
#     Input: point cloud data, size:[B,N,3]
#     Output: Image, size:[B, img_size, img_size*2, 1]
#
#     '''
#     tan = np.divide(data[:, 1], data[:, 0])
#     dis = np.sqrt(np.sum(data**2, -1))
#     cos = np.divide(data[:, 2], dis)
#
#     sigma = np.arctan(tan)
#     theta = np.arccos(cos)
#
#     x_axis = img_size * theta / np.pi
#     y_axis = img_size * ((sigma / np.pi) + 0.5)
#     x_axis = np.floor(x_axis)
#     y_axis = np.floor(y_axis)
#     color = dis
#
#     pic1 = np.zeros((img_size, img_size))
#     pic2 = np.zeros((img_size, img_size))
#
#
#     # get the idx of points satisfied with x>0
#     idx1 = np.where(data[:, 0] > 0)
#     for i in idx1[0]:
#         if color[i] > pic1[int(x_axis[i]), int(y_axis[i])]:
#             pic1[int(x_axis[i]), int(y_axis[i])] = color[i]
#
#     # get the idx of points satisfied with x<0
#     idx2 = np.where(data[:, 0] < 0)
#     for i in idx2[0]:
#         if color[i] > pic2[int(x_axis[i]), int(y_axis[i])]:
#             pic2[int(x_axis[i]), int(y_axis[i])] = color[i]
#     pic = np.concatenate((pic1, pic2), axis=1)
#     # pic = np.expand_dims(pic, axis=3)
#
#     return pic


def pjt_2d_to_3d(img):
    pass

if __name__ == '__main__':
    # d = point_datasets.point3Dataset(
    #     root = '../../data/shapenetcore_partanno_segmentation_benchmark_v0',
    #     class_choice = ['Motorbike'])

    target_label = 7
    cnt = 18

    d = point_datasets.point_modelnet40_Dataset_cls(mode='train', generate_img=False)

    idx = 0
    t_cnt = 0
    ps, cls = d[idx]
    while True:
        if cls.numpy() == target_label:
            t_cnt += 1
            if t_cnt >= cnt: break
        idx += 1
        ps, cls = d[idx]
    print(ps.size, ps.type(), cls.size, cls.type())
    # print(ps.size(), ps.type(), seg.size(), seg.type())
    print(ps.shape)

    ps_np = ps.numpy()

    # ps_np = np.array([
    #     [0.7, 0, -0.7],
    #     [0.7, 0, 0.7],
    #     [-0.7, 0, 0.7],
    #     [-0.7, 0, -0.7],
    #     [0, 0.7, -0.7],
    #     [0, 0.7, 0.7],
    #     [0, -0.7, 0.7],
    #     [0, -0.7, -0.7],
    # ])

    # ps_np = []
    # for idx in range(5):
    #     theta = np.random.random()*np.pi
    #     phi = np.random.random()*2*np.pi
    #     # x = np.random.random()*2 - 1
    #     # y = np.random.random()*np.sqrt(1-x**2)*2 - 1
    #     # z = np.sqrt(1 - x**2 - y**2)
    #     x = 0.8*np.sin(theta)*np.cos(phi)
    #     y = 0.8*np.sin(theta)*np.sin(phi)
    #     z = 0.8*np.cos(theta)
    #
    #     ps_np.append([x, y, z])
    # ps_np = np.array(ps_np)
    # print(ps_np)

    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.scatter(ps_np[:, 0], ps_np[:, 1], ps_np[:, 2])

    ps2_np_up, ps2_np_down = pjt_3d_to_2d(ps_np)
    # ps2_np = pjt_3d_to_map(ps_np)
    ps2_np_up_htmap = heatmap.get_heatmap_from_prob(ps2_np_up)
    ps2_np_down_htmap = heatmap.get_heatmap_from_prob(ps2_np_down)

    ax = fig.add_subplot(132)
    plt.imshow(ps2_np_up_htmap)

    ax = fig.add_subplot(133)
    plt.imshow(ps2_np_down_htmap)
    # ax.plot(ps2_np_htmap)

    plt.show()


