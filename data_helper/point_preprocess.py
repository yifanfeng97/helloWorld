from torchvision import transforms
import numpy as np
import torch
import projector
from PIL import Image

class norm_r1(object):
    def __init__(self):
        super(norm_r1, self).__init__()

    def __call__(self, points):
        # return points / (points.max(0) - points.min(0)) * 2
        L_max = np.sqrt(np.sum(points ** 2, -1)).max()
        return points / L_max


class cvt2img(object):
    def __init__(self, init_sz, target_sz):
        super(cvt2img, self).__init__()
        self.init_sz = init_sz
        self.target_sz = target_sz

    def __call__(self, points):
        img = projector.pjt_3d_to_2d(points, self.init_sz)
        # img = img[np.newaxis, :, :]
        img = (img*255).astype(np.uint8)
        img = Image.fromarray(img)
        # img.show()
        img = img.resize((self.target_sz, self.target_sz))
        # img.show()
        # img = img.resize((self.init_sz, self.init_sz))
        # img.show()
        return img


class rotate_points_by_angle(object):
    def __init__(self):
        super(rotate_points_by_angle, self).__init__()

    def __call__(self, points, angle):

        cos_val = np.cos(angle)
        sin_val = np.sin(angle)

        rotation_matrix = np.array([
            [cos_val, 0, sin_val],
            [0, 1, 0],
            [-sin_val, 0, cos_val]
        ])
        rotated_data = np.dot(points, rotation_matrix)
        return rotated_data

class random_rotate_points(object):
    def __init__(self):
        super(random_rotate_points, self).__init__()
        self.rotate_points_fun = rotate_points_by_angle()

    def __call__(self, points):
        angle = np.random.random() * 2 * np.pi - np.pi
        return self.rotate_points_fun(points, angle)

class ps_to_tensor(object):
    def __init__(self):
        super(ps_to_tensor, self).__init__()

    def __call__(self, points):
        points = torch.from_numpy(points)
        return points


def get_norm_points_compose():
    compose = transforms.Compose([
        norm_r1(),
        # random_rotate_points(),
        ps_to_tensor()
    ])
    return compose

def get_train_test_compose():
    compose = transforms.Compose([
        norm_r1(),
        random_rotate_points(),
        cvt2img(200, 224),
        transforms.ToTensor()
    ])
    return compose

def untransform(img):
    # img = img.transpose(1, 2, 0)
    img = img[0]
    img = (img*255).astype(np.uint8)
    img = Image.fromarray(img)
    return img