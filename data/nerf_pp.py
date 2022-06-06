from .dataset import NeRFDataset
import tensorflow as tf
import numpy as np
import os
from glob import glob
import json
import cv2

"""
Nerf++ dataset class
"""
class NerfPPData(NeRFDataset):
    def __init__(self, base_path, scene_name, type='train', white_bkgd=False, shuffle=False, **augment_params):
        assert(type in ['train', 'validation', 'test'])
        super(NerfPPData, self).__init__(shuffle=shuffle, **augment_params)

        if type == 'test' and scene_name.startswith("scan"):
            #type = "interpolation"
            type = "extrapolation"

        self.white_bkgd = white_bkgd
        self.scene_name = scene_name
        self.scene_type = type

        # json_path = os.path.join(base_path, scene_name, "transforms_%s.json" % type)
        dir_path = os.path.join(base_path, scene_name, type)
        img_path = os.path.join(dir_path, 'rgb')
        pose_path = os.path.join(dir_path, 'pose')
        intrinsic_path = os.path.join(dir_path, 'intrinsics')

        H = 720
        W = 1280
        C = 3
        for img in os.listdir(img_path):
            self.image_path.append(os.path.join(img_path, img))

            if len(self.image_path) == 1:
                im_sample = tf.image.decode_png(tf.io.read_file(self.image_path[0]))
                H, W, C = im_sample.shape

            index = img.split('.')[0]
            info = index + '.txt'
            #print(os.path.join(pose_path, info))
            pose = open(os.path.join(pose_path, info), "r").readline()
            intrinsic = open(os.path.join(intrinsic_path, info), "r").readline()
            pose_list = list(map(float, pose.split(" ")))
            intrinsic_list = list(map(float, intrinsic.split(" ")))

            pose_np = np.array(pose_list, dtype = np.float32).reshape(4,4)
            intrinsic_np = np.array(intrinsic_list, dtype=np.float32).reshape(4, 4)
            pose_np = self.process_pose(pose_np)

            intrinsic_np = self.process_intrinsic(intrinsic_np, H, W)
            self.pose.append(pose_np)
            # self.intrinsic.append(np.linalg.inv(intrinsic_np)[:3,:3])
            self.intrinsic.append(intrinsic_np[:3,:3])
        


        self.image_size = [H, W, C]

        self.near = 0.2
        self.far = 2.0

    @property
    def ray_dataset(self):
        ray_dataset = super(NerfPPData, self).ray_dataset
        if self.scene_name == 'basket' and self.scene_type != 'test':
            _c = lambda x: tf.image.central_crop(x, 0.9)
            ray_dataset = ray_dataset.map(lambda x, y: ((_c(x[0]), _c(x[1])), _c(y)))
        elif self.scene_name.startswith('scan') and self.scene_type == 'validation':
            _c = lambda x: tf.image.resize(x, (600, 800), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            ray_dataset = ray_dataset.map(lambda x, y: ((_c(x[0]), _c(x[1])), _c(y)))
        return ray_dataset

    @staticmethod
    def process_pose(C2W):
        flip_yz = np.eye(4, dtype=C2W.dtype)
        flip_yz[1, 1] = -1
        flip_yz[2, 2] = -1
        C2W = np.matmul(C2W, flip_yz)
        return C2W

    @staticmethod
    def process_intrinsic(intrinsic, H, W):
        intrinsic[0, 2] = intrinsic[0, 2] - W / 2
        intrinsic[1, 2] = intrinsic[1, 2] - H / 2
        # intrinsic[0, 0] = - intrinsic[0, 0]
        # intrinsic[1, 1] = - intrinsic[1, 1]
        intrinsic[1, 2] = - intrinsic[1, 2]
        return intrinsic