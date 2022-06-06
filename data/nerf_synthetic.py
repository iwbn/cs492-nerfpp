from .dataset import NeRFDataset
import tensorflow as tf
import numpy as np
import os
from glob import glob
import json
import cv2

"""
NeRF Synthetic dataset class
"""
class NerfSynthetic(NeRFDataset):
    def __init__(self, base_path, scene_name, type='train', white_bkgd=False, shuffle=False, **augment_params):
        assert(type in ['train', 'val', 'test'])
        super(NerfSynthetic, self).__init__(shuffle=shuffle, **augment_params)

        self.white_bkgd = white_bkgd

        json_path = os.path.join(base_path, scene_name, "transforms_%s.json" % type)
        with open(json_path, 'r') as f:
            parsed = json.loads(f.read())

        camera_angle_x = parsed['camera_angle_x']

        for frame_info in parsed['frames']:
            p = os.path.join(base_path, scene_name, frame_info['file_path'] + ".png")
            p = os.path.abspath(p)
            self.image_path.append(p)
            self.pose.append(frame_info['transform_matrix'])

        im_sample = tf.image.decode_png(tf.io.read_file(self.image_path[0]))
        H, W, C = im_sample.shape

        self.image_size = [H, W, C]

        f =  (W / 2.) / tf.math.tan(camera_angle_x / 2.)

        intrinsic_list = [[f, 0., 0.],
                          [0., f, 0.],
                          [0., 0., 1.]]

        intrinsic = np.array(intrinsic_list, dtype=np.float32)

        for _ in parsed['frames']:
            self.intrinsic.append(intrinsic)

        self.near = 2.
        self.far = 6.


    def load_image(self, abs_image_path):
        im = NeRFDataset.load_image(abs_image_path)
        if self.white_bkgd:
            im = im[..., :3] * im[..., -1:] + (1. - im[..., -1:])
        else:
            im = im[..., :3]
        return tf.ensure_shape(im, [None, None, 3])
