import tensorflow as tf
import os
from util.camera_tf import get_ray_map
import miniball

Dataset = tf.data.Dataset

"""
Abstract class for NeRF Dataset
"""
class NeRFDataset:
    def __init__(self, shuffle=False, normalize_coordinates=False, **augment_params):
        self.shuffle = shuffle

        self.image_path = []
        self.pose = []
        self.intrinsic = []
        self.normalize_coordinates = normalize_coordinates
        self.__processing_center = False

    @staticmethod
    def load_image(abs_image_path):
        im = _load_image(abs_image_path)
        return im

    @property
    def dataset(self):
        image = Dataset.from_tensor_slices(self.image_path)
        pose = Dataset.from_tensor_slices(self.pose)
        intrinsic = Dataset.from_tensor_slices(self.intrinsic)

        d = Dataset.zip((image, pose, intrinsic))
        if self.shuffle:
            d = d.shuffle(len(image))

        ld_fn = lambda im, pose, intr: (self.load_image(im), pose, intr)
        d = d.map(ld_fn, num_parallel_calls=8)


        def normalize_pose(pose, center, radius):
            rays_o = tf.reshape(pose[:3, 3], [-1])
            center = tf.constant(center, dtype=rays_o.dtype)
            rays_o = rays_o - center
            rays_o = rays_o / (radius)
            rays_o_h = tf.concat([rays_o, [1.]], axis=0)
            rays_o_h = tf.reshape(rays_o_h, [4,1])
            pose_updated = tf.concat([pose[:,0:3], rays_o_h], axis=1)
            return pose_updated

        if self.normalize_coordinates and not self.__processing_center:
            try:
                center = self.center
                radius = self.radius
            except AttributeError:
                self.compute_inside()
                center = self.center
                radius = self.radius

            d = d.map(lambda im, pose, intr: (im, normalize_pose(pose, center, radius), intr))

        d = d.map(lambda im, pose, intr: (im, {'c2w': pose, 'intrinsic': intr}))
        return d

    @property
    def ray_dataset(self):
        d = self.dataset
        d = d.map(lambda x, y: (get_ray_map(x[tf.newaxis], y['intrinsic'][tf.newaxis], y['c2w'][tf.newaxis]), x), num_parallel_calls=32)
        d = d.map(lambda x, y: ((x[0][0], x[1][0]), y), num_parallel_calls=32)
        return d

    @property
    def ray_unbatch_dataset(self):
        d = self.ray_dataset
        d = d.map(lambda x, y: ((tf.reshape(x[0], [-1,3]), tf.reshape(x[1], [-1,3])), tf.reshape(y, [-1,3])))
        d = d.unbatch()
        if self.shuffle:
            d = d.shuffle(1024*1024*64)
        return d

    def get_ray_unbatch_dataset_per_image(self, n_samples=1024):
        def sample_n(*data):
            (x1, x2), y = data
            x1_s = tf.shape(x1)[-1]
            x2_s = tf.shape(x2)[-1]
            y_s = tf.shape(y)[-1]
            x1 = tf.reshape(x1, [-1, x1_s])
            x2 = tf.reshape(x2, [-1, x2_s])
            y = tf.reshape(y, [-1, y_s])

            shuffled = tf.random.shuffle(tf.concat([x1, x2, y], axis=-1))
            sampled = shuffled[0:n_samples]

            x1, x2, y = tf.split(sampled, num_or_size_splits=[3, 3, 3], axis=-1)
            return (x1, x2), y

        d = self.ray_dataset
        d = d.map(sample_n, num_parallel_calls=32)
        return d

    def compute_inside(self):
        self.__processing_center = True
        ray_dataset = self.ray_dataset
        self.__processing_center = False
        origins = []

        center, inside_radius = compute_inside(ray_dataset)

        self.center = center
        self.radius = inside_radius

        return center, inside_radius


def compute_inside(ray_dataset):
    origins = []
    for x, y in ray_dataset.prefetch(10):
        rays_o, rays_d = x
        origins.append(rays_o[0,0])
    C, r2 = miniball.get_bounding_ball(tf.stack(origins).numpy())

    center = tf.constant(C)
    radius = tf.constant(tf.sqrt(r2))
    center = tf.cast(center, tf.float32)
    radius = tf.cast(radius, tf.float32)

    inside_radius = 1.0 * radius

    return center, inside_radius

def _load_image(abs_image_path):
    filename = abs_image_path
    im = tf.image.decode_png(tf.io.read_file(filename))
    if im is None:
        print(filename)
    im = tf.cast(im, tf.float32)/255.0
    return im