import tensorflow as tf
#import tensorflow.keras as keras
from .nerf_model import NeRFModel
from util.camera_tf import get_intersaction_sphere
from util.embedder import Embedder
from argparse import Namespace, ArgumentParser
from tensorflow.python.keras.engine import data_adapter
import util.parallel

keras = tf.keras
"""
MLP degenrate model class
"""
class NeRFDegenerate(keras.Model):
    def __init__(self, params, *args, **kwargs):
        super(NeRFDegenerate, self).__init__(*args, **kwargs)
        self.params = params

        self.nerf_model = NeRFModel(self.params)

        self.embedder = Embedder(self.params.embed_multires)
        self.view_embedder = Embedder(self.params.embed_multires_views)

        self.near=0.0
        self.far=1.0

    def call(self, inputs, training=None, mask=None):
        rays_o, rays_d = inputs

        s_o = tf.unstack(tf.shape(rays_o))
        s_d = tf.unstack(tf.shape(rays_d))

        rays_o = tf.reshape(rays_o, [-1, rays_o.shape[-1]])
        rays_d = tf.reshape(rays_d, [-1, rays_d.shape[-1]])

        render_rays_fn = lambda inputs: self.render_rays(inputs[0], inputs[1], self.near, self.far, training)
        output_signature = {'rgb':tf.float32, 'w':tf.float32}

        ret = util.parallel.batchfy(render_rays_fn, (rays_o, rays_d), output_signature, chunk=4096)

        rgb_rendered = ret['rgb']

        rgb_rendered = tf.reshape(rgb_rendered, s_o[:-1] + tf.unstack(tf.shape(rgb_rendered))[-1:])
        call_ret = {'rgb' : rgb_rendered}

        return call_ret

    def batchfy(self, net_inputs, model, chunk = 1024 * 32):
        outputs = util.parallel.batchfy(model, net_inputs, output_signature=tf.float32, chunk=chunk)

        return outputs

    def render_rays(self, rays_o, rays_d, near, far, importance, training=None):
        B = tf.unstack(tf.shape(rays_o))[0]

        # we use a sphere radius 2.0 to build a degenerate NeRF model
        _, i_point, valid = get_intersaction_sphere(rays_o, rays_d, tf.zeros([3], tf.float32), 2.0, only_forward=False, get_mask=True)
        # i_point: n_rays x 3
        # viewdier: n_rays x 3
        # valid: n_rays x 1

        viewdirs = rays_d
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)

        point_embed = self.embedder(i_point)
        viewdirs_embed = self.view_embedder(viewdirs)

        outputs = self.nerf_model([point_embed, viewdirs_embed], training=True)

        out_alpha = outputs[..., 3]  # we do not use this alpha value here.
        out_rgb = outputs[..., 0:3]
        rgb = tf.math.sigmoid(out_rgb)

        rgb = rgb * valid + tf.ones_like(rgb) * (1.-valid)

        ret = {'rgb' : rgb, 'w' : tf.zeros_like(rgb)}

        return ret


    @staticmethod
    def get_argparse(parser=None):
        if parser is None:
            parser = ArgumentParser(add_help=False)
        parser = NeRFModel.get_argparse(parser)

        parser.add_argument("--chunk_size", type=int, default=32*1024, help="")
        parser.add_argument("--n_samples", type=int, default=64, help="")
        parser.add_argument("--embed_multires", type=int, default=10, help="")
        parser.add_argument("--embed_multires_views", type=int, default=4, help="")
        parser.add_argument('--use_white_bkgd', action="store_true", help='')
        return parser
