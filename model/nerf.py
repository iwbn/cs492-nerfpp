import tensorflow as tf
#import tensorflow.keras as keras
from .nerf_model import NeRFModel, VanillaNeRFModel
from util.embedder import Embedder
from argparse import Namespace, ArgumentParser
from tensorflow.python.keras.engine import data_adapter
import util.parallel
from util.random_tf import sample_from_pdf

keras = tf.keras

"""
NeRF model
"""
class NeRF(keras.Model):
    def __init__(self, params, *args, **kwargs):
        super(NeRF, self).__init__(*args, **kwargs)
        self.params = params

        if self.params.use_vanilla:
            self.nerf_model = VanillaNeRFModel(self.params)
        else:
            self.nerf_model = NeRFModel(self.params)
        if self.params.n_importance > 0:
            if self.params.use_vanilla:
                self.nerf_model_fine = VanillaNeRFModel(self.params)
            else:
                self.nerf_model_fine = NeRFModel(self.params) ### 어차피 쓰는 parameter이 coarse model과 동일

        self.embedder = Embedder(self.params.embed_multires)
        self.view_embedder = Embedder(self.params.embed_multires_views)

        # these lines should be updated before training
        self.near=0.0
        self.far=1.0


    def call(self, inputs, training=None, mask=None):
        rays_o, rays_d = inputs

        s_o = tf.unstack(tf.shape(rays_o))
        s_d = tf.unstack(tf.shape(rays_d))

        rays_o = tf.reshape(rays_o, [-1, rays_o.shape[-1]])
        rays_d = tf.reshape(rays_d, [-1, rays_d.shape[-1]])

        render_rays_fn = lambda inputs: self.render_rays(inputs[0], inputs[1], self.near, self.far, training)
        output_signature = {'rgb':tf.float32, 'w':tf.float32, 'points':tf.float32}

        if self.params.n_importance > 0:
            output_signature['w'] = tf.float32
            output_signature['rgb_imp'] = tf.float32

        chunk = self.params.chunk_size
        ret = util.parallel.batchfy(render_rays_fn, (rays_o, rays_d), output_signature, chunk=chunk)

        rgb_rendered = ret['rgb']

        rgb_rendered = tf.reshape(rgb_rendered, s_o[:-1] + tf.unstack(tf.shape(rgb_rendered))[-1:])
        call_ret = {'rgb' : rgb_rendered}

        if 'rgb_imp' in ret:
            rgb_imp_rendered = ret['rgb_imp']


            rgb_imp_rendered = tf.reshape(rgb_imp_rendered, s_o[:-1] + tf.unstack(tf.shape(rgb_imp_rendered))[-1:])
            call_ret['rgb_0'] = call_ret['rgb']
            call_ret['rgb'] = rgb_imp_rendered

        return call_ret


    def sample_pdf(self, bins, weights, N_samples, det=False):
        samples = sample_from_pdf(bins, weights, n_samples=N_samples, deterministic=det)
        return samples


    def batchfy(self, net_inputs, model, chunk = 1024 * 32):
        outputs = util.parallel.batchfy(model, net_inputs, output_signature=tf.float32, chunk=chunk)

        return outputs

    """
    We re-implemented this function with reference function raw2outputs in NeRF code
    Reference code: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L93
    """
    def make_outputs(self, model, points, rays_d, viewdirs, z_vals):
        point_embed = self.embedder(points)
        viewdirs_embed = self.view_embedder(viewdirs)
        viewdirs_embed = tf.tile(viewdirs_embed, [1, tf.shape(point_embed)[-2], 1])

        net_inputs = [point_embed, viewdirs_embed]

        chunk = self.params.chunk_size
        outputs = self.batchfy(net_inputs, model, chunk)

        out_alpha = outputs[..., 3]
        out_rgb = outputs[..., 0:3]

        dists = z_vals[..., 1:] - z_vals[..., :-1]

        dists = tf.concat([dists, 1e10 * tf.ones_like(dists[..., :1])], axis=-1)

        dists = dists * tf.linalg.norm(rays_d[..., None, :], axis=-1)

        alpha = 1.0 - tf.math.exp(-tf.nn.relu(out_alpha) * dists)

        # B x n_samples
        weights = alpha * tf.math.cumprod(1. - alpha + 1e-10, axis=-1, exclusive=True)

        rgb = tf.math.sigmoid(out_rgb)
        rgb_rendered = tf.reduce_sum(weights[..., tf.newaxis] * rgb, axis=-2)

        acc_weight = tf.reduce_sum(weights, -1, keepdims=True)

        # white background rendering
        # reference: https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L160
        if self.params.use_white_bkgd:
            rgb_rendered = rgb_rendered + (1. - acc_weight)
        
        return rgb_rendered, weights 


    def render_rays(self, rays_o, rays_d, near, far, importance, training=None):
        B = tf.unstack(tf.shape(rays_o))[0]

        near = near * tf.ones_like(rays_d[..., :1])
        far = far * tf.ones_like(rays_d[..., :1])

        t_vals = tf.linspace(0., 1., self.params.n_samples)
        if self.params.use_linear_disparity:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))
        else:
            z_vals = near * (1. - t_vals) + far * (t_vals)

        # z_vals: B x n_samples

        if training and self.params.use_perturb > 0.:
            z_vals_perturb = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = tf.concat([z_vals_perturb, z_vals[..., -1:]], -1)
            lower = tf.concat([z_vals[..., :1], z_vals_perturb], -1)

            # stratified samples in those intervals
            t_rand = tf.random.uniform(tf.shape(z_vals))
            z_vals = lower + (upper - lower) * t_rand

        points = rays_o[..., tf.newaxis, :] + rays_d[..., tf.newaxis, :] * z_vals[..., :, tf.newaxis]

        viewdirs = rays_d[..., tf.newaxis, :]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)

        rgb_rendered, weights = self.make_outputs(self.nerf_model, points, rays_d, viewdirs, z_vals)
        
        ret = {'rgb' : rgb_rendered, 'w' : weights}
        ret['points'] = points

        if self.params.n_importance > 0:
            z_vals_imp = (z_vals[...,1:] + z_vals[...,:-1]) * 0.5
            z_samples = self.sample_pdf(z_vals_imp, weights[..., 1:-1], self.params.n_importance,
                                        det=(not training and not self.params.use_perturb))
            z_samples = tf.stop_gradient(z_samples)
            z_vals = tf.sort(tf.concat([z_vals, z_samples], -1), -1)
            points = rays_o[..., tf.newaxis, :] + rays_d[..., tf.newaxis, :] * z_vals[..., :, tf.newaxis]
            ret['points'] = points
            rgb_imp_rendered, weights = self.make_outputs(self.nerf_model_fine, points, rays_d, viewdirs, z_vals)

            ret['w'] = weights
            ret['rgb_imp'] = rgb_imp_rendered
        
        return ret


    @staticmethod
    def get_argparse(parser=None):
        if parser is None:
            parser = ArgumentParser(add_help=False)
        parser = NeRFModel.get_argparse(parser)

        """
        We follow the conventional names for optional variables appearing in NeRF GitHub.
        https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L461
        """
        parser.add_argument("--chunk_size", type=int, default=32*1024, help="")
        parser.add_argument("--n_samples", type=int, default=64, help="")
        parser.add_argument("--embed_multires", type=int, default=10, help="")
        parser.add_argument("--embed_multires_views", type=int, default=4, help="")
        parser.add_argument("--n_importance", type=int, default=0, help="")
        parser.add_argument('--use_white_bkgd', action="store_true", help='')
        parser.add_argument('--use_linear_disparity', action="store_true", help='')
        parser.add_argument('--use_perturb', action="store_true", help='')
        parser.add_argument('--use_vanilla', action="store_true", help='')
        return parser
