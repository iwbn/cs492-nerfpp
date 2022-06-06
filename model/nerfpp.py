import tensorflow as tf
#import tensorflow.keras as keras
from .nerf_model import NeRFModel
from util.embedder import Embedder
from argparse import Namespace, ArgumentParser
from tensorflow.python.keras.engine import data_adapter
import util.parallel
from util.camera_tf import get_intersaction_sphere, get_reparam_point
from util.random_tf import sample_from_pdf


keras = tf.keras
"""
NeRF++ model
"""
class NeRFPP(keras.Model):
    def __init__(self, params, *args, **kwargs):
        super(NeRFPP, self).__init__(*args, **kwargs)
        self.params = params

        # we use the same model class in NeRF++ and NeRF
        self.nerf_model = NeRFModel(self.params)
        self.nerf_model_outside = NeRFModel(self.params)  # separate model for outside

        # positional embedder
        self.embedder = Embedder(self.params.embed_multires)
        self.view_embedder = Embedder(self.params.embed_multires_views)

        self.embedder_outside = Embedder(self.params.embed_multires_outside)
        self.view_embedder_outside = Embedder(self.params.embed_multires_views_outside)

        # we should set these values before training or inference
        self.center = None
        self.radius = None
        self.near = None

    def call(self, inputs, training=None, mask=None, n_importance=-1):
        rays_o, rays_d = inputs

        s_o = tf.unstack(tf.shape(rays_o))
        s_d = tf.unstack(tf.shape(rays_d))

        rays_o = tf.reshape(rays_o, [-1, rays_o.shape[-1]])
        rays_d = tf.reshape(rays_d, [-1, rays_d.shape[-1]])

        if n_importance < 0:
            n_importance = self.params.n_importance

        render_rays_fn = lambda inputs, n_importance=n_importance: \
            self.render_rays(inputs[0], inputs[1], training, n_importance=n_importance)
        output_signature = {'rgb': tf.float32, 'w': tf.float32}

        if n_importance > 0:
            output_signature['w_imp'] = tf.float32
            output_signature['rgb_imp'] = tf.float32

        ret = util.parallel.batchfy(render_rays_fn, (rays_o, rays_d), output_signature, chunk=4096)

        rgb_rendered = ret['rgb']


        rgb_rendered = tf.reshape(rgb_rendered, s_o[:-1] + tf.unstack(tf.shape(rgb_rendered))[-1:])
        call_ret = {'rgb': rgb_rendered, 'w': ret['w']}

        if 'rgb_imp' in ret:
            rgb_imp_rendered = ret['rgb_imp']

            rgb_imp_rendered = tf.reshape(rgb_imp_rendered, s_o[:-1] + tf.unstack(tf.shape(rgb_imp_rendered))[-1:])
            call_ret['rgb_0'] = call_ret['rgb']
            call_ret['rgb'] = rgb_imp_rendered

        return call_ret

    def batchfy(self, net_inputs, model, chunk = 1024 * 32):
        outputs = util.parallel.batchfy(model, net_inputs, output_signature=tf.float32, chunk=chunk)

        return outputs

    def make_outputs(self, points, out_points, in_z_max, viewdirs, in_z_vals,
                     prev_in_zvals=None, prev_out_points=None, prev_alpha=None, prev_rgb=None):

        out_z_vals = out_points[...,-1]

        point_embed = self.embedder(points)
        viewdirs_embed = self.view_embedder(viewdirs)
        viewdirs_embed = tf.tile(viewdirs_embed, [1, tf.shape(point_embed)[-2], 1])

        point_out_embed = self.embedder_outside(out_points)
        viewdirs_out_embed = self.view_embedder_outside(viewdirs)
        viewdirs_out_embed = tf.tile(viewdirs_out_embed, [1, tf.shape(point_out_embed)[-2], 1])

        net_inputs_in = [point_embed, viewdirs_embed]
        net_inputs_out = [point_out_embed, viewdirs_out_embed]

        chunk = self.params.chunk_size
        outputs_in = self.batchfy(net_inputs_in, self.nerf_model, chunk)
        outputs_out = self.batchfy(net_inputs_out, self.nerf_model_outside, chunk)

        inner_alpha_raw = outputs_in[..., 3]
        inner_rgb_raw = outputs_in[..., :3]

        if prev_rgb is not None:
            prev_inner_alpha_raw, prev_outer_alpha_raw = tf.split(prev_alpha, num_or_size_splits=2, axis=-1)
            prev_inner_rgb_raw, prev_outer_rgb_raw = tf.split(prev_rgb, num_or_size_splits=2, axis=-2)

            inner_alpha_raw = tf.concat([inner_alpha_raw, prev_inner_alpha_raw], axis=-1)
            inner_rgb_raw = tf.concat([inner_rgb_raw, prev_inner_rgb_raw], axis=-2)
            in_z_vals = tf.concat([in_z_vals, prev_in_zvals], axis=-1)

            indices = tf.argsort(in_z_vals, axis=-1, direction="ASCENDING")

            inner_alpha_raw = tf.gather(inner_alpha_raw, indices, axis=-1, batch_dims=len(inner_alpha_raw.shape) - 1)
            inner_rgb_raw = tf.gather(inner_rgb_raw, indices, axis=-2, batch_dims=len(inner_rgb_raw.shape) - 2)
            in_z_vals = tf.gather(in_z_vals, indices, axis=-1, batch_dims=len(in_z_vals.shape) - 1)

        inner_dists = in_z_vals[..., 1:] - in_z_vals[..., :-1]
        inner_dists = tf.concat(
            [inner_dists, in_z_max - in_z_vals[..., -1:]],
            axis=-1)

        inner_alpha = 1.0 - tf.math.exp(-tf.nn.relu(inner_alpha_raw) * inner_dists)
        inner_trans = tf.math.cumprod(1. - inner_alpha + 1e-10, axis=-1, exclusive=False)
        last_trans = inner_trans[...,-1:]
        inner_trans = tf.concat([tf.ones_like(inner_trans[...,:1]), inner_trans[...,:-1]], axis=-1)
        inner_weights = inner_alpha * inner_trans

        inner_rgb = tf.math.sigmoid(inner_rgb_raw)
        inner_rgb_rendered = tf.reduce_sum(inner_weights[..., tf.newaxis] * inner_rgb, axis=-2)

        acc_weight = tf.reduce_sum(inner_weights, -1, keepdims=True)

        if self.params.use_white_bkgd:
            inner_rgb_rendered = inner_rgb_rendered + (1. - acc_weight)

        outer_alpha_raw = outputs_out[..., 3]
        outer_rgb_raw = outputs_out[..., :3]

        if prev_rgb is not None:
            prev_out_zvals = prev_out_points[..., -1]
            outer_alpha_raw = tf.concat([outer_alpha_raw, prev_outer_alpha_raw], axis=-1)
            outer_rgb_raw = tf.concat([outer_rgb_raw, prev_outer_rgb_raw], axis=-2)
            out_z_vals = tf.concat([out_z_vals, prev_out_zvals], axis=-1)

            indices = tf.argsort(out_z_vals, axis=-1, direction="DESCENDING")

            outer_alpha_raw = tf.gather(outer_alpha_raw, indices, axis=-1, batch_dims=len(outer_alpha_raw.shape) - 1)
            outer_rgb_raw = tf.gather(outer_rgb_raw, indices, axis=-2, batch_dims=len(outer_rgb_raw.shape) - 2)
            out_z_vals = tf.gather(out_z_vals, indices, axis=-1, batch_dims=len(out_z_vals.shape) - 1)

        outer_dists = -out_z_vals[..., 1:] + out_z_vals[..., :-1]
        outer_dists = tf.concat(
            [outer_dists, tf.ones_like(outer_dists[..., :1]) * 1e+10],
            axis=-1)

        outer_alpha = 1.0 - tf.math.exp(-tf.nn.relu(outer_alpha_raw) * outer_dists)
        outer_trans = tf.math.cumprod(1. - outer_alpha + 1e-10, axis=-1, exclusive=True)

        outer_weights = outer_alpha * outer_trans

        outer_rgb = tf.math.sigmoid(outer_rgb_raw)
        outer_rgb_rendered = tf.reduce_sum(outer_weights[..., tf.newaxis] * outer_rgb, axis=-2)

        acc_weight = tf.reduce_sum(outer_weights, -1, keepdims=True)

        if self.params.use_white_bkgd:
            outer_rgb_rendered = outer_rgb_rendered + (1. - acc_weight)

        rgb_rendered = inner_rgb_rendered + last_trans * outer_rgb_rendered
        weights = tf.concat([inner_weights, outer_weights], axis=-1)

        ret_alpha_raw = tf.concat([inner_alpha_raw, outer_alpha_raw], axis=-1)
        ret_rgb_raw = tf.concat([inner_rgb_raw, outer_rgb_raw], axis=-2)

        return rgb_rendered, weights, ret_alpha_raw, ret_rgb_raw

    def perturb_zvals(self, z_vals, training):
        if training and self.params.use_perturb > 0.:
            z_vals_perturb = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = tf.concat([z_vals_perturb, z_vals[..., -1:]], -1)
            lower = tf.concat([z_vals[..., :1], z_vals_perturb], -1)

            # stratified samples in those intervals
            t_rand = tf.random.uniform(tf.shape(z_vals))
            z_vals = lower + (upper - lower) * t_rand

        return z_vals

    def render_rays(self, rays_o, rays_d, training=None, return_points=False, n_importance=0):
        B = tf.unstack(tf.shape(rays_o))[0]

        in_points, in_z_vals = self.get_inner_coords(rays_o, rays_d, training=training)
        out_points, real_out_zvals = self.get_outer_coords(rays_o, rays_d, self.params.n_samples, training=training)

        viewdirs = rays_d[:, tf.newaxis]
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)

        rgb_rendered, weights, prev_alpha, prev_rgb = \
            self.make_outputs(in_points, out_points, real_out_zvals[...,:1], viewdirs, in_z_vals,
                     prev_in_zvals=None, prev_out_points=None, prev_alpha=None, prev_rgb=None)
        ret = {'rgb': rgb_rendered, 'w': weights}

        if return_points:
            ret['in_points'] = in_points
            ret['out_points'] = out_points[...,:3]
            ret['in_z_vals'] = in_z_vals
            ret['out_zvals'] = real_out_zvals

        if n_importance > 0:
            in_weights = weights[..., 0:tf.shape(in_z_vals)[-1]][..., 1:-1]
            out_weights = weights[..., tf.shape(in_z_vals)[-1]:][..., 1:-1]
            in_bins = (in_z_vals[..., 1:] + in_z_vals[..., :-1]) / 2.
            out_bins = (out_points[...,-1][..., 1:] + out_points[...,-1][..., :-1]) / 2.

            in_z_vals_a = sample_from_pdf(in_bins, in_weights, n_samples=n_importance,
                                          deterministic=(not training or not self.params.use_perturb))

            out_one_over_r_a = sample_from_pdf(tf.reverse(out_bins, axis=[-1]), tf.reverse(out_weights, axis=[-1]),
                                          n_samples=n_importance,
                                          deterministic=(not training or not self.params.use_perturb))

            out_one_over_r_a = tf.reverse(out_one_over_r_a, axis=[-1])
            out_points_a, real_out_zvals_a = self.get_outer_coords(rays_o, rays_d, n_importance,
                                                                   one_over_r=out_one_over_r_a, training=training)
            in_z_vals_a = tf.stop_gradient(in_z_vals_a)
            out_points_a = tf.stop_gradient(out_points_a)

            in_points_a = rays_o[..., tf.newaxis, :] + viewdirs * in_z_vals_a[..., :, tf.newaxis]

            rgb_rendered, weights, _, _ = \
                self.make_outputs(in_points_a, out_points_a, real_out_zvals[...,:1], viewdirs, in_z_vals_a,
                     prev_in_zvals=in_z_vals, prev_out_points=out_points, prev_alpha=prev_alpha, prev_rgb=prev_rgb)

            ret['rgb_imp'] = rgb_rendered
            ret['w_imp'] = weights
            if return_points:
                ret['in_points_a'] = in_points_a
                ret['out_points_a'] = out_points_a[..., :3]
                ret['in_zvals_a'] = in_z_vals_a
                ret['out_zvals_a'] = real_out_zvals_a

        return ret

    def get_inner_coords(self, rays_o, rays_d, training=None):
        # rays_o: N_rays x 3
        # rays_d: N_rays x 3
        # self.center: 3
        # self.radius: 1

        center = self.center[tf.newaxis]
        pnt_a, pnt_b = get_intersaction_sphere(rays_o, rays_d, center, self.radius, only_forward=False)
        t = tf.linalg.norm(pnt_a - rays_o, axis=-1, keepdims=True)
        t_b = tf.linalg.norm(pnt_b - rays_o, axis=-1, keepdims=True)
        inp_d = tf.reduce_sum((pnt_b - rays_o) * rays_d, axis=-1, keepdims=True)

        viewdirs = rays_d
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)

        near_dist = tf.linalg.norm(self.near * rays_d, axis=-1, keepdims=True)
        near_dist = tf.minimum(near_dist, t)
        near_dist = tf.where(inp_d < 0., near_dist, t_b)

        # todo: compare to the paper
        pnt_near = rays_o + viewdirs * near_dist
        pnt_far = rays_o + t * viewdirs

        t_vals = tf.linspace(0., 1., self.params.n_samples)
        t_vals = t_vals[tf.newaxis]
        z_vals = near_dist + \
                 tf.linalg.norm(pnt_far - pnt_near, axis=-1, keepdims=True) * t_vals

        z_vals = self.perturb_zvals(z_vals, training)

        points = rays_o[..., None, :] + viewdirs[..., None, :] * z_vals[..., :, None]

        return points, z_vals


    def get_outer_coords(self, rays_o, rays_d, n_samples, one_over_r=None, training=None):
        # rays_o: N_rays x 3
        # rays_d: N_rays x 3
        # self.center: 3
        # self.radius: 1

        center = self.center[tf.newaxis]
        pnt_a = get_intersaction_sphere(rays_o, rays_d, center, self.radius)

        viewdirs = rays_d
        viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)

        _nm = lambda x: tf.linalg.norm(x, axis=-1, keepdims=True)
        oa = viewdirs
        ca = pnt_a - center

        cos_t = tf.reduce_sum(oa * ca, axis=-1, keepdims=True) / _nm(oa) / _nm(ca)
        theta = tf.math.acos(cos_t)

        # todo: compare to the paper
        t_vals = tf.linspace(0., 1., self.params.n_samples)
        t_vals = t_vals[tf.newaxis]
        t_vals = self.perturb_zvals(t_vals, training)
        if one_over_r is None:
            one_over_r = tf.reverse(t_vals, axis=[-1])
            one_over_r = tf.reshape(one_over_r, [1, n_samples])
            one_over_r = tf.tile(one_over_r, [tf.shape(rays_d)[0], 1])
        one_over_r = tf.clip_by_value(one_over_r, 0., 1.)

        points_z = get_reparam_point(rays_o, pnt_a, rays_d, one_over_r, self.center, self.radius)
        one_over_r = one_over_r[...,None]

        ret_points = tf.concat([points_z, one_over_r], axis=-1)

        real_z_vals = (ret_points[..., :-1, :3] - center)
        real_z_vals *= (tf.math.divide_no_nan(1.,one_over_r[..., :-1,:])*self.radius)
        real_z_vals -= (rays_o - center)[...,None,:]

        real_z_vals = tf.linalg.norm(real_z_vals, axis=-1, keepdims=False)
        real_z_vals = tf.concat([real_z_vals, tf.ones_like(real_z_vals[...,:1])*1e+10], axis=-1)

        return ret_points, real_z_vals # 0 to 1

    """
    Custom training loop. Keras will call this function in each training iteration.
    """
    def train_step(self, data):
        rays_o, rays_d, y_rgb, sample_weight = self.parse_inputs(data)

        iterations = self.optimizer.iterations
        # Collect metrics to return
        return_metrics = {}

        with tf.GradientTape() as tape:
            ret = self.call((rays_o, rays_d), training=True, n_importance=0)
            y_pred = ret['rgb']

            mask = tf.logical_or(tf.math.is_nan(y_pred), tf.math.is_inf(y_pred))
            mask = 1. - tf.cast(mask, y_pred.dtype)
            y_pred = tf.where(mask > 0.5, y_pred, tf.zeros_like(y_pred))

            loss = self.compiled_loss(
                y_rgb * mask, y_pred, sample_weight, regularization_losses=self.losses)

        self.optimizer.minimize(loss, self.trainable_weights, tape=tape)
        return_metrics['rgb_loss'] = tf.reduce_mean(loss)

        # for debugging
        return_metrics['in_weight_mean'] = tf.reduce_mean(ret['w'][...,0:self.params.n_samples])
        return_metrics['out_weight_mean'] = tf.reduce_mean(ret['w'][...,self.params.n_samples:])
        return_metrics['rgb_loss_nan_mask'] = tf.reduce_sum(1. - mask)

        if self.params.n_importance > 0:
            self.optimizer.iterations.assign_add(-1, read_value=False)  # to prevent wrong step count
            with tf.GradientTape() as tape:
                y_pred = self.call((rays_o, rays_d), training=True, n_importance=self.params.n_importance)['rgb']

                mask = tf.logical_or(tf.math.is_nan(y_pred), tf.math.is_inf(y_pred))
                mask = 1. - tf.cast(mask, y_pred.dtype)
                y_pred = tf.where(mask > 0.5, y_pred, tf.zeros_like(y_pred))

                loss = self.compiled_loss(
                    y_rgb * mask, y_pred, sample_weight, regularization_losses=self.losses)

            self.optimizer.minimize(loss, self.trainable_weights, tape=tape)
            return_metrics['rgb_imp_loss'] = tf.reduce_mean(loss)
            return_metrics['rgb_imp_loss_nan_mask'] = tf.reduce_sum(1. - mask)

        self.compiled_metrics.update_state(y_rgb, y_pred, sample_weight)

        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    @staticmethod
    def parse_inputs(data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

        rays_o = x[0]
        rays_d = x[1]
        y_rgb = y

        return rays_o, rays_d, y_rgb, sample_weight

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
        parser.add_argument("--embed_multires_outside", type=int, default=10, help="")
        parser.add_argument("--embed_multires_views_outside", type=int, default=4, help="")
        parser.add_argument("--n_importance", type=int, default=0, help="")
        parser.add_argument('--use_white_bkgd', action="store_true", help='')
        parser.add_argument('--use_perturb', action="store_true", help='')
        return parser
