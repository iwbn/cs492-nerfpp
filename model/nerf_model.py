import tensorflow as tf
# import tensorflow.keras as keras
from argparse import Namespace, ArgumentParser

keras = tf.keras

"""
MLP model class for NeRF network. For exact reproducibility, we follow the same architecture appearing in:
https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py#L80
"""
class NeRFModel(keras.Model):
    def __init__(self, params, *args, **kwargs):
        super(NeRFModel, self).__init__(*args, **kwargs)
        self.params = params

        self.net_depth = params.net_depth
        self.net_width = params.net_width

        self.skips = params.skips

        self.dense_layers = [
            keras.layers.Dense(self.net_width, activation='relu', name="dense_%d"%i)
            for i in range(self.net_depth)
        ]

        self.dense_concat = keras.layers.Concatenate(axis=-1, name="concat_skip")

        # if use viewdirs
        self.alpha_out = keras.layers.Dense(1, activation=None, name="dense_alpha")
        self.bottleneck = keras.layers.Dense(256, activation=None, name="dense_bottleneck")
        self.viewdirs_concat = keras.layers.Concatenate(axis=-1, name="concat_viewdirs")

        self.output_layers = tf.keras.Sequential([
            keras.layers.Dense(self.net_width // 2, activation='relu', name="net_output_0"),
            keras.layers.Dense(3, activation=None, name="net_output_1"),
        ])

        self.output_concat = keras.layers.Concatenate(axis=-1, name="concat_output")

    def call(self, inputs, training=None, mask=None):
        points, view_dir = inputs

        net = points
        for i in range(len(self.dense_layers)):
            net = self.dense_layers[i](net)
            if i in self.skips:
                net = self.dense_concat([points, net])

        alpha_out = self.alpha_out(net)
        bottleneck = self.bottleneck(net)
        inp_viewdirs = self.viewdirs_concat([bottleneck, view_dir])
        net = self.output_layers(inp_viewdirs)

        outputs = self.output_concat([net, alpha_out])

        return outputs

    @staticmethod
    def get_argparse(parser=None):
        if parser is None:
            parser = ArgumentParser(add_help=False)

        """
        We follow the conventional names for optional variables appearing in NeRF GitHub.
        https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf.py#L461
        """
        parser.add_argument("--net_depth", type=int, default=8, help="")
        parser.add_argument("--net_width", type=int, default=256, help="")
        parser.add_argument("--skips", type=int, nargs='+', default=[4], help="")

        parser.add_argument('--use_viewdirs', action="store_true", help='')
        return parser

"""
MLP model class for vanilla NeRF. All inputs in the beginning of the network.
"""
class VanillaNeRFModel(NeRFModel):
    def call(self, inputs, training=None, mask=None):
        points, view_dir = inputs

        net = tf.concat([points, view_dir], axis=-1)
        for i in range(len(self.dense_layers)):
            net = self.dense_layers[i](net)
            if i in self.skips:
                net = self.dense_concat([points, net])

        alpha_out = self.alpha_out(net)
        bottleneck = self.bottleneck(net)
        net = self.output_layers(bottleneck)

        outputs = self.output_concat([net, alpha_out])

        return outputs