import box
import tensorflow as tf
# import tensorflow.keras as keras
from argparse import Namespace, ArgumentParser
from box import Box

keras = tf.keras
Lambda = keras.layers.Lambda

"""
Embedder class reference:
https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py#L22
This module is implemented using the options in the reference code for better reproducibility.
"""
class Embedder(keras.Model):
    def __init__(self, multires, **kwargs):
        super(Embedder, self).__init__(**kwargs)

        # brought from original NeRF
        self.args = {
            'include_input': True,
            'max_freq_log2': multires-1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [tf.math.sin, tf.math.cos],
        }
        self.args = Box(**self.args)

        self.max_freq = self.args.max_freq_log2
        self.n_freqs = self.args.num_freqs
        self.log_sampling = self.args.log_sampling
        self.include_input = self.args.include_input
        self.periodic_fns = self.args.periodic_fns

        self.embed_functions = self._get_embedding_fn()

    def _get_embedding_fn(self):
        embed_fns = []
        if self.include_input:
            embed_fns.append(Lambda(lambda x: x))

        if self.log_sampling:
            freq_bands = 2.**tf.linspace(0., self.max_freq, self.n_freqs)
        else:
            freq_bands = tf.linspace(2.**0., 2.**self.max_freq, self.n_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(Lambda(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq)))

        return embed_fns


    def call(self, inputs, training=None, mask=None):
        to_concat = [fn(inputs) for fn in self.embed_functions]
        output = tf.concat(to_concat, axis=-1)
        return output
