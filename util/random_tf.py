import tensorflow as tf


"""
Reference code: 
https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py#L183
"""
def sample_from_pdf(bins, weights, n_samples, deterministic=False):
    """
    :param bins: ..., n_weights
    :param weights:  ..., n_weights-1
    :param n_samples: single integer value
    :param deterministic: bool
    :return:
    """

    s = tf.unstack(tf.shape(weights))
    weights = weights + 1e-10
    pdf = weights / tf.reduce_sum(weights, -1, keepdims=True)
    cdf = tf.cumsum(pdf, -1)
    cdf = tf.concat([tf.zeros_like(cdf[..., :1]), cdf], -1)

    if deterministic:
        uniform_samples = tf.linspace(0., 1., n_samples)
        uniform_samples = tf.broadcast_to(uniform_samples, tf.unstack(tf.shape(cdf))[:-1] + [n_samples])
    else:
        uniform_samples = tf.random.uniform(tf.unstack(tf.shape(cdf))[:-1] + [n_samples])

    indices = tf.searchsorted(cdf, uniform_samples, side='right')
    lower = tf.maximum(0, indices - 1)
    upper = tf.minimum(tf.shape(cdf)[-1] - 1, indices)

    # lower, upper: ..., n_weights+1
    indices_lu = tf.stack([lower, upper], -1)
    cdf_g = tf.gather(cdf, indices_lu, axis=-1, batch_dims=len(s)-1)
    bins_g = tf.gather(bins, indices_lu, axis=-1, batch_dims=len(s)-1)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = tf.where(denom < 1e-10, tf.ones_like(denom), denom)
    t = (uniform_samples - cdf_g[..., 0]) / denom
    final_samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return final_samples


if __name__ == "__main__":
    print(sample_from_pdf(tf.constant([[1.0, 3.0, 4.0],[1.0, 3.0, 4.0]]), tf.constant([[100.0, 1.0],[1.0, 4.0]]), 10))