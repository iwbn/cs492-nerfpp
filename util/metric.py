import tensorflow as tf
import util.lpips_tfv2


class PSNR(tf.keras.metrics.Metric):
    def __init__(self, name='PSNR', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.mean = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
        y_pred = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        psnr = -10. * tf.math.log(mse) / tf.math.log(10.)
        self.mean.update_state(psnr)

    def result(self):
        res = self.mean.result()
        return res

    def reset_state(self):
        self.mean.reset_state()
        super(PSNR, self).reset_state()



class SSIM(tf.keras.metrics.Metric):
    def __init__(self, name='SSIM', **kwargs):
        super(SSIM, self).__init__(name=name, **kwargs)
        self.mean = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim = tf.image.ssim(y_true*255., y_pred*255., max_val=255.0, filter_size=11,
                      filter_sigma=1.5, k1=0.01, k2=0.03)
        self.mean.update_state(ssim)

    def result(self):
        res = self.mean.result()
        return res

    def reset_state(self):
        self.mean.reset_state()
        super(SSIM, self).reset_state()


class MS_SSIM(tf.keras.metrics.Metric):
    def __init__(self, name='MS_SSIM', **kwargs):
        super(MS_SSIM, self).__init__(name=name, **kwargs)
        self.mean = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        ssim = tf.image.ssim_multiscale(y_true*255., y_pred*255., max_val=255.0, filter_size=11,
                      filter_sigma=1.5, k1=0.01, k2=0.03)
        self.mean.update_state(ssim)

    def result(self):
        res = self.mean.result()
        return res

    def reset_state(self):
        self.mean.reset_state()
        super(MS_SSIM, self).reset_state()

class LPIPS(tf.keras.metrics.Metric):
    def __init__(self, name='LPIPS', model='net-lin', net='alex', version=0.1, **kwargs):
        super(LPIPS, self).__init__(name=name, **kwargs)
        self.mean = tf.keras.metrics.Mean()
        self.lpips_fn = util.lpips_tfv2.get_lpips_fn(model=model, net=net, version=version)

    def update_state(self, y_true, y_pred, sample_weight=None):
        lpips = self.lpips_fn(y_true, y_pred)
        self.mean.update_state(lpips)

    def result(self):
        res = self.mean.result()
        return res

    def reset_state(self):
        self.mean.reset_state()
        super(LPIPS, self).reset_state()