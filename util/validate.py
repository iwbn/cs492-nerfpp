import tensorflow as tf
from data.nerf_synthetic import NerfSynthetic
from data.nerf_pp import NerfPPData
from util.metric import PSNR, SSIM

class ValidateCallback(tf.keras.callbacks.Callback):
    def __init__(self, skip_on_start=False):
        super(ValidateCallback, self).__init__()
        self.skip_on_start = skip_on_start

class ValidateNeRFCallback(ValidateCallback):
    def __init__(self, tb_callback:tf.keras.callbacks.TensorBoard, dataset_name:str, save_image_every=100, max_outputs=3, smallset=False, **kwargs):
        super(ValidateNeRFCallback, self).__init__(**kwargs)
        self.batch = 0
        self.tb_callback = tb_callback
        self.dataset_name = dataset_name
        self.save_image_every = save_image_every
        self.max_image_outputs = max_outputs
        self.smallset = smallset
        self.metrics = [tf.keras.metrics.RootMeanSquaredError(), SSIM(), PSNR()]
        self.dataset_fn = None

    def set_model(self, model):
        self.model = model
        self.model_call = tf.function(lambda x: model.call(x)['rgb'])
        self._train_step = self.model._train_counter
        obj = self.dataset_fn()
        dataset = obj.ray_dataset
        if self.smallset:
            dataset = dataset.take(self.max_image_outputs)
        self.dataset = dataset

    def on_train_begin(self, logs=None):
        if self.model.optimizer.iterations == 0:
            y_trues = []
            for i, d in enumerate(self.dataset.batch(1).prefetch(self.max_image_outputs)):
                if i >= self.max_image_outputs: break
                x, y = d
                y_true = y
                y_trues.append(y_true)

            y_trues = tf.concat(y_trues, axis=0)
            with self.tb_callback._val_writer.as_default():
                tf.summary.image("batch_%s_gt" % (self.dataset_name), y_trues,
                                 step=self.model.optimizer.iterations, max_outputs=self.max_image_outputs)

    def on_epoch_begin(self, epoch, logs=None):
        if not self.skip_on_start and self.model.optimizer.iterations == 0:
            self.on_epoch_end(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        iteration = self.model.optimizer.iterations
        dataset = self.dataset.batch(1).prefetch(10)
        rgb_preds = []
        for i, (x, y) in enumerate(dataset):
            pred = self.model_call(x)

            for metric in self.metrics:
                metric.update_state(y, pred)

            if i < self.max_image_outputs:
                rgb_preds.append(pred)

        rgb_preds = tf.concat(rgb_preds, axis=0)
        with self.tb_callback._val_writer.as_default():
            tf.summary.image("batch_%s" % (self.dataset_name), rgb_preds, step=iteration,
                             max_outputs=self.max_image_outputs)
            for metric in self.metrics:
                tf.summary.scalar("batch_%s" % metric.name, metric.result(), step=iteration)
                metric.reset_state()

class ValidateOnLego(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnLego, self).__init__(*args, dataset_name="lego", **kwargs)

        def _get_dataset():
            d = NerfSynthetic('nerfdata/nerf_synthetic', 'lego', 'val', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfSynthetic('nerfdata/nerf_synthetic', 'lego', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset


class ValidateOnAfrica(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnAfrica, self).__init__(*args, dataset_name="africa", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'africa', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'africa', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset


class ValidateOnBasket(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnBasket, self).__init__(*args, dataset_name="basket", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'basket', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'basket', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset


class ValidateOnShip(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnShip, self).__init__(*args, dataset_name="ship", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'ship', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'ship', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset


class ValidateOnStatue(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnStatue, self).__init__(*args, dataset_name="statue", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'statue', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'statue', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset


class ValidateOnTorch(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnTorch, self).__init__(*args, dataset_name="torch", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'torch', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'torch', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset


class ValidateOnM60(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnM60, self).__init__(*args, dataset_name="M60", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'tat_intermediate_M60', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'tat_intermediate_M60', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset

class ValidateOnPlayground(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnPlayground, self).__init__(*args, dataset_name="playground", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'tat_intermediate_Playground', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'tat_intermediate_Playground', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset

class ValidateOnTrain(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnTrain, self).__init__(*args, dataset_name="train", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'tat_intermediate_Train', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'tat_intermediate_Train', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset

class ValidateOnTruck(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnTruck, self).__init__(*args, dataset_name="truck", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'tat_training_Truck', 'test', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'tat_training_Truck', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset

class ValidateOnScan65(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnScan65, self).__init__(*args, dataset_name="scan65", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'scan65_paper', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'scan65_paper', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset

class ValidateOnScan106(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnScan106, self).__init__(*args, dataset_name="scan106", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'scan106_paper', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'scan106_paper', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset


class ValidateOnScan118(ValidateNeRFCallback):
    def __init__(self, *args, normalize_coordinates=False, **kwargs):
        super(ValidateOnScan118, self).__init__(*args, dataset_name="scan118", **kwargs)

        def _get_dataset():
            d = NerfPPData('nerfppdata', 'scan118_paper', 'validation', normalize_coordinates=normalize_coordinates, white_bkgd=True,)
            if normalize_coordinates:
                d_train = NerfPPData('nerfppdata', 'scan118_paper', 'train', white_bkgd=True,)
                d_train.compute_inside()
                d.center = d_train.center
                d.radius = d_train.radius
            return d

        self.dataset_fn = _get_dataset