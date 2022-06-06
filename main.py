import tensorflow as tf

from model.nerf import NeRF
from model.nerf_degenerate import NeRFDegenerate
from model.nerfpp import NeRFPP
from math import ceil

from data.nerf_synthetic import NerfSynthetic
from data.nerf_pp import NerfPPData
import os
from argparse import ArgumentParser
from util.validate import *
import util.metric
from util.checkpoint import CheckpointManagerCallback


main_parser = ArgumentParser()
def main_parser_def(main_parser):
    main_parser.add_argument("ckpt_path", type=str, help="log and ckpts are saved")
    main_parser.add_argument("--random_seed", type=str, default="1234", help="random seed")
    main_parser.add_argument("--model_type", type=str, default="nerf", help="nerf | nerfpp | nerf-degen")
    main_parser.add_argument("--dataset_type", type=str, default="nerfdata", help="nerfdata | nerfppdata")
    main_parser.add_argument("--dataset_name", type=str, help="")
    main_parser.add_argument("--max_step", type=int, default=1000000, help="maximum step to train")
    main_parser.add_argument("--val_step", type=int, default=50000, help="validation every n step")
    main_parser.add_argument("--rays_per_batch", type=int, default=160000, help="")
    main_parser.add_argument("--normalize_coordinates", action="store_true", help="")
    main_parser.add_argument('--no_batching', action="store_true", help='')

    main_parser.add_argument("--learning_rate", "-l", type=float, default=1e-4, help="learning_rate")
    main_parser.add_argument("--lrate_decay", type=int, default=250, help="learning_rate decay step (x 1000)")
    main_parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    main_parser.add_argument('--gpus', '-g', type=int, nargs='+', help='gpus to use')
    main_parser.add_argument('--skip_validation_at_start', action="store_true", help='')

main_parser_def(main_parser)
args, _ = main_parser.parse_known_args()

tf.random.set_seed(int(args.random_seed))

if args.model_type == "nerf":
    model_fn = NeRF
elif args.model_type == "nerf-degen":
    model_fn = NeRFDegenerate
elif args.model_type == "nerfpp":
    model_fn = NeRFPP
else:
    raise ValueError("Value '%s' is not suitable for model_type." % args.model_type)

opt_parser = model_fn.get_argparse()
main_parser = ArgumentParser(parents=[opt_parser])
main_parser_def(main_parser)
args = main_parser.parse_args()

if args.gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in args.gpus])
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for d in physical_devices:
            tf.config.experimental.set_memory_growth(d, True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

if args.gpus and len(args.gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
    # strategy = tf.distribute.experimental.CentralStorageStrategy()
else:
    strategy = None

steps_per_epoch = args.val_step
max_epochs = int(ceil(args.max_step / steps_per_epoch))
max_steps = steps_per_epoch * max_epochs

if strategy is not None:
    with strategy.scope():
        nerf = model_fn(args)
else:
    nerf = model_fn(args)

batch_size=args.rays_per_batch
if args.dataset_type == "nerfdata":
    dataset_name = 'lego'
    if args.dataset_name:
        dataset_name = args.dataset_name
    trainset = NerfSynthetic('nerfdata/nerf_synthetic', 'lego', 'train', args.use_white_bkgd, shuffle=True,
                             normalize_coordinates=args.normalize_coordinates)
elif args.dataset_type == "nerfppdata":
    dataset_name = 'africa'
    if args.dataset_name:
        dataset_name = args.dataset_name
        if dataset_name == 'M60':
            dataset_name = 'tat_intermediate_M60'
        elif dataset_name == 'playground':
            dataset_name = 'tat_intermediate_Playground'
        elif dataset_name == 'train':
            dataset_name = 'tat_intermediate_Train'
        elif dataset_name == 'truck':
            dataset_name = 'tat_training_Truck'

    trainset = NerfPPData('nerfppdata', dataset_name, 'train', shuffle=True,
                          normalize_coordinates=args.normalize_coordinates)
else:
    raise ValueError("%s is not defined" %args.dataset_type)

if args.model_type == "nerf" or args.model_type == "nerf-degen":
    if args.dataset_type == "nerfppdata":
        trainset.compute_inside()
        if dataset_name.startswith("scan"):
            nerf.far = trainset.radius * 4.0
        else:
            nerf.far = trainset.radius * 8.0
    else:
        nerf.far = trainset.far
    nerf.near = trainset.near

    if args.normalize_coordinates:
        nerf.near /= trainset.radius
        nerf.far /= trainset.radius
elif args.model_type == "nerfpp":
    trainset.compute_inside()
    nerf.near = trainset.near
    nerf.center = trainset.center
    nerf.radius = trainset.radius
    if args.normalize_coordinates:
        nerf.near /= trainset.radius
        nerf.center *= 0.
        nerf.radius /= trainset.radius

if args.no_batching:
    trainset = trainset.get_ray_unbatch_dataset_per_image(batch_size)
    trainset = trainset.repeat(-1).prefetch(10)
else:
    trainset = trainset.ray_unbatch_dataset
    trainset = trainset.batch(batch_size).repeat(-1).prefetch(10)

if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
else:
    lrate = args.learning_rate

optimizer = tf.optimizers.Adam(
    learning_rate=lrate,
    epsilon=1e-7,
)

os.makedirs(args.ckpt_path, exist_ok=True)
ckpt = tf.train.Checkpoint(model=nerf)
ckpt_man = tf.train.CheckpointManager(ckpt, args.ckpt_path, max_to_keep=None)


loss = tf.keras.losses.MeanSquaredError()
mse = tf.keras.metrics.MeanSquaredError()
rmse = tf.keras.metrics.RootMeanSquaredError()
psnr = util.metric.PSNR()
ssim = util.metric.SSIM()
metrics=[mse, psnr]


def compile_nerf(restore: bool):
    if strategy is not None:
        with strategy.scope():
            nerf.compile(loss=loss, optimizer=optimizer, metrics=metrics)
            if restore:
                ckpt_man.restore_or_initialize()
    else:
        nerf.compile(loss=loss, optimizer=optimizer, metrics=metrics)
        if restore:
            ckpt_man.restore_or_initialize()


initial_epoch = 0
# restore from chackpoint
if ckpt_man.latest_checkpoint is not None:
    compile_nerf(restore=True)
    initial_epoch = nerf.optimizer.iterations.numpy() // steps_per_epoch
else:
    compile_nerf(restore=False)

nerf.build([(None,None,None,3), (None,None,None,3)])

callback_tb = tf.keras.callbacks.TensorBoard(log_dir=args.ckpt_path, update_freq=10)
callback_ckpt = CheckpointManagerCallback(ckpt_man)

if args.dataset_type == "nerfdata":
    callback_vis = ValidateOnLego(callback_tb, skip_on_start=args.skip_validation_at_start,
                                  normalize_coordinates=args.normalize_coordinates, smallset=True)
elif args.dataset_type == "nerfppdata":
    dataset_name = args.dataset_name if args.dataset_name else 'africa'
    if dataset_name == 'africa':
        callback_vis = ValidateOnAfrica(callback_tb, skip_on_start=args.skip_validation_at_start,
                                    normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name == 'basket':
        callback_vis = ValidateOnBasket(callback_tb, skip_on_start=args.skip_validation_at_start,
                                        normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name == 'ship':
        callback_vis = ValidateOnShip(callback_tb, skip_on_start=args.skip_validation_at_start,
                                        normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name == 'statue':
        callback_vis = ValidateOnStatue(callback_tb, skip_on_start=args.skip_validation_at_start,
                                        normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name == 'torch':
        callback_vis = ValidateOnTorch(callback_tb, skip_on_start=args.skip_validation_at_start,
                                        normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name == 'M60':
        callback_vis = ValidateOnM60(callback_tb, skip_on_start=args.skip_validation_at_start,
                                        normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name == 'playground':
        callback_vis = ValidateOnPlayground(callback_tb, skip_on_start=args.skip_validation_at_start,
                                        normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name == 'train':
        callback_vis = ValidateOnTrain(callback_tb, skip_on_start=args.skip_validation_at_start,
                                        normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name == 'truck':
        callback_vis = ValidateOnTruck(callback_tb, skip_on_start=args.skip_validation_at_start,
                                        normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name.startswith('scan65'):
        callback_vis = ValidateOnScan65(callback_tb, skip_on_start=args.skip_validation_at_start,
                                       normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name.startswith('scan106'):
        callback_vis = ValidateOnScan106(callback_tb, skip_on_start=args.skip_validation_at_start,
                                       normalize_coordinates=args.normalize_coordinates, smallset=True)
    elif dataset_name.startswith('scan118'):
        callback_vis = ValidateOnScan118(callback_tb, skip_on_start=args.skip_validation_at_start,
                                       normalize_coordinates=args.normalize_coordinates, smallset=True)
    else:
        raise ValueError
else:
    raise ValueError("%s is not defined" % args.dataset_type)

callbacks = [callback_tb, callback_ckpt, callback_vis]

# train
nerf.fit(x=trainset, epochs=max_epochs, steps_per_epoch=steps_per_epoch, initial_epoch=initial_epoch, callbacks=callbacks)
