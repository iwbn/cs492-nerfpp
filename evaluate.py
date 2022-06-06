import tensorflow as tf
import cv2

from util import lpips_tfv2
from model.nerf import NeRF
from model.nerfpp import NeRFPP

from data.nerf_synthetic import NerfSynthetic
from data.nerf_pp import NerfPPData
from model.nerf_degenerate import NeRFDegenerate
import os
from argparse import ArgumentParser
import util.metric
from time import time

main_parser = ArgumentParser()
def main_parser_def(main_parser):
    main_parser.add_argument("ckpt_path", type=str, help="log and ckpts are saved")
    main_parser.add_argument("--model_type", type=str, default="nerf", help="nerf | nerfpp")
    main_parser.add_argument("--dataset_name", type=str, help="")
    main_parser.add_argument("--dataset_type", type=str, default="nerfdata", help="nerfdata | nerfppdata")
    main_parser.add_argument("--normalize_coordinates", action="store_true", help="")
    main_parser.add_argument("--use_training_split", action="store_true", help="")
    main_parser.add_argument("--no_save_images", action="store_true", help="")

    main_parser.add_argument('--gpus', '-g', type=int, nargs='+', help='gpus to use')

main_parser_def(main_parser)
args, _ = main_parser.parse_known_args()


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

if strategy is not None:
    with strategy.scope():
        nerf = model_fn(args)
else:
    nerf = model_fn(args)

if args.dataset_type == "nerfdata":
    dataset_name = 'lego'
    if args.dataset_name:
        dataset_name = args.dataset_name
    trainset = NerfSynthetic('nerfdata/nerf_synthetic', 'lego', 'train', args.use_white_bkgd, shuffle=True,
                             normalize_coordinates=args.normalize_coordinates)
    if args.use_training_split:
        testset = NerfSynthetic('nerfdata/nerf_synthetic', 'lego', 'train', args.use_white_bkgd, shuffle=False,
                                normalize_coordinates=args.normalize_coordinates)
    else:
        testset = NerfSynthetic('nerfdata/nerf_synthetic', 'lego', 'test', args.use_white_bkgd, shuffle=False,
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
    if args.use_training_split:
        testset = NerfPPData('nerfppdata', dataset_name, 'train', shuffle=False,
                             normalize_coordinates=args.normalize_coordinates)
    else:
        testset = NerfPPData('nerfppdata', dataset_name, 'test', shuffle=False,
                             normalize_coordinates=args.normalize_coordinates)
else:
    raise ValueError("%s is not defined" %args.dataset_type)


if args.model_type == "nerf" or args.model_type == "nerf-degen":
    if args.dataset_type == "nerfppdata":
        trainset.compute_inside()
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
        testset.radius = trainset.radius
        testset.center = trainset.center

ckpt_dir = args.ckpt_path
if not os.path.isdir(ckpt_dir):
    ckpt_dir = os.path.split(ckpt_dir)[0]
ckpt = tf.train.Checkpoint(model=nerf)
ckpt_man = tf.train.CheckpointManager(ckpt, ckpt_dir, max_to_keep=None)


mse = tf.keras.metrics.MeanSquaredError()
rmse = tf.keras.metrics.RootMeanSquaredError()
psnr = util.metric.PSNR()
ssim = util.metric.SSIM()
metrics=[mse, psnr]


initial_epoch = 0
# restore from chackpoint
if ckpt_man.latest_checkpoint is not None or not os.path.isdir(args.ckpt_path):
    if not os.path.isdir(args.ckpt_path):
        status = ckpt.restore(args.ckpt_path, )
        ckpt_path = args.ckpt_path
        print("%s is restored" % ckpt_path)
    else:
        ckpt_path = ckpt_man.latest_checkpoint
        status = ckpt.restore(ckpt_man.latest_checkpoint, )
        print("%s is restored" % ckpt_man.latest_checkpoint)

else:
    raise ValueError("checkpoint failed")

nerf.build([(None,None,None,3), (None,None,None,3)])

metrics = [tf.keras.metrics.RootMeanSquaredError(), util.metric.PSNR(), util.metric.SSIM(),
           util.metric.MS_SSIM()]#, util.metric.LPIPS(model='net-lin',net='vgg')]
nerf_call = tf.function(nerf.call)


imsave_path = ckpt_path + "-results"
if args.use_training_split:
    imsave_path = imsave_path + "-train"

if not args.no_save_images:
    os.makedirs(imsave_path, exist_ok=True)

dataset = testset.ray_dataset
import numpy as np


psnr = util.metric.PSNR()

num_samples = dataset.cardinality()
for i, (x, y) in enumerate(dataset):

    st = time()
    pred = nerf_call(x)
    et = time()

    rgb = tf.clip_by_value(pred['rgb'], 0., 1.)

    psnr.update_state(y, rgb)
    psnr_val = psnr.result().numpy()
    psnr.reset_state()
    #cv2.imshow("im", np.uint8(rgb.numpy() * 255.)[..., [2, 1, 0]])
    #cv2.waitKey(1)
    if not args.no_save_images:
        cv2.imwrite(os.path.join(imsave_path, "%03d_gt.png" % i), np.uint8(y.numpy() * 255.)[..., [2, 1, 0]])
        cv2.imwrite(os.path.join(imsave_path,"%03d_psnr_%.8f.png" % (i, psnr_val)), np.uint8(rgb.numpy() * 255.)[...,[2,1,0]])


    for metric in metrics:
        metric.update_state(y, rgb)

    print("[%d/%d] (%.2f sec)" % (i + 1, num_samples, et - st))
    for metric in metrics:
        print("%s: %s" % (metric.name, metric.result().numpy()))

for metric in metrics:
    print("%s: %s"%(metric.name, metric.result().numpy()))
    metric.reset_state()