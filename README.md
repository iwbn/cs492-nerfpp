# NeRF++ TensorFlow Repository

This code repository aims to build NeRF++ with TensorFlow.

Original Repository: https://github.com/Kai-46/nerfplusplus

## How to train

Training example (please refer to `training_examples.sh`):
```sh
python main.py ckpt/nerf_africa_nolratedecay --use_viewdirs --n_samples 128 --dataset_type nerfppdata --dataset_name africa --n_importance 256 --use_perturb --lrate_decay 50000 \
--rays_per_batch 1024 --gpus 0 --chunk_size 2048 --val_step 5000 --max_step 1000000 --no_batching \
--learning_rate 5e-4
```

## How to evaluate

evaluate example (use the same network settings as training):
```sh
python evaluate.py ckpt/nerf_africa_nolratedecay/ckpt-250000 \
--model_type nerfpp --dataset_type nerfppdata --dataset_name africa \
--use_viewdirs --n_samples 64 --n_importance 128 \
--chunk_size 4096 \
--normalize_coordinates --gpus 0
```

## Datasets
We provide links to download:
[NeRF Synthetic](https://github.com/bmild/nerf#running-code), 
[DTU](https://roboimagedata.compute.dtu.dk/?page_id=36), 
[LF](https://github.com/Kai-46/nerfplusplus#data), 
[Tanks and Temples (T&T)](https://github.com/isl-org/FreeViewSynthesis#data).

### Dataset Directories
Please place each dataset as below:

* [WORKING DIRECTORY]
  * nerfdata
    * nerf_synthetic
      * lego
      * chair
      * ...
  * nerfppdata
    * africa
    * basket
    * scan65
    * scan106
    * scan118
    * ship
    * statue
    * tat_intermediate_M60
    * tat_intermediate_Playground
    * tat_intermediate_Train
    * tat_training_Truck
    * torch

