# baseline nerf
python main.py ckpt/nerf_lego_importance --use_viewdirs --n_samples 64 --n_importance 128 --use_perturb --lrate_decay 500 --use_white_bkgd \
--rays_per_batch 2048 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 1000000 --no_batching \
--learning_rate 5e-4

# nerf degenerate
python main.py ckpt/nerf_lego_degenerate --model_type nerf-degen --use_viewdirs --n_samples 256 --lrate_decay 500 \
--rays_per_batch 65536 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 1000000 --no_batching --use_white_bkgd \
--learning_rate 5e-4

# nerfpp africa
python main.py ckpt/nerfpp_africa --model_type nerfpp --dataset_type nerfppdata --use_viewdirs --n_samples 64 --n_importance 128 \
--rays_per_batch 2048 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 250000 --no_batching --use_perturb \
--learning_rate 5e-4 --lrate_decay 50000 --normalize_coordinates

# nerfpp basket
python main.py ckpt/nerfpp_basket --model_type nerfpp --dataset_type nerfppdata --dataset_name basket --use_viewdirs --n_samples 64 --n_importance 128 \
--rays_per_batch 2048 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 1000000 --no_batching --use_perturb \
--learning_rate 5e-4 --lrate_decay 50000 --normalize_coordinates

# nerfpp ship
python main.py ckpt/nerfpp_ship --model_type nerfpp --dataset_type nerfppdata --dataset_name ship --use_viewdirs --n_samples 64 --n_importance 128 \
--rays_per_batch 2048 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 1000000 --no_batching --use_perturb \
--learning_rate 5e-4 --lrate_decay 50000 --normalize_coordinates

# nerfpp torch
python main.py ckpt/nerfpp_torch --model_type nerfpp --dataset_type nerfppdata --dataset_name torch --use_viewdirs --n_samples 64 --n_importance 128 \
--rays_per_batch 2048 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 1000000 --no_batching --use_perturb \
--learning_rate 5e-4 --lrate_decay 50000 --normalize_coordinates

# nerfpp M60
python main.py ckpt/nerfpp_M60 --model_type nerfpp --dataset_type nerfppdata --dataset_name M60 --use_viewdirs --n_samples 64 --n_importance 128 \
--rays_per_batch 2048 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 1000000 --no_batching --use_perturb \
--learning_rate 5e-4 --lrate_decay 50000 --normalize_coordinates

# nerfpp playground
python main.py ckpt/nerfpp_playground --model_type nerfpp --dataset_type nerfppdata --dataset_name playground --use_viewdirs --n_samples 64 --n_importance 128 \
--rays_per_batch 2048 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 1000000 --no_batching --use_perturb \
--learning_rate 5e-4 --lrate_decay 50000 --normalize_coordinates

# nerfpp train
python main.py ckpt/nerfpp_train_nolratedecay  --model_type nerfpp --dataset_type nerfppdata --dataset_name train --use_viewdirs --n_samples 64 --n_importance 128 \
--rays_per_batch 2048 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 1000000 --no_batching --use_perturb \
--learning_rate 5e-4 --lrate_decay 50000 --normalize_coordinates

# nerfpp truck
python main.py ckpt/nerfpp_truck_nolratedecay  --model_type nerfpp --dataset_type nerfppdata --dataset_name truck --use_viewdirs --n_samples 64 --n_importance 128 \
--rays_per_batch 2048 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 1000000 --no_batching --use_perturb \
--learning_rate 5e-4 --lrate_decay 50000 --normalize_coordinates

# nerf africa
python main.py ckpt/nerf_africa_nolratedecay --use_viewdirs --n_samples 128 --dataset_type nerfppdata --dataset_name africa --n_importance 256 --use_perturb --lrate_decay 50000 \
--rays_per_batch 1024 --gpus 0 --chunk_size 2048 --val_step 5000 --max_step 1000000 --no_batching \
--learning_rate 5e-4

# nerf basket
python main.py ckpt/nerf_basket_nolratedecay --use_viewdirs --n_samples 128 --dataset_type nerfppdata --dataset_name basket --n_importance 256 --use_perturb --lrate_decay 50000 \
--rays_per_batch 1024 --gpus 0 --chunk_size 2048 --val_step 5000 --max_step 1000000 --no_batching \
--learning_rate 5e-4

# nerf ship
python main.py ckpt/nerf_ship_nolratedecay --use_viewdirs --n_samples 128 --dataset_type nerfppdata --dataset_name ship --n_importance 256 --use_perturb --lrate_decay 50000 \
--rays_per_batch 1024 --gpus 0 --chunk_size 2048 --val_step 5000 --max_step 1000000 --no_batching \
--learning_rate 5e-4

# nerf torch
python main.py ckpt/nerf_torch_nolratedecay --use_viewdirs --n_samples 128 --dataset_type nerfppdata --dataset_name torch --n_importance 256 --use_perturb --lrate_decay 50000 \
--rays_per_batch 1024 --gpus 0 --chunk_size 2048 --val_step 5000 --max_step 1000000 --no_batching \
--learning_rate 5e-4

# nerf M60
python main.py ckpt/nerf_M60_nolratedecay --use_viewdirs --n_samples 128 --dataset_type nerfppdata --dataset_name M60 --n_importance 256 --use_perturb --lrate_decay 50000 \
--rays_per_batch 1024 --gpus 0 --chunk_size 2048 --val_step 5000 --max_step 1000000 --no_batching \
--learning_rate 5e-4

# nerf playground
python main.py ckpt/nerf_playground_nolratedecay --use_viewdirs --n_samples 128 --dataset_type nerfppdata --dataset_name playground --n_importance 256 --use_perturb --lrate_decay 50000 \
--rays_per_batch 1024 --gpus 0 --chunk_size 2048 --val_step 5000 --max_step 250001 --no_batching \
--learning_rate 5e-4

# nerf train
python main.py ckpt/nerf_train_nolratedecay --use_viewdirs --n_samples 128 --dataset_type nerfppdata --dataset_name train --n_importance 256 --use_perturb --lrate_decay 50000 \
--rays_per_batch 1024 --gpus 0 --chunk_size 2048 --val_step 5000 --max_step 1000000 --no_batching \
--learning_rate 5e-4

# nerf truck
python main.py ckpt/nerf_truck_nolratedecay --use_viewdirs --n_samples 128 --dataset_type nerfppdata --dataset_name truck --n_importance 256 --use_perturb --lrate_decay 50000 \
--rays_per_batch 1024 --gpus 0 --chunk_size 4096 --val_step 5000 --max_step 1000000 --no_batching \
--learning_rate 5e-4