set -ex

python train.py --dataroot ./datasets/visceral_spine --lambda_identity 0 --name ct_mr_visceral_spine_2 --niter 100 --niter_decay 10 --model cycle_gan --gpu_ids 2 --preprocess none --dataset_mode unaligned --input_nc 1 --output_nc 1  --max_dataset_size 7500

# python train.py --dataroot ./datasets/ct_mr_nrad --max_dataset_size 4000 --lambda_identity 0 --name ct_mr_nrad_2 --niter 100 --niter_decay 10 --model cycle_gan --gpu_ids 0 --preprocess none --dataset_mode unaligned --input_nc 1 --output_nc 1 

# python train.py --dataroot ./datasets/apple2orange --gpu_ids 6 --name a2o --model cycle_gan --pool_size 50

# python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout
