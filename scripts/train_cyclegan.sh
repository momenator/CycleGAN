set -ex

python train.py --dataroot ./datasets/visceral_full --name ct_mr_cyclegan_patch --model cycle_gan --gpu_ids 6,3,0,5 --preprocess none --dataset_mode unaligned --input_nc 1 --output_nc 1 --max_dataset_size 55000 --num_threads 24 --batch_size 24

# python train.py --dataroot ./datasets/visceral_full --name ct_mr_cyclegan_patch2_small --model cycle_gan --gpu_ids 6 --preprocess none --dataset_mode unaligned --input_nc 1 --output_nc 1 --max_dataset_size 3500 --num_threads 24 --batch_size 24 

# python train.py --dataroot ./datasets/apple2orange --gpu_ids 6 --name a2o --model cycle_gan --pool_size 50

# python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --pool_size 50 --no_dropout
