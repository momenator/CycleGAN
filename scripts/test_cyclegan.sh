set -ex

python test.py --direction AtoB  --dataroot ./datasets/visceral_full --name ct_mr_cyclegan_patch --model cycle_gan --gpu_ids 5 --dataset_mode unaligned --input_nc 1 --output_nc 1 --phase test --no_dropout

# python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout
