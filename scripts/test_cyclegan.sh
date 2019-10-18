set -ex

# python test.py --direction AtoB  --dataroot ./datasets/visceral_full --name ct_mr_cyclegan_patch_no_identity --model cycle_gan --gpu_ids 6 --dataset_mode unaligned --input_nc 1 --output_nc 1 --phase test --no_dropout

python test.py --direction AtoB --dataroot ./datasets/visceral_spine --preprocess none --name ct_mr_visceral_spine_5 --model cycle_gan --gpu_ids 2 --dataset_mode unaligned --input_nc 1 --output_nc 1 --phase test --no_dropout

# python test.py --direction AtoB --epoch 30 --dataroot ./datasets/visceral_spine --preprocess none --name ct_mr_visceral_spine_2 --model cycle_gan --gpu_ids 7 --dataset_mode unaligned --input_nc 1 --output_nc 1 --phase test --no_dropout

# python test.py --direction AtoB  --dataroot ./datasets/ct_mr_nrad --name ct_mr_nrad_2 --model cycle_gan --gpu_ids 0 --dataset_mode unaligned --input_nc 1 --output_nc 1 --phase test --no_dropout

# python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan --phase test --no_dropout
