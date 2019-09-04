"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
def prepare_fake_scans(model, data):
avez('./res_test.npz',ef prepare_fake_scans(model, data):

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.scan import reconstruct_scan
import numpy as np


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        fake_B = reconstruct_scan(data['A'], model, (256, 256), 'AtoB')
        fake_A = reconstruct_scan(data['B'], model, (256, 256), 'BtoA')
        np.savez('./results/ct_mr_cyclegan_patch/fake_scans/fake_MR_{}.npz'.format(data['A_paths'][0]), data=fake_B)
        np.savez('./results/ct_mr_cyclegan_patch/fake_scans/fake_CT_{}.npz'.format(data['B_paths'][0]), data=fake_A)
        # model.set_input(data)  # unpack data from data loader
        # model.test()           # run inference
        # visuals = model.get_current_visuals()  # get image results
        # img_path = model.get_image_paths()     # get image paths
        # if i % 5 == 0:  # save images to an HTML file
        #    print('processing (%04d)-th image... %s' % (i, img_path))
        # real_A = visuals['real_A'].cpu().numpy()
        # real_B = visuals['real_B'].cpu().numpy()
        # fake_A = visuals['fake_A'].cpu().numpy()
        # fake_B = visuals['fake_B'].cpu().numpy()
        # rec_A = visuals['rec_A'].cpu().numpy()
        # rec_B = visuals['rec_B'].cpu().numpy() 
        # print(visuals)
        # np.savez('./results/ct_mr_cyclegan_patch/test_latest/test_{}.npz'.format(i), real_A=real_A, real_B=real_B, fake_A=fake_A, fake_B=fake_B, rec_A=rec_A, rec_B=rec_B)
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()  # save the HTML
