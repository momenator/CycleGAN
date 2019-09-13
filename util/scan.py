import numpy as np
from scipy.ndimage import zoom
from skimage.util import view_as_windows
from itertools import product
from typing import Tuple
import torch
import torch.nn as nn

# Patch and Unpatchify
def patchify(patches: np.ndarray, patch_size: Tuple[int, int], step: int = 1):
    return view_as_windows(patches, patch_size, step)


def unpatchify(patches: np.ndarray, imsize: Tuple[int, int]):

    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    divisor = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, p_h, p_w = patches.shape

    # Calculat the overlap size in each axis
    o_w = (n_w * p_w - i_w) / (n_w - 1)
    o_h = (n_h * p_h - i_h) / (n_h - 1)

    # The overlap should be integer, otherwise the patches are unable to reconstruct into a image with given shape
    assert int(o_w) == o_w
    assert int(o_h) == o_h

    o_w = int(o_w)
    o_h = int(o_h)

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i,j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    return image / divisor


# resize image
def resize_img(img, size=256):
    x, y = img.shape
    return zoom(img, (size/x, size/y))


def get_all_patches(volume, side='c', dim=256):
    """
        side = either 'c', 'a', 's'
        a - axial
        c - coronal
        s - sagittal
    """
    a, c, s = volume.shape
    all_patches = []
    
    if side == 'a':
        count = a
    elif side == 'c':
        count = c
    else:
        count = s
    
    for i in range(count):
        if side == 'a':
            scan_slice = volume[i,:,:]
        elif side == 'c':
            scan_slice = volume[:,i,:]
        else:
            scan_slice = volume[:,:,i]
        patches = get_patches_from_2d_img(scan_slice)
        all_patches.append(patches)
    
    all_patches = np.array(all_patches).reshape(-1, dim, dim)
    
    return all_patches


def extract_patches_2d(img,patch_shape,step=[1.0,1.0],batch_first=False):
    patch_H, patch_W = patch_shape[0], patch_shape[1]
    if(img.size(2)<patch_H):
        num_padded_H_Top = (patch_H - img.size(2))//2
        num_padded_H_Bottom = patch_H - img.size(2) - num_padded_H_Top
        padding_H = nn.ConstantPad2d((0,0,num_padded_H_Top,num_padded_H_Bottom),0)
        img = padding_H(img)
    if(img.size(3)<patch_W):
        num_padded_W_Left = (patch_W - img.size(3))//2
        num_padded_W_Right = patch_W - img.size(3) - num_padded_W_Left
        padding_W = nn.ConstantPad2d((num_padded_W_Left,num_padded_W_Right,0,0),0)
        img = padding_W(img)
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    patches_fold_H = img.unfold(2, patch_H, step_int[0])
    if((img.size(2) - patch_H) % step_int[0] != 0):
        patches_fold_H = torch.cat((patches_fold_H,img[:,:,-patch_H:,].permute(0,1,3,2).unsqueeze(2)),dim=2)
    patches_fold_HW = patches_fold_H.unfold(3, patch_W, step_int[1])   
    if((img.size(3) - patch_W) % step_int[1] != 0):
        patches_fold_HW = torch.cat((patches_fold_HW,patches_fold_H[:,:,:,-patch_W:,:].permute(0,1,2,4,3).unsqueeze(3)),dim=3)
    patches = patches_fold_HW.permute(2,3,0,1,4,5)
    patches = patches.reshape(-1,img.size(0),img.size(1),patch_H,patch_W)
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    return patches


def reconstruct_from_patches_2d(patches,img_shape,step=[1.0,1.0],batch_first=False):
    if(batch_first):
        patches = patches.permute(1,0,2,3,4)
    patch_H, patch_W = patches.size(3), patches.size(4)
    img_size = (patches.size(1), patches.size(2),max(img_shape[0], patch_H), max(img_shape[1], patch_W))
    step_int = [0,0]
    step_int[0] = int(patch_H*step[0]) if(isinstance(step[0], float)) else step[0]
    step_int[1] = int(patch_W*step[1]) if(isinstance(step[1], float)) else step[1]
    nrow, ncol = 1 + (img_size[-2] - patch_H)//step_int[0], 1 + (img_size[-1] - patch_W)//step_int[1]
    r_nrow = nrow + 1 if((img_size[2] - patch_H) % step_int[0] != 0) else nrow
    r_ncol = ncol + 1 if((img_size[3] - patch_W) % step_int[1] != 0) else ncol
    patches = patches.reshape(r_nrow,r_ncol,img_size[0],img_size[1],patch_H,patch_W)
    img = torch.zeros(img_size, device = patches.device)
    overlap_counter = torch.zeros(img_size, device = patches.device)
    for i in range(nrow):
        for j in range(ncol):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += patches[i,j,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0):
        for j in range(ncol):
            img[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += patches[-1,j,]
            overlap_counter[:,:,-patch_H:,j*step_int[1]:j*step_int[1]+patch_W] += 1
    if((img_size[3] - patch_W) % step_int[1] != 0):
        for i in range(nrow):
            img[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += patches[i,-1,]
            overlap_counter[:,:,i*step_int[0]:i*step_int[0]+patch_H,-patch_W:] += 1
    if((img_size[2] - patch_H) % step_int[0] != 0 and (img_size[3] - patch_W) % step_int[1] != 0):
        img[:,:,-patch_H:,-patch_W:] += patches[-1,-1,]
        overlap_counter[:,:,-patch_H:,-patch_W:] += 1
    img /= overlap_counter
    if(img_shape[0]<patch_H):
        num_padded_H_Top = (patch_H - img_shape[0])//2
        num_padded_H_Bottom = patch_H - img_shape[0] - num_padded_H_Top
        img = img[:,:,num_padded_H_Top:-num_padded_H_Bottom,]
    if(img_shape[1]<patch_W):
        num_padded_W_Left = (patch_W - img_shape[1])//2
        num_padded_W_Right = patch_W - img_shape[1] - num_padded_W_Left
        img = img[:,:,:,num_padded_W_Left:-num_padded_W_Right]
    return img


def save_slice(visuals):
    real_A = visuals['real_A'].cpu().numpy()
    real_B = visuals['real_B'].cpu().numpy()
    fake_A = visuals['fake_A'].cpu().numpy()
    fake_B = visuals['fake_B'].cpu().numpy()
    rec_A = visuals['rec_A'].cpu().numpy()
    rec_B = visuals['rec_B'].cpu().numpy() 
    # print(visuals)
    np.savez('./test_slice.npz', real_A=real_A, real_B=real_B, fake_A=fake_A, fake_B=fake_B, rec_A=rec_A, rec_B=rec_B)


def reconstruct_scan(scan, model, im_size, direction):
    """
        Only supports coronal scans for now!
        Returns a fake scan given a real one in numpy array
    """
    
    _, _, x, y, z = scan.size()
    print("New scan. Size: ", scan.size())
    scan = scan.view(x, y, z)
    rec_scan = None
    step = 200   
     
    for i in range(y):
        sl = scan[:, i, :]
        
        # patchify - only step=1 works
        patches = extract_patches_2d(sl.view(1, 1, x, z), im_size, (step, step))
        # patches = patchify(sl, im_size, step=1)
        rec_patches = []

        print("Processing slice ", i, " out of ", y)

        for i, patch in enumerate(patches):
            # set up data here
            # patch = torch.from_numpy(patch).view(1, 1, im_size[0], im_size[1])
            model.set_input({ 'A': patch, 'B': patch, 'A_paths': [], 'B_paths': [] })  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            # if i % 15 == 0:
            #    print("Processing patch ", i, " out of ", patches.shape[0])            
                # save_slice(visuals)
            if direction == 'AtoB':
                rec_patches.append(visuals['fake_B'])
            else:
                rec_patches.append(visuals['fake_A'])
        
        rec_patches = torch.stack(rec_patches)
        rec_slice = reconstruct_from_patches_2d(rec_patches, (x, z), (step, step))
        # rec_slice_buff = rec_slice.cpu().numpy().reshape(x, z)
        # np.savez('./test_rec_slice.npz', data=rec_slice_buff)
        # rec_patches = unpatchify(np.array(rec_patches).reshape(original_shape), im_size)
        rec_slice = rec_slice.cpu().numpy().reshape(x, 1, z)
        if rec_scan is None:
            rec_scan = rec_slice
        else:
            rec_scan = np.concatenate((rec_scan, rec_slice), axis=1)
        # np.savez('./test_rec_slice.npz', data=rec_slice)
        # print(rec_scan.shape)    
    rec_scan = np.array(rec_scan)
    return rec_scan


