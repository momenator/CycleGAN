import numpy as np
from scipy.ndimage import zoom
from skimage.util import view_as_windows
from itertools import product
from typing import Tuple
import torch
import torch.nn as nn
import SimpleITK as sitk


# resize image
def resize_img(img, size=256):
    x, y = img.shape
    return zoom(img, (size/x, size/y))


def extract_patches_2d(img, patch_shape, step=[1.0,1.0], batch_first=False):
    
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


def reconstruct_from_patches_2d(patches, img_shape, step=[1.0,1.0], batch_first=False):
    
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


def save_slice(visuals, target_path):
    data = {}
    data['real_A'] = visuals['real_A'].cpu().numpy()
    data['real_B'] = visuals['real_B'].cpu().numpy()
    data['fake_A'] = visuals['fake_A'].cpu().numpy()
    data['fake_B'] = visuals['fake_B'].cpu().numpy()
    data['rec_A'] = visuals['rec_A'].cpu().numpy()
    data['rec_B'] = visuals['rec_B'].cpu().numpy() 
    np.savez('{}.npz'.format(target_path), data=data)


def get_fake_and_rec_scans(scan, model, patch_size, direction='AtoB', side = 'c', step=(64, 64)):
    """
        Returns a fake scan given a real one in numpy array
    """
    
    _, _, x, y, z = scan.size()
    
    print("Processing New scan. Size: ", scan.size())
    
    scan = scan.view(x, y, z)
    rec_scan = None
    fake_scan = None
    slices = y
    slice_dim = (x, z)
    
    if side == 's':
        slices = x
        slice_dim = (y, z)

    elif side == 'a':
        slices = z
        slice_dim = (x, y)

     
    for i in range(slices):
        sl = scan[:, i, :]
        patches = extract_patches_2d(sl.view(1, 1, x, z), patch_size, step)

        if side == 's':
            sl = scan[i, :, :]
            patches = extract_patches_2d(sl.view(1, 1, y, z), patch_size, step)

        elif side == 'a':
            sl = scan[:, :, i]
            patches = extract_patches_2d(sl.view(1, 1, x, y), patch_size, step)
        
        fake_patches = []
        rec_patches = []

        print("Processing slice ", i, " out of ", slices)

        for i, patch in enumerate(patches):
            # set up data here
            model.set_input({ 'A': patch, 'B': patch, 'A_paths': [], 'B_paths': [] })  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results

            if direction == 'AtoB':
                fake_patches.append(visuals['fake_B'])
                rec_patches.append(visuals['rec_A'])
            else:
                fake_patches.append(visuals['fake_A'])
                rec_patches.append(visuals['rec_B'])
        
        fake_patches = torch.stack(fake_patches)
        fake_slice = reconstruct_from_patches_2d(fake_patches, slice_dim, step)

        rec_patches = torch.stack(rec_patches)
        rec_slice = reconstruct_from_patches_2d(rec_patches, slice_dim, step)

        join_axis = 1
        if side == 's':
            fake_slice = rec_slice.cpu().numpy().reshape(1, y, z)
            rec_slice = rec_slice.cpu().numpy().reshape(1, y, z)
            join_axis = 0

        elif side == 'a':
            fake_slice = rec_slice.cpu().numpy().reshape(x, y, 1)
            rec_slice = rec_slice.cpu().numpy().reshape(x, y, 1)
            join_axis = 2

        else:
            fake_slice = rec_slice.cpu().numpy().reshape(x, 1, z)
            rec_slice = rec_slice.cpu().numpy().reshape(x, 1, z)

        if fake_scan is None:
            fake_scan = fake_slice
        else:
            fake_scan = np.concatenate((fake_scan, fake_slice), axis=join_axis)

        if rec_scan is None:
            rec_scan = rec_slice
        else:
            rec_scan = np.concatenate((rec_scan, rec_slice), axis=join_axis)

    fake_scan = np.array(fake_scan)
    rec_scan = np.array(rec_scan)

    return fake_scan, rec_scan


def save_fake_and_rec_scans(target_path, scan_name, fake_scan, rec_scan):
    
    # save fake scan as npz
    np.savez('{}{}.npz'.format(target_path, scan_name), data=fake_scan)

    # save reconstructed scan as npz
    np.savez('{}rec_{}.npz'.format(target_path, scan_name), data=rec_scan)

    # save scans as nii file
    fake_itk = sitk.GetImageFromArray(fake_scan)
    sitk.WriteImage(fake_itk, '{}fake_{}.nii.gz'.format(target_path, scan_name))

