import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import SimpleITK as sitk
import pandas as pd
import scipy.ndimage


def convert_tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor

def load_model(state_dict, model):
    # load state dict
    model.stems.load_state_dict(state_dict['stem_state_dict'])
    # if hypernet in model attribute
    if hasattr(model, 'hypernet'):
        model.hypernet.load_state_dict(state_dict['hypernet_state_dict'])
    return model


def load_model_from_dir(checkpoint_dir, model):
    # glob file with suffix pth
    from pathlib import Path as pa
    import re
    p = pa(checkpoint_dir)
    # check if p has subdir named model_wts
    if (p/'model_wts').exists():
        p = p/'model_wts'
    p = p.glob('*.pth')
    p = sorted(p, key=lambda x: [int(n) for n in re.findall(r'\d+', str(x))])
    model_path = str(p[-1])
    load_model(torch.load(model_path), model)
    return model_path


def find_surf(seg, kernel=3, thres=1):
    '''
    Find near-surface voxels of a segmentation.

    Args:
        seg: (**,D,H,W)
        radius: int

    Returns:
        surf: (**,D,H,W)
    '''
    if thres<=0:
        return torch.zeros_like(seg).bool()
    pads    = tuple((kernel-1)//2 for _ in range(6))
    seg_k   = F.pad(seg, pads, mode='constant', value=0).unfold(-3, kernel, 1).unfold(-3, kernel, 1).unfold(-3, kernel, 1)
    seg_num = seg_k.sum(dim=(-1,-2,-3))
    surf    = (seg_num<(kernel**3)*thres) & seg.bool()
    # how large a boundary we want to remove?
    # surf = (seg_num<(kernel**3//2)) & seg.bool()
    return surf


def show_img(res, save_path=None, norm=True, cmap=None, inter_dst=5) -> Image:
    import torchvision.transforms as T
    res = tt(res)
    if norm: res = normalize(res)
    if res.ndim>=3:
        return T.ToPILImage()(visualize_3d(res, cmap=cmap, inter_dst=inter_dst))
    # normalize res
    # res = (res-res.min())/(res.max()-res.min())

    pimg = T.ToPILImage()(res)
    if save_path:
        pimg.save(save_path)
    return pimg


def convert_nda_to_itk(nda: np.ndarray, itk_image: sitk.Image):
    """From a numpy array, get an itk image object, copying information
    from an existing one. It switches the z-axis from last to first position.

    Args:
        nda (np.ndarray): 3D image array
        itk_image (sitk.Image): Image object to copy info from

    Returns:
        new_itk_image (sitk.Image): New Image object
    """
    #import pdb; pdb.set_trace()
    #new_itk_image = sitk.GetImageFromArray(np.moveaxis(nda, -1, 0)) # Why doesn't it work for the abdominal dataset?
    nda           = np.moveaxis(nda, [0, 1, 2], [2, 1, 0])
    new_itk_image = sitk.GetImageFromArray(nda) # Abdomen?
    new_itk_image.SetOrigin(itk_image.GetOrigin())
    new_itk_image.SetSpacing(itk_image.GetSpacing())
    new_itk_image.CopyInformation(itk_image)
    return new_itk_image

import nibabel as nib
def convert_nflow_to_itk(deformation_field: np.ndarray):
    """From a numpy array, get an itk image object, copying information
    from an existing one. It switches the z-axis from last to first position.

    Args:
        nda (np.ndarray): 4D image array
        itk_image (sitk.Image): Image object to copy info from

    Returns:
        new_itk_image (sitk.Image): New Image object
    """
    #import pdb; pdb.set_trace()
    deformation_field = np.transpose(deformation_field, (1, 2, 3, 0))

    # Create a NIfTI image
    nifti_img = nib.Nifti1Image(deformation_field, affine=np.eye(4))
    return nifti_img

def convert_itk_to_nda(itk_image: sitk.Image):
    """From an itk Image object, get a numpy array. It moves the first z-axis
    to the last position (np.ndarray convention).

    Args:
        itk_image (sitk.Image): Image object to convert

    Returns:
        result (np.ndarray): Converted nda image
    """
    return np.moveaxis(sitk.GetArrayFromImage(itk_image), 0, -1)



def normalize_flow(deformation_field):
    min_val = deformation_field.min()
    max_val = deformation_field.max()
    normalized_field = 2 * (deformation_field - min_val) / (max_val - min_val) - 1
    return normalized_field


def normalize_keypoints(points, original_spacing, target_spacing):
    """
    Normalize points from original voxel spacing to target voxel spacing.
    
    Args:
        points (torch.Tensor): Tensor of points of shape (N, 3).
        original_spacing (torch.Tensor or list or tuple): Original voxel spacing.
        target_spacing (torch.Tensor or list or tuple): Target voxel spacing.
    
    Returns:
        torch.Tensor: Normalized points of shape (N, 3).
    """
    original_spacing = torch.tensor(original_spacing, dtype=points.dtype, device=points.device)
    target_spacing   = torch.tensor(target_spacing, dtype=points.dtype, device=points.device)
    scale_factors    = original_spacing / target_spacing
    return points * scale_factors



def resample_image_to_spacing(image, original_spacing, new_spacing):
    
    """
    Resample the image to a new voxel spacing.
    
    Args:
        image (numpy.ndarray): 3D array of image slices of shape (D, H, W).
        original_spacing (tuple): Original voxel spacing (sx, sy, sz).
        new_spacing (tuple): New voxel spacing (nx, ny, nz).
    
    Returns:
        resampled_image (numpy.ndarray): Resampled image.
    """
    resize_factors  = np.array(original_spacing) / np.array(new_spacing)
    resampled_image = scipy.ndimage.zoom(image, resize_factors, order=1)
    return resampled_image


def resample_flow_considering_img_to_spacing(flow, original_spacing, new_spacing, resampled_image_size):
    """
    Resample the deformation field to a new voxel spacing.
    
    Args:
        deformation_field (numpy.ndarray): 4D array of deformation field of shape (3, D, H, W).
        original_spacing (tuple): Original voxel spacing (sx, sy, sz).
        new_spacing (tuple): New voxel spacing (nx, ny, nz).
        resampled_image_shape (tuple): Shape of the resampled image (D, H, W).
    
    Returns:
        resampled_deformation_field (numpy.ndarray): Resampled deformation field.
    """
    resize_factors              = np.array(original_spacing) / np.array(new_spacing)
    resampled_deformation_field = np.zeros((3, *resampled_image_size))
    for i in range(3):
        resampled_deformation_field[i] = scipy.ndimage.zoom(flow[i], resize_factors, order=1)
        # Adjust deformation magnitudes according to the new spacing
        resampled_deformation_field[i] *= resize_factors[i]

    return resampled_deformation_field
 

def resample_flow_to_spacing(deformation_field, original_spacing, new_spacing):
    """
    Resample the deformation field to a new voxel spacing.
    
    Args:
        deformation_field (numpy.ndarray): 4D array of deformation field of shape (3, D, H, W).
        original_spacing (tuple): Original voxel spacing (sx, sy, sz).
        new_spacing (tuple): New voxel spacing (nx, ny, nz).
    
    Returns:
        resampled_deformation_field (numpy.ndarray): Resampled deformation field.
    """
    resize_factors = np.array(original_spacing) / np.array(new_spacing)
    original_shape = deformation_field.shape[1:]  # Get the original shape (D, H, W)
    new_shape      = np.round(np.array(original_shape) * resize_factors).astype(int)  # Calculate new shape
    resampled_deformation_field = np.zeros((3, *new_shape))
    for i in range(3):
        resampled_deformation_field[i] = scipy.ndimage.zoom(deformation_field[i], resize_factors, order=1)
        # Adjust deformation magnitudes according to the new spacing
        resampled_deformation_field[i] *= resize_factors[i]
    return resampled_deformation_field


def resample_to_spacing(image, deformation_field, original_spacing, new_spacing):
    """
    Resample the image and deformation field to a new voxel spacing.
    
    Args:
        image (numpy.ndarray): 3D array of image slices of shape (D, H, W).
        deformation_field (numpy.ndarray): Deformation field of shape (3, D, H, W).
        original_spacing (tuple): Original voxel spacing (sx, sy, sz).
        new_spacing (tuple): New voxel spacing (nx, ny, nz).
    
    Returns:
        resampled_image (numpy.ndarray): Resampled image.
        resampled_deformation_field (numpy.ndarray): Resampled deformation field.
    """
    resize_factors = np.array(original_spacing) / np.array(new_spacing)
    
    # Resample the image
    resampled_image = scipy.ndimage.zoom(image, resize_factors, order=1)
    
    # Resample each component of the deformation field separately
    resampled_deformation_field = np.zeros((3, *resampled_image.shape))
    for i in range(3):
        resampled_deformation_field[i] = scipy.ndimage.zoom(deformation_field[i], resize_factors, order=1)
        # Adjust deformation magnitudes according to the new spacing
        resampled_deformation_field[i] *= resize_factors[i]

    return resampled_image, resampled_deformation_field




def normalize_keypoints(points, original_spacing, target_spacing):
    """
    Normalize points from original voxel spacing to target voxel spacing.
    
    Args:
        points (torch.Tensor): Tensor of points of shape (N, 3).
        original_spacing (torch.Tensor or list or tuple): Original voxel spacing.
        target_spacing (torch.Tensor or list or tuple): Target voxel spacing.
    
    Returns:
        torch.Tensor: Normalized points of shape (N, 3).
    """
    original_spacing = torch.tensor(original_spacing, dtype=points.dtype, device=points.device)
    target_spacing   = torch.tensor(target_spacing, dtype=points.dtype, device=points.device)
    scale_factors    = original_spacing / target_spacing
    return points * scale_factors


def apply_deformation_to_keypoints(moving_keypoints, deformation_field, fixed_keypoints):
    """
    Apply the deformation field to the moving keypoints.
    
    Args:
        moving_keypoints (torch.Tensor): Tensor of moving keypoints of shape (B, N, 3).
        deformation_field (torch.Tensor): Tensor of deformation field of shape (B, 3, W, H, D).
    
    Returns:
        torch.Tensor: Transformed moving keypoints of shape (B, N, 3).
    """
    batch_size = moving_keypoints.shape[0]
    mov_kps = normalize_keypoints(torch.squeeze(moving_keypoints, 0), [1, 1, 1], [1, 1, 1])
    fix_kps = normalize_keypoints(torch.squeeze(fixed_keypoints, 0), [1, 1, 1], [1, 1, 1])
    mov_kps = mov_kps[None, :]
    fix_kps = fix_kps[None, :]
    transformed_keypoints = []

    for b in range(batch_size):
        batch_keypoints = mov_kps[b]
        
        batch_deformation_field = deformation_field[b]
        transformed_batch_keypoints = []
        for keypoint in batch_keypoints:
            x, y, z = keypoint
            dx = batch_deformation_field[0, int(x), int(y), int(z)]
            dy = batch_deformation_field[1, int(x), int(y), int(z)]
            dz = batch_deformation_field[2, int(x), int(y), int(z)]
            transformed_batch_keypoints.append([x + dx, y + dy, z + dz])
        
        transformed_keypoints.append(transformed_batch_keypoints)
    
    transformed_keypoints = torch.tensor(transformed_keypoints)
    data = np.hstack((convert_tensor_to_numpy(torch.squeeze(fixed_keypoints, 0).cpu()),
                      convert_tensor_to_numpy(torch.squeeze(moving_keypoints, 0).cpu()), 
                      convert_tensor_to_numpy(torch.squeeze(fix_kps, 0).cpu()),
                      convert_tensor_to_numpy(torch.squeeze(mov_kps, 0).cpu()),
                      convert_tensor_to_numpy(torch.squeeze(transformed_keypoints, 0).cpu())))
    
    columns = ['F_x_1.75', 'F_y_1.25', 'F_z_1.75', 
               'M_x_1.75', 'M_y_1.25', 'M_z_1.75', 
               'F_x_1', 'F_y_1', 'F_z_1', 
               'M_x_1', 'M_y_1', 'M_z_1', 
               'transformed_x', 'transformed_y', 'transformed_z']
    df = pd.DataFrame(data, columns=columns)
    df.to_csv('./normalized_keypoints.csv', index=False)
    
    return transformed_keypoints, fix_kps