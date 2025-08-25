import cv2
import numpy as np
import torch
import torch.nn.functional as F

def resize_transform(
    img: np.ndarray,
    image_size: int,
    patch_size: int,
    interpolation=cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Resize or pad an image (NumPy ndarray) so its dimensions are divisible by patch_size.

    Args:
        img (np.ndarray): Input image in H x W x C (RGB) format.
        image_size (int): Target size for the smaller dimension (height).
        patch_size (int): Patch size to align width and height to multiples.
        interpolation (int): Interpolation method for resizing.
        pad_value (int): Fill value for padding.

    Returns:
        np.ndarray: Padded/resized image (RGB) with shape divisible by patch_size.
    """
    h, w = img.shape[:2]

    # Scale the height to image_size, adjust width to preserve aspect ratio
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))

    # Delete this line, is for test:
    w_patches = h_patches

    img_resized = cv2.resize(img, (w_patches*patch_size, h_patches*patch_size), interpolation=interpolation)

    return img_resized

def image_to_tensor(img, mean, std):
    """
    Converts an img to a tensor ready to be used in NN
    """
    img = img.astype(np.float32) / 255.
    img = (img.transpose(2,0,1) - mean) / std
    return torch.from_numpy(img)

def tensor_to_image(tensor, mean, std):
    """
    Convert normalized tensor back to image format for display (H x W x 3, uint8).
    """
    # tensor shape: (3, H, W)
    img = tensor.clone().cpu().float()
    # If batch dimension exists and is 1, squeeze it safely
    if img.ndim == 4 and img.shape[0] == 1:
        img = img.squeeze(0)  # shape now: C, H, W
    # Un-normalize
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    img = img * std + mean  # invert normalization
    # Clamp values to [0, 1], then to [0, 255]
    img = img.clamp(0, 1)
    # Convert to numpy and reshape to H x W x C
    img = img.permute(1, 2, 0).numpy()
    # Convert to uint8 for display
    img = (img * 255).astype(np.uint8)

    return img  # RGB format