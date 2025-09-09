import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms
import matplotlib.pyplot as plt

def resize_transform(
    img: np.ndarray,
    image_size: int,
    patch_size: int,
    interpolation=cv2.INTER_LINEAR,
    pad_value: int = 0
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

# --------- conversion: outputs -> semantic & panoptic maps -----------
def outputs_to_maps(semantic_logits, image_size):
    """
    Convert model outputs at full resolution to semantic + instance maps.

    Assumptions:
      - semantic_logits: torch.Tensor of shape (C,H,W) or (1,C,H,W).
                         channel indices should match your dataset's class ids.
    Returns:
      semantic_pred: (H,W) np.int32 semantic class indices (argmax of semantic head)
    """

    # add batch dim if missing
    if semantic_logits.dim() == 3:
        semantic_logits = semantic_logits.unsqueeze(0)  # 1,C,H,W

    H_img, W_img = image_size

    # Upsample logits to image resolution
    semantic_logits_up = F.interpolate(semantic_logits, size=(H_img, W_img), mode='bilinear', align_corners=False)[0]  # C,H,W       # Q,H,W

    # semantic prediction
    semantic_prob = F.softmax(semantic_logits_up.unsqueeze(0), dim=1)[0]  # (C,H,W) tensor
    semantic_pred = torch.argmax(semantic_prob, dim=0).cpu().numpy().astype(np.int32)  # H,W

    return semantic_pred



def generate_segmentation_overlay(
    image,
    semantic_map,
    class_names=None,
    alpha=0.6,
    background_index=0,
    seed=42,
    draw_semantic_labels=True,
    semantic_label_fontsize=10
):
    """
    Generate a segmentation overlay image with optional labels drawn.

    Args:
        image: H x W x 3 (RGB), uint8 or float in [0..1] or [0..255]
        semantic_map: H x W int (category ids)
        class_names: optional list mapping category id -> name
        alpha: overlay transparency
        background_index: category id for background
        seed: random seed for colors
        draw_semantic_labels: annotate overlay with class names
        semantic_label_fontsize: fontsize for semantic labels (used by OpenCV)

    Returns:
        out_sem (np.ndarray): H x W x 3 uint8 image with overlay and optional labels
    """

    # Normalize image to uint8
    img = np.array(image)
    if img.dtype in (np.float32, np.float64):
        if img.max() <= 1.0:
            img_u8 = (img * 255).astype(np.uint8)
        else:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_u8 = img.copy()

    H, W = semantic_map.shape[:2]

    # Palette for colors
    rng = np.random.RandomState(seed)
    n_colors = max(100, int(np.max(semantic_map) + 1))
    palette = (rng.randint(0, 256, size=(n_colors, 3))).astype(np.uint8)

    # Helper: overlay one mask
    def overlay(base, mask, color, alpha):
        out = base.astype(np.float32).copy()
        color_f = np.array(color, dtype=np.float32).reshape(1, 1, 3)
        m3 = np.stack([mask]*3, axis=-1).astype(np.float32)
        out = out * (1 - alpha * m3) + color_f * (alpha * m3)
        return out.clip(0, 255).astype(np.uint8)

    # Apply overlays for all classes
    out_sem = img_u8.copy()
    unique_c = np.unique(semantic_map)
    label_info = []  # store (cls, y0, x0) for labels

    for cls in unique_c:
        if cls < 0 or cls == background_index:
            continue
        mask = (semantic_map == int(cls))
        if mask.sum() == 0:
            continue

        color = palette[int(cls) % len(palette)]
        out_sem = overlay(out_sem, mask, color, alpha)

        # Save centroid info for later labeling
        ys, xs = np.nonzero(mask)
        if ys.size > 0:
            y0 = int(np.mean(ys))
            x0 = int(np.mean(xs))
            label_info.append((cls, y0, x0))

    # Draw labels at the end, after all overlays
    if draw_semantic_labels:
        for cls, y0, x0 in label_info:
            label = class_names[cls] if (class_names is not None and cls < len(class_names)) else str(cls)
            font_scale = semantic_label_fontsize / 10
            thickness = max(1, semantic_label_fontsize // 4)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            # Draw black background rectangle
            cv2.rectangle(out_sem, (x0 - text_w//2 - 2, y0 - text_h//2 - 2),
                          (x0 + text_w//2 + 2, y0 + text_h//2 + 2), (0, 0, 0), -1)
            # Draw white text
            cv2.putText(out_sem, label, (x0 - text_w//2, y0 + text_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return out_sem


def visualize_maps(image, semantic_map, class_names=None, alpha=0.6, figsize=(12, 8),
                   draw_semantic_labels=True, semantic_label_fontsize=10, background_index=0, seed=42):
    overlay_img = generate_segmentation_overlay(
        image, semantic_map, class_names=class_names, alpha=alpha,
        draw_semantic_labels=draw_semantic_labels, semantic_label_fontsize=semantic_label_fontsize,
        background_index=background_index, seed=seed
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(overlay_img)
    axes[1].set_title("Semantic map")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()