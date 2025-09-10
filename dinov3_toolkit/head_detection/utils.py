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

def decode_outputs(
    model_outputs,   # dict with lists per-level like in your model
    image_shape,     # (H_img, W_img)
    strides = [16, 32, 64],
    score_thresh = 0.05,
    nms_thresh = 0.6,
    max_detections = 100
):
    """
    Decode FCOS-style model outputs (assumes reg outputs are l,t,r,b in pixels and >=0).
    Returns: boxes (K,4) xyxy, scores (K,), labels (K,) (labels are 0..C-1)
    Notes:
      - Assumes batch size 1 at inference. If B>1 adapt accordingly.
      - Make sure `strides` correspond to each FPN level returned in model_outputs['cls'].
    """
    device = model_outputs['cls'][0].device
    num_levels = len(model_outputs['cls'])

    all_boxes = []
    all_scores = []
    all_labels = []

    for lvl_idx in range(num_levels):
        cls_logits = model_outputs['cls'][lvl_idx]  # (B,C,H,W)
        reg_out = model_outputs['reg'][lvl_idx]     # (B,4,H,W)
        ctr_logits = model_outputs['ctr'][lvl_idx]  # (B,1,H,W)

        # assume B == 1 for inference
        assert cls_logits.shape[0] == 1, "decode_fcos_outputs assumes batch size 1 during inference."
        cls_logits = cls_logits[0]   # (C,H,W)
        reg_out = reg_out[0]         # (4,H,W)
        ctr_logits = ctr_logits[0]   # (1,H,W)

        C, H, W = cls_logits.shape
        stride = strides[lvl_idx]

        # build center coordinates in image pixels (x,y)
        shifts_x = (torch.arange(0, W, device=device) + 0.5) * stride
        shifts_y = (torch.arange(0, H, device=device) + 0.5) * stride
        ys, xs = torch.meshgrid(shifts_y, shifts_x, indexing='ij')  # (H,W)
        xs = xs.reshape(-1)  # (Nloc,)
        ys = ys.reshape(-1)
        centers = torch.stack([xs, ys], dim=1)  # (Nloc,2)

        # flatten tensors
        cls_logits_flat = cls_logits.reshape(C, -1).permute(1, 0)  # (Nloc, C)
        reg_flat = reg_out.reshape(4, -1).permute(1, 0)            # (Nloc, 4)
        ctr_logits_flat = ctr_logits.reshape(-1)                  # (Nloc,)

        # probabilities
        cls_prob = torch.sigmoid(cls_logits_flat)   # (Nloc, C)
        ctr_prob = torch.sigmoid(ctr_logits_flat).unsqueeze(1)  # (Nloc,1)
        final_scores = cls_prob * ctr_prob  # (Nloc, C)

        # keep locations where any class score exceeds threshold
        keep_mask_any = (final_scores > score_thresh).any(dim=1)
        if keep_mask_any.sum() == 0:
            continue

        centers_keep = centers[keep_mask_any]                     # (Nk,2)
        # reg_out is produced by the network already passed through ReLU/softplus in the head;
        # nevertheless, clamp to >=0 to be robust.
        reg_keep = reg_flat[keep_mask_any].clamp(min=0.0)         # (Nk,4)
        scores_keep = final_scores[keep_mask_any]                # (Nk, C)

        Nk = scores_keep.shape[0]
        cls_idx_grid = torch.arange(C, device=device).unsqueeze(0).expand(Nk, C)  # (Nk, C)

        # decode per-location per-class boxes
        x_cent = centers_keep[:, 0].unsqueeze(1).expand(-1, C)  # (Nk, C)
        y_cent = centers_keep[:, 1].unsqueeze(1).expand(-1, C)
        l = reg_keep[:, 0].unsqueeze(1).expand(-1, C)
        t = reg_keep[:, 1].unsqueeze(1).expand(-1, C)
        r = reg_keep[:, 2].unsqueeze(1).expand(-1, C)
        b = reg_keep[:, 3].unsqueeze(1).expand(-1, C)

        x1 = x_cent - l
        y1 = y_cent - t
        x2 = x_cent + r
        y2 = y_cent + b

        boxes_lvl = torch.stack([x1, y1, x2, y2], dim=2).reshape(-1, 4)  # (Nk*C, 4)
        scores_lvl = scores_keep.reshape(-1)                            # (Nk*C,)
        labels_lvl = cls_idx_grid.reshape(-1)                           # (Nk*C,)

        # filter by score threshold
        keep = scores_lvl > score_thresh
        if keep.sum() == 0:
            continue

        boxes_lvl = boxes_lvl[keep]
        scores_lvl = scores_lvl[keep]
        labels_lvl = labels_lvl[keep]

        all_boxes.append(boxes_lvl)
        all_scores.append(scores_lvl)
        all_labels.append(labels_lvl)

    # no detections
    if len(all_boxes) == 0:
        return (torch.empty((0,4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.long, device=device))

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # clip to image boundaries
    Himg, Wimg = image_shape
    boxes[:, 0].clamp_(min=0.0, max=float(Wimg))
    boxes[:, 1].clamp_(min=0.0, max=float(Himg))
    boxes[:, 2].clamp_(min=0.0, max=float(Wimg))
    boxes[:, 3].clamp_(min=0.0, max=float(Himg))

    # per-class NMS
    keep_indices = []
    unique_labels = labels.unique()
    for lab in unique_labels:
        lab_mask = (labels == lab)
        boxes_l = boxes[lab_mask]
        scores_l = scores[lab_mask]
        if boxes_l.numel() == 0:
            continue
        keep_l = nms(boxes_l, scores_l, nms_thresh)
        if keep_l.numel() == 0:
            continue
        # convert to global indices
        global_idx = torch.nonzero(lab_mask, as_tuple=False).squeeze(1)[keep_l]
        keep_indices.append(global_idx)

    if len(keep_indices) == 0:
        return (torch.empty((0,4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.long, device=device))

    keep_indices = torch.cat(keep_indices)

    # sort by score descending and keep top-k
    sorted_idx = torch.argsort(scores[keep_indices], descending=True)
    sorted_idx = keep_indices[sorted_idx][:max_detections]

    return boxes[sorted_idx], scores[sorted_idx], labels[sorted_idx]

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

def decode_outputs(
    model_outputs,   # dict with lists per-level like in your model
    image_shape,     # (H_img, W_img)
    strides = [16, 32, 64],
    score_thresh = 0.05,
    nms_thresh = 0.6,
    max_detections = 100
):
    """
    Decode FCOS-style model outputs (assumes reg outputs are l,t,r,b in pixels and >=0).
    Returns: boxes (K,4) xyxy, scores (K,), labels (K,) (labels are 0..C-1)
    Notes:
      - Assumes batch size 1 at inference. If B>1 adapt accordingly.
      - Make sure `strides` correspond to each FPN level returned in model_outputs['cls'].
    """
    device = model_outputs['cls'][0].device
    num_levels = len(model_outputs['cls'])

    all_boxes = []
    all_scores = []
    all_labels = []

    for lvl_idx in range(num_levels):
        cls_logits = model_outputs['cls'][lvl_idx]  # (B,C,H,W)
        reg_out = model_outputs['reg'][lvl_idx]     # (B,4,H,W)
        ctr_logits = model_outputs['ctr'][lvl_idx]  # (B,1,H,W)

        # assume B == 1 for inference
        assert cls_logits.shape[0] == 1, "decode_fcos_outputs assumes batch size 1 during inference."
        cls_logits = cls_logits[0]   # (C,H,W)
        reg_out = reg_out[0]         # (4,H,W)
        ctr_logits = ctr_logits[0]   # (1,H,W)

        C, H, W = cls_logits.shape
        stride = strides[lvl_idx]

        # build center coordinates in image pixels (x,y)
        shifts_x = (torch.arange(0, W, device=device) + 0.5) * stride
        shifts_y = (torch.arange(0, H, device=device) + 0.5) * stride
        ys, xs = torch.meshgrid(shifts_y, shifts_x, indexing='ij')  # (H,W)
        xs = xs.reshape(-1)  # (Nloc,)
        ys = ys.reshape(-1)
        centers = torch.stack([xs, ys], dim=1)  # (Nloc,2)

        # flatten tensors
        cls_logits_flat = cls_logits.reshape(C, -1).permute(1, 0)  # (Nloc, C)
        reg_flat = reg_out.reshape(4, -1).permute(1, 0)            # (Nloc, 4)
        ctr_logits_flat = ctr_logits.reshape(-1)                  # (Nloc,)

        # probabilities
        cls_prob = torch.sigmoid(cls_logits_flat)   # (Nloc, C)
        ctr_prob = torch.sigmoid(ctr_logits_flat).unsqueeze(1)  # (Nloc,1)
        final_scores = cls_prob * ctr_prob  # (Nloc, C)

        # keep locations where any class score exceeds threshold
        keep_mask_any = (final_scores > score_thresh).any(dim=1)
        if keep_mask_any.sum() == 0:
            continue

        centers_keep = centers[keep_mask_any]                     # (Nk,2)
        # reg_out is produced by the network already passed through ReLU/softplus in the head;
        # nevertheless, clamp to >=0 to be robust.
        reg_keep = reg_flat[keep_mask_any].clamp(min=0.0)         # (Nk,4)
        scores_keep = final_scores[keep_mask_any]                # (Nk, C)

        Nk = scores_keep.shape[0]
        cls_idx_grid = torch.arange(C, device=device).unsqueeze(0).expand(Nk, C)  # (Nk, C)

        # decode per-location per-class boxes
        x_cent = centers_keep[:, 0].unsqueeze(1).expand(-1, C)  # (Nk, C)
        y_cent = centers_keep[:, 1].unsqueeze(1).expand(-1, C)
        l = reg_keep[:, 0].unsqueeze(1).expand(-1, C)
        t = reg_keep[:, 1].unsqueeze(1).expand(-1, C)
        r = reg_keep[:, 2].unsqueeze(1).expand(-1, C)
        b = reg_keep[:, 3].unsqueeze(1).expand(-1, C)

        x1 = x_cent - l
        y1 = y_cent - t
        x2 = x_cent + r
        y2 = y_cent + b

        boxes_lvl = torch.stack([x1, y1, x2, y2], dim=2).reshape(-1, 4)  # (Nk*C, 4)
        scores_lvl = scores_keep.reshape(-1)                            # (Nk*C,)
        labels_lvl = cls_idx_grid.reshape(-1)                           # (Nk*C,)

        # filter by score threshold
        keep = scores_lvl > score_thresh
        if keep.sum() == 0:
            continue

        boxes_lvl = boxes_lvl[keep]
        scores_lvl = scores_lvl[keep]
        labels_lvl = labels_lvl[keep]

        all_boxes.append(boxes_lvl)
        all_scores.append(scores_lvl)
        all_labels.append(labels_lvl)

    # no detections
    if len(all_boxes) == 0:
        return (torch.empty((0,4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.long, device=device))

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    # clip to image boundaries
    Himg, Wimg = image_shape
    boxes[:, 0].clamp_(min=0.0, max=float(Wimg))
    boxes[:, 1].clamp_(min=0.0, max=float(Himg))
    boxes[:, 2].clamp_(min=0.0, max=float(Wimg))
    boxes[:, 3].clamp_(min=0.0, max=float(Himg))

    # per-class NMS
    keep_indices = []
    unique_labels = labels.unique()
    for lab in unique_labels:
        lab_mask = (labels == lab)
        boxes_l = boxes[lab_mask]
        scores_l = scores[lab_mask]
        if boxes_l.numel() == 0:
            continue
        keep_l = nms(boxes_l, scores_l, nms_thresh)
        if keep_l.numel() == 0:
            continue
        # convert to global indices
        global_idx = torch.nonzero(lab_mask, as_tuple=False).squeeze(1)[keep_l]
        keep_indices.append(global_idx)

    if len(keep_indices) == 0:
        return (torch.empty((0,4), device=device),
                torch.empty((0,), device=device),
                torch.empty((0,), dtype=torch.long, device=device))

    keep_indices = torch.cat(keep_indices)

    # sort by score descending and keep top-k
    sorted_idx = torch.argsort(scores[keep_indices], descending=True)
    sorted_idx = keep_indices[sorted_idx][:max_detections]

    return boxes[sorted_idx], scores[sorted_idx], labels[sorted_idx]

def generate_detection_overlay(image_np, boxes_xyxy, scores, labels, class_names=None):
    """
    Return image with detections drawn on it (numpy array).
    image_np: HxWx3 numpy (uint8 or float).
    boxes_xyxy: tensor (K,4) xyxy in pixels (can be on GPU or CPU).
    """

    # move to cpu numpy
    boxes_np = boxes_xyxy.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)

    # convert image to uint8 if float in [0,1]
    if image_np.dtype.kind == 'f':
        img = (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img = image_np.astype(np.uint8)

    img_out = img.copy()

    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = map(int, box.tolist())
        score = float(scores_np[i])
        lab = int(labels_np[i])
        label_text = f"{lab}:{score:.2f}" if class_names is None else f"{class_names[lab]}:{score:.2f}"

        # Draw rectangle
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red box

        # Put text above box
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_out, (x1, y1 - th - 4), (x1 + tw, y1), (0, 0, 255), -1)  # filled background
        cv2.putText(img_out, label_text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img_out


def detection_inference(model_detection, feats, img_size, score_thresh=0.2, nms_thresh=0.6):
    """
    Run object detection inference using a detection model.

    Args:
        model_detection: The detection model to apply (takes feature maps as input).
        feats: Feature maps extracted from an image (usually backbone outputs).
        img_size: (H, W) of the image
        score_thresh: Minimum confidence score threshold to keep predictions.
        nms_thresh: IoU threshold for non-maximum suppression (NMS).

    Returns:
        boxes: Tensor of predicted bounding boxes (after NMS).
        scores: Confidence scores for each predicted box.
        labels: Predicted class labels for each box.
    """

    # Forward pass through the detection model to get raw outputs
    outputs = model_detection(feats)


    # Compute the stride (scaling factor) of the first output level
    first_stride = img_size[0] / outputs['cls'][0].shape[2]

    # Collect strides for each feature level (used for decoding predictions)
    strides = [first_stride]
    for l in range(1, len(outputs['cls'])):
        strides.append(first_stride * 2**l)

    # Decode raw model outputs into final predictions
    # Applies thresholding and non-maximum suppression
    boxes, scores, labels = decode_outputs(
        outputs,
        img_size,   # Original image size
        strides,
        score_thresh=score_thresh,
        nms_thresh=nms_thresh
    )

    return boxes, scores, labels

def generate_detection_overlay(image_np, boxes_xyxy, scores, labels, class_names=None):
    """
    Return image with detections drawn on it (numpy array).
    image_np: HxWx3 numpy (uint8 or float).
    boxes_xyxy: tensor (K,4) xyxy in pixels (can be on GPU or CPU).
    """

    # move to cpu numpy
    boxes_np = boxes_xyxy.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy().astype(int)

    # convert image to uint8 if float in [0,1]
    if image_np.dtype.kind == 'f':
        img = (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        img = image_np.astype(np.uint8)

    img_out = img.copy()

    for i, box in enumerate(boxes_np):
        x1, y1, x2, y2 = map(int, box.tolist())
        score = float(scores_np[i])
        lab = int(labels_np[i])
        label_text = f"{lab}:{score:.2f}" if class_names is None else f"{class_names[lab]}:{score:.2f}"

        # Draw rectangle
        cv2.rectangle(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red box

        # Put text above box
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_out, (x1, y1 - th - 4), (x1 + tw, y1), (0, 0, 255), -1)  # filled background
        cv2.putText(img_out, label_text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return img_out


def plot_detections(image_np, boxes_xyxy, scores, labels, class_names=None, figsize=(10,10)):
    """
    Plot detections. image_np: HxWx3 numpy (uint8 or float).
    boxes_xyxy: tensor (K,4) xyxy in pixels (can be on GPU or CPU).
    """

    fig, ax = plt.subplots(1,1, figsize=figsize)
    # convert image to uint8 if float in [0,1]
    img = generate_detection_overlay(image_np, boxes_xyxy, scores, labels, class_names)
    ax.imshow(img)

    ax.axis('off')
    plt.show()