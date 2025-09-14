import numpy as np
import cv2

def depth_to_colormap(
    depth_m: np.ndarray,
    dmin: float | None = None,
    dmax: float | None = None,
    colormap: int = cv2.COLORMAP_INFERNO,
    invert: bool = True,
) -> np.ndarray:
    """
    Convert an HxW depth map in meters to a color (BGR) visualization.

    Args:
        depth_m: HxW float array (meters). NaN or <=0 treated as invalid.
        dmin:   Near clip in meters. If None, uses 5th percentile of valid depths.
        dmax:   Far clip in meters.  If None, uses 95th percentile of valid depths.
        colormap: OpenCV colormap (e.g., cv2.COLORMAP_TURBO, JET, INFERNO...).
        invert:  If True, nearer = brighter/warmer (flip normalization).

    Returns:
        color_bgr: HxWx3 uint8 image in BGR order.
    """
    depth = np.asarray(depth_m, dtype=np.float32)
    assert depth.ndim == 2, "depth_m must be HxW"

    valid = np.isfinite(depth) & (depth > 0)

    # Choose visualization range
    if dmin is None or dmax is None:
        if np.any(valid):
            vals = depth[valid]
            if dmin is None:
                dmin = float(np.percentile(vals, 5))
            if dmax is None:
                dmax = float(np.percentile(vals, 95))
        else:
            dmin, dmax = 0.1, 1.0  # fallback if nothing valid
    if dmax <= dmin:
        dmax = dmin + 1e-6

    # Normalize to 0..255 for applyColorMap
    norm = (depth - dmin) / (dmax - dmin)
    norm = np.clip(norm, 0.0, 1.0)
    if invert:
        norm = 1.0 - norm
    gray8 = (norm * 255.0).astype(np.uint8)

    # Colorize (BGR)
    color_bgr = cv2.applyColorMap(gray8, colormap)

    # Paint invalid pixels black
    if not np.all(valid):
        color_bgr[~valid] = (0, 0, 0)

    return color_bgr