import torch
import torch.nn as nn

# ------------ Backbone wrapper for DINO (or any ViT that can return intermediate layers) ------------
class DinoBackbone(nn.Module):
    """
    Wraps a DINO vision transformer to return spatial feature maps (B, C, H, W).
    Expects `dino_model.get_intermediate_layers(img, n=..., reshape=True, norm=True)` to be available.
    If your DINO API differs, replace `get_intermediate_layers` call accordingly.
    """
    def __init__(self, dino_model, n_layers = 12):
        """
        dino_model: pretrained DINO model instance
        layer_idx: which intermediate layer to extract (-1 for last)
        proj_dim: optional channel projection to reduce dimension for lightweight heads
        """
        super().__init__()
        self.dino = dino_model
        self.n_layers = n_layers

    def forward(self, x):
        """
        x: (B, 3, H, W) in range expected by DINO (make sure to normalize as the DINO expects)
        returns: (B, proj_dim, Hf, Wf)
        """
        # Get intermediate layers. Many DINO versions provide this helper:
        feats = self.dino.get_intermediate_layers(x, n=range(self.n_layers),
                                                     reshape=True, norm=True)
        # get_intermediate_layers often returns a list; pick the element
        feat = feats[-1]  # (B, C, Hf, Wf)

        return feat