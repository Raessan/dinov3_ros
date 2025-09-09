import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask2FormerLiteHead(nn.Module):
    def __init__(self,
                 in_channels=384,
                 num_classes=134,   # things + stuff + background (semantic head)
                 hidden_dim=256,
                 target_size=(640, 640)):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.target_size = target_size

        # 1) Pixel embedding: project backbone -> hidden_dim
        self.pixel_embed = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # small separable conv to increase receptive field cheaply
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        )

        # 6) Semantic head: small conv classifier (operates at feature res, we'll upsample + refine)
        self.semantic_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        )

        # --- change the semantic_refine to operate on num_semantic_classes channels ---
        self.semantic_refine = nn.Sequential(
            # depthwise over classes (cheap) then pointwise to mix (keeps params small)
            nn.Conv2d(self.num_classes, self.num_classes,
                    kernel_size=3, padding=1, groups=self.num_classes, bias=False),
            nn.Conv2d(self.num_classes, self.num_classes, kernel_size=1)
        )

    def forward(self, features):
        """
        Args:
            features: tensor (B, in_channels, Hf, Wf)
        Returns:
            semantic_logits: (B, num_classes, H_img, W_img)
            mask_logits: (B, num_queries, H_img, W_img)
            class_logits: (B, num_queries, num_classes)
        """
        B, C, Hf, Wf = features.shape
        H_img, W_img = self.target_size

        # pixel embeddings
        pix = self.pixel_embed(features)   # (B, hidden_dim, Hf, Wf)

        # compute semantic logits at low feature resolution (cheap)
        semantic_logits_low = self.semantic_head(pix)  # (B, num_semantic_classes, Hf, Wf)
        # upsample logits (channels = num_semantic_classes, small)
        semantic_logits = F.interpolate(semantic_logits_low, size=(H_img, W_img),
                                        mode='bilinear', align_corners=False)

        # small refine operating on num_semantic_classes channels (cheap)
        semantic_logits = self.semantic_refine(semantic_logits)

        return semantic_logits

# ----------------- Quick smoke test / example -----------------
if __name__ == '__main__':

    from model_backbone import DinoBackbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 640
    image = torch.randn(1,3,img_size,img_size).to(device)
    dinov3_dir = '/home/rafa/deep_learning/projects/segmentation_dinov3/dinov3'
    dinov3_weights_path = '/home/rafa/deep_learning/projects/segmentation_dinov3/results/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
    dino_model = torch.hub.load(
        repo_or_dir=dinov3_dir,
        model="dinov3_vits16plus",
        source="local",
        weights=dinov3_weights_path
    )
    n_layers_dino = 12
    dino_backbone = DinoBackbone(dino_model, n_layers_dino).to(device)
    feat = dino_backbone(image)
    print(feat.shape)

    C = feat.shape[1]

    segmentation_head = Mask2FormerLiteHead(in_channels=C,
                                            num_classes=134,   # things + stuff + background (semantic head)
                                            hidden_dim=256,
                                            target_size=(320, 320)).to(device)
    
    
    # ----------------- Utility: parameter counting -----------------
    def count_parameters(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Print parameter counts
    print('Segmentation params: ', count_parameters(segmentation_head))

    # Forward pass
    semantic_logits = segmentation_head(feat)
    print("semantic_logits: ", semantic_logits.shape)