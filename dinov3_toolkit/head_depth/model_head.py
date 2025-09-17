import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- small helpers ---------
def make_norm(norm, ch, groups_gn=8):
    return nn.GroupNorm(min(groups_gn, ch), ch) if norm == 'gn' else nn.BatchNorm2d(ch)

class DSConv(nn.Module):
    """Depthwise 3x3 -> Norm -> GELU -> Pointwise 1x1 -> Norm -> GELU."""
    def __init__(self, in_ch, out_ch, norm='gn', groups_gn=8):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.bn1 = make_norm(norm, in_ch, groups_gn)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = make_norm(norm, out_ch, groups_gn)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

class DSDown2(nn.Module):
    """Depthwise stride-2 downsample + pointwise (cheap and crisp)."""
    def __init__(self, ch, norm='gn', groups_gn=8):
        super().__init__()
        self.dw  = nn.Conv2d(ch, ch, 3, stride=2, padding=1, groups=ch, bias=False)
        self.bn1 = make_norm(norm, ch, groups_gn)
        self.pw  = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn2 = make_norm(norm, ch, groups_gn)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x

class FeatureFusionBlockLite(nn.Module):
    """Project + (optional) skip add + one DS residual to refine."""
    def __init__(self, ch=160, norm='gn', groups_gn=8, use_skip=True):
        super().__init__()
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn   = make_norm(norm, ch, groups_gn)
        self.act  = nn.GELU()
        self.res  = nn.Sequential(DSConv(ch, ch, norm=norm, groups_gn=groups_gn))
        self.use_skip = use_skip
    def forward(self, x, skip=None):
        x = self.act(self.bn(self.proj(x)))
        if self.use_skip and skip is not None:
            x = x + skip
        x = self.res(x)
        return x

# --------- DPT-mini head for DINOv3 ---------
class DepthHeadLite(nn.Module):
    """
    Small DPT-like decoder:
      - Reassemble 4 branches from one DINOv3 map (↑×4, ↑×2, =, ↓×2)
      - Map to common width (common_ch)
      - Progressive fusion (coarse -> fine)
      - Tiny depth head

    Default config is ~1–2M params (well < 5M).
    """
    def __init__(self,
                 in_ch=384,
                 out_size=(640, 640),
                 # per-branch project widths (smaller = fewer params)
                 proj0_ch=32,   # will be upsampled x4
                 proj1_ch=64,   # upsampled x2
                 proj2_ch=96,   # identity
                 proj3_ch=128,  # downsampled x2
                 common_ch=160, # common fusion width
                 dropout=0.2,
                 norm='gn',
                 groups_gn=8,
                 align_corners=False):
        super().__init__()
        self.out_size = out_size
        self.align_corners = align_corners

        self.drop = nn.Dropout2d(p=dropout)

        # 1×1 projections from (384) to per-branch widths
        self.proj0 = nn.Conv2d(in_ch, proj0_ch, 1, bias=False)
        self.proj1 = nn.Conv2d(in_ch, proj1_ch, 1, bias=False)
        self.proj2 = nn.Conv2d(in_ch, proj2_ch, 1, bias=False)
        self.proj3 = nn.Conv2d(in_ch, proj3_ch, 1, bias=False)

        # Resizing (parameter-free up; cheap DS down for the coarsest)
        self.down3 = DSDown2(proj3_ch, norm=norm, groups_gn=groups_gn)

        # Map each branch to common width with DSConv (cheap)
        self.to_c0 = DSConv(proj0_ch, common_ch, norm=norm, groups_gn=groups_gn)
        self.to_c1 = DSConv(proj1_ch, common_ch, norm=norm, groups_gn=groups_gn)
        self.to_c2 = DSConv(proj2_ch, common_ch, norm=norm, groups_gn=groups_gn)
        self.to_c3 = DSConv(proj3_ch, common_ch, norm=norm, groups_gn=groups_gn)

        # Fusion (4 stages)
        self.fuse3 = FeatureFusionBlockLite(common_ch, norm=norm, groups_gn=groups_gn, use_skip=False) # start
        self.fuse2 = FeatureFusionBlockLite(common_ch, norm=norm, groups_gn=groups_gn, use_skip=True)
        self.fuse1 = FeatureFusionBlockLite(common_ch, norm=norm, groups_gn=groups_gn, use_skip=True)
        self.fuse0 = FeatureFusionBlockLite(common_ch, norm=norm, groups_gn=groups_gn, use_skip=True)

        # Tiny depth head: conv -> upsample -> conv -> ReLU -> 1x1
        self.h1 = nn.Conv2d(common_ch, common_ch//2, 3, padding=1)
        self.h2 = nn.Conv2d(common_ch//2, common_ch//4, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(common_ch//4, 1, 1)
        self.softplus = nn.Softplus(beta=1.0, threshold=20.0)

        # Norms for head
        self.h1_bn = make_norm(norm, common_ch//2, groups_gn)
        self.h2_bn = make_norm(norm, common_ch//4, groups_gn)

        # Min depth
        self.min_depth = 1e-3

    def _upsample(self, x, size):
        return F.interpolate(x, size=size, mode='bilinear', align_corners=self.align_corners)

    def forward(self, feat_1x):  # (B,384,Hf,Wf), e.g., 40x40 for 640 input with stride 16
        B, C, Hf, Wf = feat_1x.shape

        # Reassemble branches
        b0 = self.proj0(feat_1x)                        # (Hf, Wf)   -> later ↑×4
        b1 = self.proj1(feat_1x)                        # (Hf, Wf)   -> later ↑×2
        b2 = self.proj2(feat_1x)                        # (Hf, Wf)   -> =
        b3 = self.proj3(feat_1x)                        # (Hf, Wf)   -> ↓×2
        b3 = self.down3(b3)                              # (Hf/2, Wf/2)

        # Map to common width
        f0 = self.to_c0(self._upsample(b0, (Hf*4, Wf*4)))  # (4Hf, 4Wf)
        f1 = self.to_c1(self._upsample(b1, (Hf*2, Wf*2)))  # (2Hf, 2Wf)
        f2 = self.to_c2(b2)                                # (Hf,  Wf)
        f3 = self.to_c3(b3)                                # (Hf/2,Wf/2)

        # Progressive fusion: start coarse (f3) and go to fine (f0)
        x = self.fuse3(f3, None)                           # (Hf/2, Wf/2)
        x = self._upsample(x, f2.shape[-2:]); x = self.fuse2(x, f2)   # -> (Hf,  Wf)
        x = self._upsample(x, f1.shape[-2:]); x = self.fuse1(x, f1)   # -> (2Hf, 2Wf)
        x = self.drop(x)
        x = self._upsample(x, f0.shape[-2:]); x = self.fuse0(x, f0)   # -> (4Hf, 4Wf)

        # Small head
        x = self.relu(self.h1_bn(self.h1(x)))
        x = self.drop(x)
        x = self._upsample(x, (x.shape[-2]*2, x.shape[-1]*2))         # one gentle upsample
        x = self.relu(self.h2_bn(self.h2(x)))
        x = self.out(x)

        # To make depth positive
        x = self.softplus(x) + self.min_depth

        # Final resize to requested output (e.g., 640x640)
        x = self._upsample(x, self.out_size)                           # (B,1,H,W)
        return x
    
# ----------------- Quick smoke test / example -----------------
if __name__ == '__main__':

    from model_backbone import DinoBackbone
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    img_size = 640
    image = torch.randn(4,3,img_size,img_size).to(device)
    dinov3_dir = '/home/rafa/deep_learning/projects/depth_dinov3/dinov3'
    dinov3_weights_path = '/home/rafa/deep_learning/projects/depth_dinov3/results/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'
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

    depth_head = DepthHeadLite(in_ch=C, out_size=(640, 640)).to(device)
    
    # ----------------- Utility: parameter counting -----------------
    def count_parameters(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Print parameter counts
    print('Depth params: ', count_parameters(depth_head))

    # Forward pass
    depth_map = depth_head(feat)
    print("depth_map: ", depth_map.shape)