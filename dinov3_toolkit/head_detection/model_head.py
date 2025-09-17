import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
    
# ----------------- Lightweight conv helpers -----------------
class ConvGNReLU(nn.Module):
    """3x3 conv + GroupNorm + ReLU
    GroupNorm is useful for small batch sizes (common in detection training).
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        # groups must divide out_ch; choose min(groups, out_ch)
        gn_groups = min(groups, out_ch)
        self.gn = nn.GroupNorm(gn_groups, out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))
    
# ----------------- Lightweight FPN -----------------
class LightFPN(nn.Module):
    """
    Builds a 3-level FPN (P3, P4, P5) from a single input feature map (from ViT/DINO).
    The design:
      - proj: 1x1 projection from backbone dim -> C
      - P3 = proj(x)
      - P4 = stride-2 conv(P3)
      - P5 = stride-2 conv(P4)
      - top-down upsample and add with 3x3 smoothers

    This keeps things simple and lightweight when the backbone is a ViT that returns
    a single spatial feature map.
    """
    def __init__(self, in_channels: int, out_channels: int = 192):
        super().__init__()
        self.out_channels = out_channels
        # 1x1 projection -> use ConvGNReLU (1x1 conv with GN+ReLU)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True)
        )

        # downsample convs to create P4 and P5
        self.down1 = ConvGNReLU(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.down2 = ConvGNReLU(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # smoothers after top-down: use ConvGNReLU (3x3 conv + GN + ReLU)
        self.smooth3 = ConvGNReLU(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = ConvGNReLU(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth5 = ConvGNReLU(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # x: (B, C_in, H, W)
        p3 = self.proj(x)               # (B, C, H, W)
        p4 = self.down1(p3)             # (B, C, H/2, W/2)
        p5 = self.down2(p4)             # (B, C, H/4, W/4)

        # top-down
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_td = p4 + p5_up
        p4_td = self.smooth4(p4_td)

        p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_td = p3 + p4_up
        p3_td = self.smooth3(p3_td)

        p5_td = self.smooth5(p5)

        # Return in increasing stride order like common detectors: [P3, P4, P5]
        return [p3_td, p4_td, p5_td]


# ----------------- FCOS-style head -----------------
class FCOSHead(nn.Module):
    """
    FCOS-style anchor-free head with classification, regression and centerness branches.
    - num_classes: number of object classes (COCO -> 80)
    - in_channels: channels coming from FPN (== out_channels of LightFPN)
    - num_convs: depth of the conv towers (4 in the canonical design)
    """
    def __init__(self, in_channels: int = 192, num_classes: int = 80, num_convs: int = 4,
                 prior_prob: float = 0.01, num_levels: int = 3):  # set num_levels to len(FPN)
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        cls_tower = []
        reg_tower = []
        for _ in range(num_convs):
            cls_tower.append(ConvGNReLU(in_channels, in_channels))
            reg_tower.append(ConvGNReLU(in_channels, in_channels))
        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        # final convs
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_reg = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # learnable scale per level (so output magnitude per P-level can adapt)
        self.scales = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(num_levels)])

        # init bias for focal loss (prior)
        nn.init.constant_(self.cls_logits.bias, -torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        for l in [self.bbox_reg, self.centerness]:
            nn.init.normal_(l.weight, std=0.001)
            if l.bias is not None:
                nn.init.zeros_(l.bias)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, List[torch.Tensor]]:
        """
        features: list of feature maps [P3, P4, P5], each (B, C, H_i, W_i)
        returns dict with keys: 'cls', 'reg', 'ctr' each a list of tensors per level
        cls logits: (B, num_classes, H_i, W_i)
        reg: (B, 4, H_i, W_i)
        ctr: (B, 1, H_i, W_i)
        """
        cls_outputs, reg_outputs, ctr_outputs = [], [], []
        for l, f in enumerate(features):
            cls_feat = self.cls_tower(f)
            reg_feat = self.reg_tower(f)

            cls_out = self.cls_logits(cls_feat)
            reg_raw = self.bbox_reg(reg_feat)  # raw values

            reg_out = F.softplus(reg_raw) * self.scales[l]

            ctr_out = self.centerness(reg_feat)

            cls_outputs.append(cls_out)
            reg_outputs.append(reg_out)
            ctr_outputs.append(ctr_out)

        return {"cls": cls_outputs, "reg": reg_outputs, "ctr": ctr_outputs}


# ----------------- Full Head module that plugs onto DinoBackbone -----------------
class DinoFCOSHead(nn.Module):
    """
    Wraps LightFPN + FCOSHead. Expects the backbone to provide a single spatial
    feature map (B, C_backbone, Hf, Wf). This matches the DinoBackbone you
    posted earlier.
    """
    def __init__(self, backbone_out_channels: int = 768, fpn_channels: int = 192,
                 num_classes: int = 80, num_convs: int = 4):
        super().__init__()
        self.fpn = LightFPN(backbone_out_channels, out_channels=fpn_channels)
        self.head = FCOSHead(in_channels=fpn_channels, num_classes=num_classes, num_convs=num_convs)

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        # x is the backbone feature map (B, C_backbone, Hf, Wf)
        pyramid = self.fpn(x)
        out = self.head(pyramid)
        return out

# ----------------- Quick smoke test / example -----------------
if __name__ == '__main__':

    from model_backbone import DinoBackbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    height = 800
    width = 800
    image = torch.randn(1,3,height, width).to(device)
    dinov3_dir = "/home/rafa/deep_learning/projects/object_detection_dinov3/dinov3"
    dino_model = torch.hub.load(
        repo_or_dir=dinov3_dir,
        model="dinov3_vits16plus",
        source="local"
    )
    n_layers_dino = 12
    dino_backbone = DinoBackbone(dino_model, n_layers_dino).to(device)
    feat = dino_backbone(image)
    print(feat.shape)

    C = feat.shape[1]

    fpn_ch = 192
    num_classes = 80
    full_head = DinoFCOSHead(backbone_out_channels=C, fpn_channels=fpn_ch, num_classes=num_classes).to(device)

    # Forward pass
    outputs = full_head(feat)
    print('Outputs keys:', outputs.keys())
    for k, v in outputs.items():
        print(f"{k} levels: {[t.shape for t in v]}")

    # ----------------- Utility: parameter counting -----------------
    def count_parameters(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Print parameter counts
    print('FPN params: ', count_parameters(full_head.fpn))
    print('Head params: ', count_parameters(full_head.head))
    print('Total head params (fpn+head): ', count_parameters(full_head))