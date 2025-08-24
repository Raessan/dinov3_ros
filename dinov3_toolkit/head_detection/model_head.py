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
        # 1x1 projection
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # downsample convs to create P4 and P5
        self.down1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # smoothers after top-down
        self.smooth3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

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
                 prior_prob: float = 0.01):
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

        # final prediction convs
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.bbox_reg = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        # init bias for focal loss (prior)
        nn.init.constant_(self.cls_logits.bias, -torch.log(torch.tensor((1 - prior_prob) / prior_prob)))
        # small init for heads
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
        cls_outputs = []
        reg_outputs = []
        ctr_outputs = []
        for f in features:
            cls_feat = self.cls_tower(f)
            reg_feat = self.reg_tower(f)

            cls_out = self.cls_logits(cls_feat)
            reg_out = self.bbox_reg(reg_feat)
            reg_out = F.softplus(reg_out) 
            ctr_out = self.centerness(reg_feat)

            # reg_out is predicted as raw offsets; training code often applies exp/clamp
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