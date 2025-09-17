import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return self.act(x)

class ASPPDecoder(nn.Module):
    def __init__(self, num_classes, in_ch=384, proj_ch=576, branch_ch=288, target_size=(640, 640)):
        """
        num_classes: NC
        in_ch: 384 from DINOv3
        proj_ch: top-level projection (here 576)
        branch_ch: per-branch channel after projection (here 288)
        This configuration yields ≈4.39M params for NC=21.
        """
        super().__init__()
        self.target_size = target_size
        # initial expand: 384 -> proj_ch
        self.initial = nn.Sequential(
            nn.Conv2d(in_ch, proj_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True)
        )

        # project for ASPP branches: proj_ch -> branch_ch
        self.project = nn.Sequential(
            nn.Conv2d(proj_ch, branch_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True)
        )

        # 4 parallel 3x3 conv branches (simulating lightweight ASPP)
        self.branches = nn.ModuleList([
            nn.Conv2d(branch_ch, branch_ch, kernel_size=3, padding=1, bias=True)
            for _ in range(4)
        ])
        self.branches_bn = nn.ModuleList([nn.BatchNorm2d(branch_ch) for _ in range(4)])
        # concat -> project
        self.fuse = nn.Sequential(
            nn.Conv2d(branch_ch * 4, proj_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(proj_ch),
            nn.ReLU(inplace=True)
        )

        # one DS block on proj_ch
        self.ds = DepthwiseSeparable(proj_ch, proj_ch)

        # final classifier head
        self.classifier = nn.Conv2d(proj_ch, num_classes, kernel_size=1, bias=True)

    def forward(self, x,):
        # x: (B, 384, 40, 40)
        x = self.initial(x)
        x = self.project(x)
        branches_out = []
        for conv, bn in zip(self.branches, self.branches_bn):
            y = conv(x)
            y = bn(y)
            y = F.relu(y, inplace=True)
            branches_out.append(y)
        x = torch.cat(branches_out, dim=1)  # B, branch_ch*4, h, w
        x = self.fuse(x)                     # B, proj_ch, h, w
        x = self.ds(x)                       # B, proj_ch, h, w
        logits = self.classifier(x)          # B, NC, h, w

        logits = F.interpolate(logits, size=self.target_size, mode='bilinear', align_corners=False)
        return logits

# ----------------- Quick smoke test / example -----------------
if __name__ == '__main__':

    from model_backbone import DinoBackbone
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = 640
    image = torch.randn(4,3,img_size,img_size).to(device)
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

    segmentation_head = ASPPDecoder(num_classes=134, in_ch=C, target_size=(320, 320)).to(device)
    
    # ----------------- Utility: parameter counting -----------------
    def count_parameters(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Print parameter counts
    print('Segmentation params: ', count_parameters(segmentation_head))

    # Forward pass
    semantic_logits = segmentation_head(feat)
    print("semantic_logits: ", semantic_logits.shape)