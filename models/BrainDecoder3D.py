import torch
import torch.nn as nn
import torch.nn.functional as F
from models.meta.electrode_names import channels
from models.VisualTransforms import EEGScalpMap

class R2Plus1D_Block(nn.Module):
    def __init__(self, in_ch, out_ch, k_t=3, k_s=3, stride_t=1, stride_s=1, p_t=1, p_s=1):
        super().__init__()
        # Temporal then spatial
        self.temporal = nn.Conv3d(in_ch, out_ch, kernel_size=(k_t,1,1),
                                  stride=(stride_t,1,1), padding=(p_t,0,0), bias=False)
        self.spatial  = nn.Conv3d(out_ch, out_ch, kernel_size=(1,k_s,k_s),
                                  stride=(1,stride_s,stride_s), padding=(0,p_s,p_s), bias=False)
        self.bn = nn.BatchNorm3d(out_ch)
        self.act = nn.SiLU()
        # Residual if needed
        self.down = None
        if stride_t>1 or stride_s>1 or in_ch!=out_ch:
            self.down = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=(stride_t, stride_s, stride_s), bias=False),
                nn.BatchNorm3d(out_ch)
            )

    def forward(self, x):
        out = self.spatial(self.temporal(x))
        out = self.bn(out)
        if self.down is not None:
            x = self.down(x)
        return self.act(out + x)

class EEG3DNet(nn.Module):
    def __init__(self, n_classes=40, base=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, base, kernel_size=(3,3,3), padding=(1,1,1), bias=False),
            nn.BatchNorm3d(base), nn.SiLU()
        )
        # (T,H,W) downsampling schedule: temporal stride sparingly; spatial stride a bit more
        self.layer1 = R2Plus1D_Block(base,   base,   stride_t=1, stride_s=2)  # H,W: 32->16
        self.layer2 = R2Plus1D_Block(base,   base*2, stride_t=2, stride_s=2)  # T: /2, H,W: 16->8
        self.layer3 = R2Plus1D_Block(base*2, base*2, stride_t=1, stride_s=2)  # H,W: 8->4
        self.layer4 = R2Plus1D_Block(base*2, base*4, stride_t=2, stride_s=2)  # T: /2, H,W: 4->2
        self.head   = nn.Sequential(
            nn.AdaptiveAvgPool3d((1,1,1)), nn.Flatten(),
            nn.Linear(base*4, n_classes)
        )

    def forward(self, x):  # x: (B,1,T,32,32)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)
      
class Model(nn.Module):
    def __init__(self):
      super().__init__()
      self.map = EEGScalpMap("models/meta/map_v2.csv", electrodes=channels)
      self.model = EEG3DNet(n_classes=40)
      
    def forward(self, x):
      x = self.map(x.squeeze())  # (B, T, H, W)
      x = x.unsqueeze(1)  # (B, 1, T, H
      return self.model(x)