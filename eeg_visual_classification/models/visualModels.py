import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from torchvision.models import resnet18
from self_attention_cv import TransformerEncoder


class ViT(nn.Module):
    def __init__(
        self,
        img_dim,
        in_channels=1,
        patch_dim=16,
        num_classes=5,
        dim=512,
        blocks=6,
        heads=4,
        dim_linear_block=1024,
        dim_head=None,
        dropout=0,
        transformer=None,
        classification=True,
        classification_structure=True,
    ):
        """
        Args:
            img_dim: the spatial image size
            in_channels: number of img channels
            patch_dim: desired patch dim
            num_classes: classification task classes
            dim: the linear layer's dim to project the patches for MHSA
            blocks: number of transformer blocks
            heads: number of heads
            dim_linear_block: inner dim of the transformer linear block
            dim_head: dim head in case you want to define it. defaults to dim/heads
            dropout: for pos emb and transformer
            transformer: in case you want to provide another transformer implementation
            classification: creates an extra CLS token
        """
        super().__init__()
        assert img_dim % patch_dim == 0, f"patch size {patch_dim} not divisible"
        self.p = patch_dim
        self.classification = classification
        self.classification_structure = classification_structure
        tokens = (img_dim // patch_dim) ** 2
        self.token_dim = in_channels * (patch_dim**2)
        self.dim = dim
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self.project_patches = nn.Linear(self.token_dim, dim)

        self.emb_dropout = nn.Dropout(dropout)
        if self.classification or self.classification_structure:
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
            self.pos_emb1D = nn.Parameter(torch.randn(tokens + 1, dim))
            self.mlp_head = nn.Linear(dim, num_classes)
        else:
            self.pos_emb1D = nn.Parameter(torch.randn(tokens, dim))

        if transformer is None:
            self.transformer = TransformerEncoder(
                dim,
                blocks=blocks,
                heads=heads,
                dim_head=self.dim_head,
                dim_linear_block=dim_linear_block,
                dropout=dropout,
            )
        else:
            self.transformer = transformer

    def expand_cls_to_batch(self, batch):
        """
        Args:
            batch: batch size
        Returns: cls token expanded to the batch size
        """
        return self.cls_token.expand([batch, -1, -1])

    def forward(self, img, mask=None):
        batch_size = img.shape[0]
        # print("this batch_size: ", batch_size)
        # print("image shape : ", img.shape)
        img_patches = rearrange(
            img,
            "b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)",
            patch_x=self.p,
            patch_y=self.p,
        )
        # print("image patches : ", img_patches.shape)
        # project patches with linear layer + add pos emb
        img_patches = self.project_patches(img_patches)

        if self.classification or self.classification_structure:
            img_patches = torch.cat(
                (self.expand_cls_to_batch(batch_size), img_patches), dim=1
            )

        patch_embeddings = self.emb_dropout(img_patches + self.pos_emb1D)
        # print(f'patch embeddings shape: {patch_embeddings.shape}')
        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.transformer(patch_embeddings, mask)
        # print(f'y shape: {y.shape}')

        if self.classification:
            # we index only the cls token for classification. nlp tricks :P
            return self.mlp_head(y[:, 0, :])
        else:
            return y


class SimpleEEGCNN(nn.Module):
    def __init__(self, num_classes=40):
        super(SimpleEEGCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
        )

        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))

    def forward_features(self, x):
        x = self.features(x)
        return nn.Flatten()(x)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResizeTo64(nn.Module):
    """
    Resizes spectrogram-like tensors to (B, C, 64, 64).

    Accepts:
        x of shape (B, C, F, T) or (C, F, T)  # will add batch dim if missing

    Args:
        mode: Interpolation mode: 'bilinear' (default), 'bicubic', or 'nearest'
        align_corners: Only relevant for bilinear/bicubic (default False)
    """

    def __init__(self, mode: str = "bilinear", align_corners: bool = False):
        super().__init__()
        if mode not in ("bilinear", "bicubic", "nearest"):
            raise ValueError(
                f"Unsupported mode '{mode}'. Choose from 'bilinear', 'bicubic', 'nearest'."
            )
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (C, F, T) by promoting to batch of 1
        squeeze_batch = False
        if x.ndim == 3:  # (C, F, T)
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.ndim != 4:
            raise ValueError(f"Expected (B,C,F,T) or (C,F,T), got {tuple(x.shape)}")

        # (B, C, F, T) -> resize to (B, C, 64, 64)
        x_resized = F.interpolate(
            x,
            size=(64, 64),
            mode=self.mode,
            align_corners=(
                self.align_corners if self.mode in ("bilinear", "bicubic") else None
            ),
            antialias=True if self.mode in ("bilinear", "bicubic") else False,
        )

        if squeeze_batch:
            x_resized = x_resized.squeeze(0)
        return x_resized


# Resnet-18 wrapper model
class ResNet18Wrapper(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.resnet = resnet18()
        self.resnet.conv1 = nn.Conv2d(
            128, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)
