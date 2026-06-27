from __future__ import annotations

import torch
from torch import nn

#Convert a CSI tensor into a sequence of patch tokens
#[B,3,30,30]→[B,36,32] 
class CSIPatchEmbedding(nn.Module):
    

    def __init__(
        self,
        in_channels: int = 3,
        image_size: tuple[int, int] = (30, 30), #time packets x subcarriers
        patch_size: tuple[int, int] = (5, 5),  #dimension of each patch of the matrix (not-overlapping patches)
        embed_dim: int = 32,
    ) -> None:
        super().__init__()

        #unpack the image and patch tuples
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        #check if the patches fit into the matrix 
        if image_height % patch_height != 0:
            raise ValueError("image height must be divisible by patch height")

        if image_width % patch_width != 0:
            raise ValueError("image width must be divisible by patch width")


        self.image_size = image_size
        self.patch_size = patch_size

        #calculate : #patches in each dimension and #totalpatches
        self.num_patches_h = image_height // patch_height
        self.num_patches_w = image_width // patch_width
        self.num_patches = self.num_patches_h * self.num_patches_w

        # convolutional patch embedding layer 
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,  #patches not overlapping
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.projection(x)
        # [batch_size, embed_dim, num_patches_h, num_patches_w]

        x = x.flatten(2)
        # [batch_size, embed_dim, num_patches]

        x = x.transpose(1, 2)
        # [batch_size, num_patches, embed_dim] ready for transformer input

        return x