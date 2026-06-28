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

class LLT(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        image_size: tuple[int, int] = (30, 30),
        patch_size: tuple[int, int] = (5, 5),
        embed_dim: int = 32,
        dropout: float = 0.1,
        num_classes: int = 176, #reference points 
    ) -> None:
        super().__init__()

        # use the prev. defined CSIPatchEmbedding to get patch tokens 
        self.patch_embedding = CSIPatchEmbedding(
            in_channels=in_channels,
            image_size=image_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embedding.num_patches

        #build a trainable parameter: class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        #build a trainable parameter: position embedding
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.position_dropout = nn.Dropout(dropout)

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # call csipatchembedding to get patch tokens
        tokens = self.patch_embedding(x)

        batch_size = tokens.shape[0]

        #[1,1,32]→[batch_size,1,32]
        class_tokens = self.class_token.expand(batch_size, -1, -1)

        #concat on token dimension (dim=1) to get  num_patches+1
        tokens = torch.cat((class_tokens, tokens), dim=1)

        # we sum tokens + position_embedding 
        tokens = tokens + self.position_embedding

        tokens = self.position_dropout(tokens)

        return tokens