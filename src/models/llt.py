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
        depth: int = 2, #encoder layers 
        num_heads: int = 4, #attention heads
        mlp_ratio: float = 2.0,
        num_classes: int = 176, #reference points 
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
             raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
                        )    
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
        hidden_dim = int(embed_dim * mlp_ratio)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=depth,
            )
        
        self.norm = nn.LayerNorm(embed_dim)
        #from vector of embed_dim to vector of num_classes (logits)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        self._initialize_weights()

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = num_patches

    def _initialize_weights(self) -> None:
            nn.init.trunc_normal_(self.class_token, std=0.02)
            nn.init.trunc_normal_(self.position_embedding, std=0.02)

            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

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
        
        encoded_tokens = self.encoder(tokens)
        encoded_tokens = self.norm(encoded_tokens)

        cls_features = encoded_tokens[:, 0]

        logits = self.classifier(cls_features)
        return logits

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )