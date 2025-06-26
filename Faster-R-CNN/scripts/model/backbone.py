'''
A script that initializes the Backbone class (feature extractor)
'''

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Type, Union

# Define a type hint for the model function for clarity
ModelFn = Type[Union[models.ResNet]]

class BackBone(nn.Module):
    """
    A flexible backbone feature extractor using torchvision's ResNet models.
    """
    # Use a dictionary to map depth to the correct model function and weights enum
    LOOKUP: Dict[int, Dict[str, Union[ModelFn, Type[models.WeightsEnum]]]] = {
        18: {"model": models.resnet18, "weights": models.ResNet18_Weights.DEFAULT},
        34: {"model": models.resnet34, "weights": models.ResNet34_Weights.DEFAULT},
        50: {"model": models.resnet50, "weights": models.ResNet50_Weights.DEFAULT},
        101: {"model": models.resnet101, "weights": models.ResNet101_Weights.DEFAULT},
        152: {"model": models.resnet152, "weights": models.ResNet152_Weights.DEFAULT},
    }

    def __init__(self,
                 resnet_depth: int,
                 requires_grad: bool = False):
        """
        Initializes the ResNet backbone.

        Args:
            resnet_depth (int): The depth of the ResNet model (18, 34, 50, 101, 152).
            requires_grad (bool): If True, the backbone parameters will be trainable (for fine-tuning).
                                  If False, they will be frozen.
        """
        super().__init__()  # Proper way to initialize an nn.Module

        if resnet_depth not in self.LOOKUP:
            raise ValueError(
                f"Unsupported ResNet depth: {resnet_depth}. "
                f"Please use one of {sorted(list(self.LOOKUP.keys()))}"
            )

        # 1. Load the model using the lookup table
        model_info = self.LOOKUP[resnet_depth]
        weights = model_info["weights"] 
        resnet_model = model_info["model"](weights=weights)

        # 2. Extract all feature layers except for last 2 - AvgPool and FC
        feature_layers = list(resnet_model.children())[:-2]
        
        self.backbone = nn.Sequential(*feature_layers)

        # 3. Freeze parameters if required
        if not requires_grad:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Set to evaluation mode if not training
        if not requires_grad:
            self.backbone.eval()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass. 
        """
        return self.backbone(tensor)
    