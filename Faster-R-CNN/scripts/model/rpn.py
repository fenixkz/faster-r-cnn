'''
A script that initializes the Region Proposal Network (RPN)
'''


import torch
import torch.nn as nn

class RPNHead(nn.Module):
    """
    Region Proposal Network Head.

    Takes features from the backbone and predicts objectness scores
    and bounding box regression deltas for a set of anchors at each
    spatial location.

    Args:
        in_channels (int): Number of channels in the input feature map.
        num_anchors (int): Number of anchors per spatial location.
        intermediate_channels (int): Number of channels in the intermediate layer (optional, defaults based on paper).
    """
    def __init__(self, in_channels: int, num_anchors: int, intermediate_channels: int = 512):
        super().__init__()
        self.num_anchors = num_anchors

        self.conv_3x3 = nn.Conv2d(in_channels=in_channels,
                                  out_channels=intermediate_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Classfication head (objectness score) - predicts 2 scores per anchor (bg/fg)
        self.cls = nn.Conv2d(in_channels=intermediate_channels,
                             out_channels=num_anchors*2, # each anchor having 2 logits for bg/fg classification
                             kernel_size=1,
                             stride=1,
                             padding=0)
        # Bounding box regression head - predicts 4 real values per anchor, i.e. deltas that map proposal box to true bounding box
        self.regr = nn.Conv2d(in_channels=intermediate_channels,
                             out_channels=num_anchors*4, # each anchor having 2 logits for bg/fg classification
                             kernel_size=1,
                             stride=1,
                             padding=0)
        
        # Initialize weights (as recommended in Faster R-CNN paper)
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialization heuristic from Faster R-CNN paper
        # Initialize 3x3 and heads with 0-mean Gaussian, std 0.01
        for layer in [self.conv_3x3, self.cls, self.regr]:
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features: torch.Tensor):
        """
        Args:
            features (Tensor): Input feature map from the backbone, shape (B, C, H, W).

        Returns:
            Tuple[Tensor, Tensor]:
                - rpn_cls_logits (Tensor): Classification logits, shape (B, num_anchors * 2, H, W).
                - rpn_bbox_pred (Tensor): Bbox regression predictions, shape (B, num_anchors * 4, H, W).
        """
        # Pass through 3x3 convolution and ReLU
        t = self.relu(self.conv_3x3(features)) # (B, intermediate_channels, H, W)

        # Get classification logits
        rpn_cls_logits = self.cls(t) # (B, num_anchors * 2, H, W)

        # Get bounding box regression predictions
        rpn_bbox_pred = self.regr(t) # (B, num_anchors * 4, H, W)

        return rpn_cls_logits, rpn_bbox_pred
    
class RPNLoss(nn.Module):
    """
    Loss that combines classification and regression head of RPN
    """

    def __init__(self, regr_lambda = 10) -> None:
        super().__init__()
        # Cross Entropy for classification with mean reduction
        self.cls_loss_func = nn.CrossEntropyLoss(reduction='mean')
        # Smooth L1 Loss for regression with sum reduction (summing over all elements in the batch)
        self.reg_loss_func = nn.SmoothL1Loss(reduction='sum')
        # A weight for regression
        self.lambda_ = regr_lambda

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, regr_deltas: torch.Tensor, target_deltas: torch.Tensor):
        """
        Args:
            logits (Tensor): Predicted class logits, shape (b, 2).
            labels (Tensor): Ground truth labels, shape (b,).
            regr_deltas (Tensor): Predicted regression deltas, shape (b, 4).
            target_deltas (Tensor): Ground truth regression deltas, shape (b, 4).

        Returns:
            Tensor: Total loss, a scalar.
        """
        # --- Classification Loss ---
        cls_loss = self.cls_loss_func(logits, labels) 

        # --- Regression Loss ---
        # Find positive samples in the mini-batch
        positive_idxs = torch.where(labels == 1)[0]
        num_positives = positive_idxs.numel()
        if num_positives > 0:
            reg_loss = self.reg_loss_func(regr_deltas, target_deltas) / num_positives
        else:
            # If no positive samples, regression loss is 0
            reg_loss = torch.tensor(0.0, device=regr_deltas.device, dtype=regr_deltas.dtype)

        # --- Total Loss ---
        total_loss = cls_loss + self.lambda_ * reg_loss
        return total_loss 
