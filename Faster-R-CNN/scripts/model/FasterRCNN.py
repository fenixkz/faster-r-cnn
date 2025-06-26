import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import BackBone
from torchvision.ops import RoIAlign, nms
from typing import Tuple
from model.rpn import RPNHead, RPNLoss
from utils.anchors import generate_anchors, label_anchors, label_proposals
import numpy as np


class FasterRCNN(nn.Module):

    def __init__(self, image_width: int, image_height: int, resnet_depth: int, num_classes: int, device: str, fc_hidden_dim: int = 1024, anchors_mini_batch_size: int = 256, dropout: float = 0.1):
        super(FasterRCNN, self).__init__()
        self.device = device
        # Scales for anchor generaration (area)
        scales = [128, 256, 512]
        # Aspect ratios for anchor generation (w:h ratio)
        aspect_ratios = [0.5, 1.0, 1.5]
        # Generate anchors given the params
        anchors = generate_anchors(image_height=image_height, image_width=image_width, stride=32, scales=scales, aspect_ratios=aspect_ratios) 
        # A boolean mask to filter anchors that out of image boundaries
        self.valid_idxs = np.where(
                    (anchors[:, 0] >= 0) & (anchors[:, 1] >= 0) & (anchors[:, 2] < image_width) & (anchors[:, 3] < image_height))[0]
        # Keep only valid anchors, total number is N_A
        self.anchors = torch.from_numpy(anchors[self.valid_idxs]).to(device) 
        # Total number of classes, for face detection = 2
        self.num_classes = num_classes
        # Backbone (feature extractor) ResNet 
        self.backbone = BackBone(resnet_depth).to(device)
        # Pass a dummy input to get the number of channels
        dummy_input = torch.randn(1, 3, image_height, image_width).to(device)
        output = self.backbone(dummy_input)
        print(f"Backbone output feature map shape: {output.shape}")
        stride = image_height / output.shape[2]
        print(f"Calculated stride: {stride}")
        # Region Proposal Network Head, given the feature map from backbone computes classification of whether the object is present in each anchor and 4 deltas that adjust the anchor to true bounding box
        self.rpn = RPNHead(in_channels=output.shape[1], num_anchors=len(scales) * len(aspect_ratios)).to(device)
        # Losses that combine classification and regression losses
        self.rpn_loss = RPNLoss(regr_lambda=1)
        self.faster_loss = RPNLoss(regr_lambda=1)
        # ROI Align layer, writing one from scratch is a good exercise but it will be many times slower than torchvision implementation
        self.roi_align = RoIAlign(
            output_size=(7, 7),
            spatial_scale=1.0/stride,
            sampling_ratio=2  # Number of sampling points per bin (default=2)
        )
        # Dropout to fight overfitting
        self.dropout = nn.Dropout(p=dropout) 
        # First linear layers, input is flattened channels per 7x7 roi windows
        self.fc1 = nn.Linear(output.shape[1] * 7 * 7, fc_hidden_dim)  # 7x7 is ROI pooled size
        self.fc2 = nn.Linear(fc_hidden_dim, fc_hidden_dim)
        # Classification head
        self.cls_head = nn.Linear(fc_hidden_dim, num_classes)
        # Regression head
        self.bbox_head= nn.Linear(fc_hidden_dim, 4 * (num_classes - 1))
        self.relu = nn.ReLU()
        # Mini-batch of anchors 
        self.anchors_mini_batch_size = anchors_mini_batch_size
        self.min_size = 16  # Minimum box size threshold as in the paper
        self.face_confidence_threshold = 0.8  # Threshold for face detection
        self.nms_iou_threshold = 0.7  # IoU threshold for NMS
        self.top_k = 500
        self.image_width = image_width
        self.image_height = image_height

    def _compute_target_deltas(self, source_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Finds a set of 4 deltas (transformations) that map source_tensor into target_tensor. Namely:
        - tx = (Gcx - Pcx) / Pw
        - ty = (Gcy - Pcy) / Ph
        - tw = log(Gw / Pw)
        - th = log(Gh / Ph)

        Args:
            source_tensor (Tensor): Source tensor, shape (N, 4).
            target_tensor (Tensor): Target tensor, shape (N, 4).

        Returns:
            Tensor: Target deltas, shape (N, 4).
        """
        with torch.no_grad():
            target_deltas = torch.full_like(target_tensor, 0)
            eps = torch.finfo(torch.float32).eps
            Px1 = source_tensor[:, 0]
            Py1 = source_tensor[:, 1]
            Px2 = source_tensor[:, 2]
            Py2 = source_tensor[:, 3]
            Pcx = (Px1 + Px2) / 2
            Pcy = (Py1 + Py2) / 2
            Pw = (Px2 - Px1).clamp(min=eps)
            Ph = (Py2 - Py1).clamp(min=eps)

            Gx1 = target_tensor[:, 0]
            Gy1 = target_tensor[:, 1]
            Gx2 = target_tensor[:, 2]
            Gy2 = target_tensor[:, 3]
            Gcx = (Gx1 + Gx2) / 2
            Gcy = (Gy1 + Gy2) / 2
            Gw = (Gx2 - Gx1).clamp(min=eps)
            Gh = (Gy2 - Gy1).clamp(min=eps)

            target_deltas[:, 0] = (Gcx - Pcx) / Pw
            target_deltas[:, 1] = (Gcy - Pcy) / Ph
            target_deltas[:, 2] = torch.log(Gw / Pw)
            target_deltas[:, 3] = torch.log(Gh / Ph)
        return target_deltas
    
    def _generate_mini_batch(self, labels: torch.Tensor, ratio: float):
        """
        Samples anchor indices for RPN loss calculation across a batch.

        Args:
            labels (Tensor): Assigned labels (0=neg, 1=pos, -1=ignore) for all anchors
                            across the batch. Shape: (B, N_A).
            ratio (float): Target fraction of positive samples.

        Returns:
            Tuple[Tensor, Tensor]:
                - sampled_batch_indices (Tensor): Batch indices of sampled anchors. Shape (N_sampled,).
                - sampled_anchor_indices (Tensor): Anchor indices (within N_A) of sampled anchors. Shape (N_sampled,).
                                                These two tensors form coordinate pairs (b, a).
        """
        B, N_A = labels.shape
        device = labels.device

        # 1. Find all positive and negative indices across the batch
        pos_indices_tuple = torch.where(labels == 1) # (pos_b, pos_a)
        neg_indices_tuple = torch.where(labels == 0) # (neg_b, neg_a)

        num_pos_available = pos_indices_tuple[0].numel()
        num_neg_available = neg_indices_tuple[0].numel()

        # 2. Determine target number of positives and negatives
        target_num_pos = min(int(self.anchors_mini_batch_size * ratio), num_pos_available)
        # Ensure total doesn't exceed desired, adjust negatives
        target_num_neg = min(self.anchors_mini_batch_size - target_num_pos, num_neg_available)


        # 3. Sample positive indices randomly
        sampled_pos_b = torch.empty((0,), dtype=torch.long, device=device)
        sampled_pos_a = torch.empty((0,), dtype=torch.long, device=device)
        if num_pos_available > 0 and target_num_pos > 0:
            pos_perm = torch.randperm(num_pos_available, device=device)[:target_num_pos]
            sampled_pos_b = pos_indices_tuple[0][pos_perm]
            sampled_pos_a = pos_indices_tuple[1][pos_perm]

        # 4. Sample negative indices randomly
        sampled_neg_b = torch.empty((0,), dtype=torch.long, device=device)
        sampled_neg_a = torch.empty((0,), dtype=torch.long, device=device)
        if num_neg_available > 0 and target_num_neg > 0:
            neg_perm = torch.randperm(num_neg_available, device=device)[:target_num_neg]
            sampled_neg_b = neg_indices_tuple[0][neg_perm]
            sampled_neg_a = neg_indices_tuple[1][neg_perm]

        # 5. Combine sampled indices
        sampled_batch_indices = torch.cat([sampled_pos_b, sampled_neg_b])
        sampled_anchor_indices = torch.cat([sampled_pos_a, sampled_neg_a])

        return sampled_batch_indices, sampled_anchor_indices

    def apply_deltas_(self, source_tensor: torch.Tensor, deltas: torch.Tensor):
        """
        Applies the deltas to transform source_tensor. Namely performs these 4 operations:
        - Gcx = Pw*tx + Pcx
        - Gcy = Ph*ty + Pcy
        - Gw = e^{tw}*Pw
        - Gh = e^{th}*Ph

        Args:
            source_tensor (Tensor): Tensor to be transformed, shape (N, 4)
            deltas (Tensor): Tensor of delta transformations, shape either (B, N, 4) or (N, 4)
        
        Returns:
            Tensor: A target tensor, i.e. a transformed by deltas source tensor, shape (N, 4)
        """
        if len(deltas.shape) == 2:
            deltas = deltas.unsqueeze(0) # Add a batch dimension
        eps = torch.finfo(deltas.dtype).eps
        p_x1 = source_tensor[:, 0]
        p_y1 = source_tensor[:, 1]
        p_x2 = source_tensor[:, 2]
        p_y2 = source_tensor[:, 3]

        p_w = (p_x2 - p_x1).clamp(min=eps)
        p_h = (p_y2 - p_y1).clamp(min=eps)
        p_cx = p_x1 + 0.5 * p_w
        p_cy = p_y1 + 0.5 * p_h

        tx = deltas[:, :, 0]
        ty = deltas[:, :, 1]
        tw = deltas[:, :, 2]
        th = deltas[:, :, 3]
        # Ensure exponentiation doesn't lead to NaN/inf for extreme values. Use 5 as an arbitrary high value
        tw = torch.clamp(tw, max=5) # Clamp log-width delta
        th = torch.clamp(th, max=5) # Clamp log-height delta

        G_cx = tx * p_w + p_cx
        G_cy = ty * p_h + p_cy
        G_w = torch.exp(tw) * p_w
        G_h = torch.exp(th) * p_h

        G_x1 = G_cx - 0.5 * G_w
        G_y1 = G_cy - 0.5 * G_h
        G_x2 = G_cx + 0.5 * G_w
        G_y2 = G_cy + 0.5 * G_h

        target_tensor = torch.stack((G_x1, G_y1, G_x2, G_y2), dim=2)
        return target_tensor

    def prepare_target_pred_deltas(self, labels: torch.Tensor, anchors: torch.Tensor, bbox_coords: torch.Tensor, pred_deltas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes target deltas and prediction deltas to calculate the regression loss. 
        First it selects only positive anchors and then computes tx, ty, tw, th for each positive anchor to match the ground truth boxes.

        Args:
            labels (Tensor): Labels, shape (N_A,).
            anchors (Tensor): Anchor boxes, shape (N_A, 4).
            bbox_coords (Tensor): Ground truth boxes, shape (N_A, 4).

        Returns:
            Tensor: Target deltas, shape (N, 4). Where N is a subset of N_A, only positive examples.
            Tensor: Prediction deltas, shape (N, 4). Where N is a subset of N_A, only positive examples.
        """
        positive_idxs = torch.where(labels == 1)[0]       # Get indexes of all anchors that are positive
        positive_anchors = anchors[positive_idxs]         # Get all anchors that are positive
        positive_bbox_coords = bbox_coords[positive_idxs] # Get all bboxes that are positive
        
        target_deltas = self._compute_target_deltas(positive_anchors, positive_bbox_coords) # Compute deltas for all positive anchors
        return target_deltas, pred_deltas[positive_idxs]

    def forward_rpn_pass(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (Tensor): Input feature map from the backbone, shape (B, C, H, W).

        Returns:
            Tuple[Tensor, Tensor]:
                - rpn_cls_scores (Tensor): Classification scores, shape (B, N, 2). Where N is number of valid anchors
                - rpn_pred_deltas (Tensor): Regression deltas, shape (B, N, 4). Where N is number of valid anchors
        """
        batch_size = x.shape[0]
        cls_scores, deltas = self.rpn(x) # Two outputs from two RPN heads. cls_scores is (B, k*2, 20, 20) and box_regression is (B, k*4, 20, 20)
        cls_scores = cls_scores.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2) # (B, N_A, 2)
        deltas = deltas.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)         # (B, N_A, 4)
        return cls_scores[:, torch.from_numpy(self.valid_idxs).long(), :], deltas[:, torch.from_numpy(self.valid_idxs).long(), :]

    def prepare_proposals(self, logits: torch.Tensor, deltas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given the anchors and predicted deltas from RPN, it does the following:
            1. Converts the anchors into proposals boxes with 4 coordinates
            2. Filters too small proposals
            3. Applies Non-Maximum Supression (NMS) to filter highly-overlapping proposals
            4. Pads the proposals if not enough for top-K filtering
            5. Selects top-K proposals 

        Args:
            logits (Tensor): Predicted class logits, shape (B, N, 2).
            deltas (Tensor): Predicted deltas, shape (B, N, 4).

        Returns:
            Tensor: Proposals, shape (B, K, 5). Where K is self.top_k. Each proposal is having the first dimension the batch index
            Tensor: Mask of valid proposals, shape (B, K). Where K is self.top_K
        """
        batch_size = logits.shape[0]

        # Get a confidence score that each anchor is a face
        scores = F.softmax(logits, dim = 2)[:, :, 1]    # (B, N)

        # Convert predictions into boxes with real coordinates
        proposals = self.apply_deltas_(self.anchors, deltas)

        # Initialize lists to store data
        final_proposals_list = []
        mask_list = []

        for i in range(batch_size):
            proposals_per_batch = proposals[i].clone() 
            scores_per_batch = scores[i]
            
            # 1. Clip proposals to image boundaries
            proposals_per_batch[:, 0] = torch.clamp(proposals_per_batch[:, 0], min=0, max=self.image_width)
            proposals_per_batch[:, 1] = torch.clamp(proposals_per_batch[:, 1], min=0, max=self.image_height)
            proposals_per_batch[:, 2] = torch.clamp(proposals_per_batch[:, 2], min=0, max=self.image_width)
            proposals_per_batch[:, 3] = torch.clamp(proposals_per_batch[:, 3], min=0, max=self.image_height)
            
            # 2. Filter out small boxes
            widths = proposals_per_batch[:, 2] - proposals_per_batch[:, 0]
            heights = proposals_per_batch[:, 3] - proposals_per_batch[:, 1]
            # A boolean mask 
            keep_size = (widths >= self.min_size) & (heights >= self.min_size)
            # Keep only big enough proposals
            proposals_per_batch = proposals_per_batch[keep_size]
            scores_per_batch = scores_per_batch[keep_size]

            # 3. If there aren't any proposals left, then add an empty tensor
            if proposals_per_batch.shape[0] == 0:
                final_proposals_list.append(torch.zeros((0, 5), dtype=proposals_per_batch.dtype, device=self.device))
                continue

            # 4. Apply NMS to get rid of highly overlapping proposals
            keep_idxs = nms(proposals_per_batch, scores_per_batch, self.nms_iou_threshold)
            proposals_per_batch = proposals_per_batch[keep_idxs]
            scores_per_batch = scores_per_batch[keep_idxs]

            # 5. Sort and select top-K proposals
            _, sorted_indices = torch.sort(scores_per_batch, descending=True)
            top_k = min(self.top_k, sorted_indices.size(0))
            top_indices = sorted_indices[:top_k]
            proposals_per_batch = proposals_per_batch[top_indices] 
            mask = torch.ones((self.top_k,), dtype=torch.bool, device = self.device)  
            # If we don't have K proposals, we pad
            if top_k < self.top_k:
                padded_boxes = torch.zeros((self.top_k, proposals_per_batch.size(1)), dtype=proposals_per_batch.dtype, device=self.device)
                padded_boxes[:top_k] = proposals_per_batch
                mask[top_k:] = 0
                proposals_per_batch = padded_boxes
            
            # 6. Add batch index to proposals
            batch_idx_tensor = torch.full((proposals_per_batch.shape[0], 1), i, dtype=proposals_per_batch.dtype, device=self.device)
            proposals_per_batch = torch.cat((batch_idx_tensor, proposals_per_batch), dim=1)

            # 7. Append to lists
            final_proposals_list.append(proposals_per_batch)
            mask_list.append(mask)
        
        proposals = torch.stack(final_proposals_list) # Shape: (B, top_k, 5)
        masks = torch.stack(mask_list) # Shape (B, top_k)
        return proposals, masks
    
    def forward(self, images: torch.Tensor, gt_bboxes: torch.Tensor = None, gt_masks: torch.Tensor = None):
        batch_size = images.shape[0] # Get the batch size
        
        # Backbone pass
        x = self.backbone(images) # Extract feature map, the shape is (B, C, W/S, H/s) given the input image shape (B, 3, 640, 640) and our params -> (B, 512, 20, 20)

        ##########################
        #######  RPN part ########
        ##########################

        rpn_logits, rpn_deltas = self.forward_rpn_pass(x) # Two outputs: classification logits (B, N_A, 2) and regression deltas (B, N_A, 4) where N_A is the number of anchors
        
        if self.training:
            # Use IoU thresholding to label each anchor with either 1, 0 or -1, also get the indexes of the bboxes that are best matched to each anchor
            rpn_labels, rpn_bbox_idxs = label_anchors(self.anchors, gt_bboxes, gt_masks, upper_thresh=0.5, lower_thresh=0.3) 
            
            # Get a tensor with the coordinates of bboxes that are matched with each anchor
            batch_indices = torch.arange(batch_size, device=self.device).view(batch_size, 1).expand_as(rpn_bbox_idxs) # Shape (B, N_A)
            # Get actual coordinates of each bbox that has a highest match with each anchor in the tensor
            bbox_coords = gt_bboxes[batch_indices, rpn_bbox_idxs] # Shape (B, N_valid_a, 4)
            
            # Create a mini-batch of positive:negative examples from all images in the batch
            # mb_b -- indexes of batch dim
            # mb_a -- indexes of anchor dim
            mb_b, mb_a = self._generate_mini_batch(rpn_labels, ratio=0.5) # Sample a mini-batch of anchors to train on
            
            mb_anchors = self.anchors[mb_a]            # Anchors in the mini-batch, shape: (N_valid_A, 4)
            mb_logits = rpn_logits[mb_b, mb_a, :]      # Prediction of either object or not for all anchors in the mini-batch, shape: (N_valid_A, 2)
            mb_labels = rpn_labels[mb_b, mb_a]         # Labels computed via IoU for all anchors, shape: (N_valid_A,)
            mb_pred_deltas = rpn_deltas[mb_b, mb_a, :] # Prediction of deltas to tailor the anchor coordinates, shape: (N_valid_A, 4)
            mb_bbox_coords = bbox_coords[mb_b, mb_a]   # BBox coordinates that have highest IoU with each anchor in the mini-batch, shape: (N_valid_A, 4)
            
            # Compute target for regression loss
            target_deltas, pred_deltas = self.prepare_target_pred_deltas(mb_labels, mb_anchors, mb_bbox_coords, mb_pred_deltas)
            # print("RPN target labels:", torch.unique(mb_labels, return_counts=True))
            # print("RPN target deltas (sample):", target_deltas[0])
            # print("RPN pred deltas (sample):", pred_deltas[0])
            # RPN LOSS
            rpn_loss = self.rpn_loss(mb_logits, mb_labels, pred_deltas, target_deltas)
            
        #################################
        #######  END of RPN part ########
        #################################
        
        ########################################
        ####### Prepare Valid Proposals ########
        ########################################
        
            with torch.no_grad():
                proposals, mask = self.prepare_proposals(rpn_logits, rpn_deltas) # (B, K, 5) and (B, K)
            
            # Label each proposal with IoU
            labels, matched_gt_idxs = label_proposals(proposals, mask, gt_bboxes, gt_masks)
            # Same routine to get the coordinates of actual bboxes that each anchor has highest match with
            batch_indices = torch.arange(batch_size, device=self.device).view(batch_size, 1).expand_as(matched_gt_idxs)
            bbox_coords = gt_bboxes[batch_indices, matched_gt_idxs] 
            # Sample a mini-batch of proposals to train on
            mb_b_idxs, mb_a_idxs = self._generate_mini_batch(labels, 0.25) 
            proposals = proposals[mb_b_idxs, mb_a_idxs]
            labels = labels[mb_b_idxs, mb_a_idxs]
            bbox_coords = bbox_coords[mb_b_idxs, mb_a_idxs]
        else:
            with torch.no_grad():
                proposals, mask = self.prepare_proposals(rpn_logits, rpn_deltas) # (B, K, 5) and (B, K)
            # Reshape to (B*K, 5)
            proposals = proposals.view(batch_size * self.top_k, 5)
            mask = mask.view(batch_size * self.top_k)
            proposals = proposals[mask]

        ###############################################
        ####### End of Prepare Valid Proposals ########
        ###############################################

        ###############################
        ####### Fast R-CNN Head #######
        ###############################
        
        # Apply ROI Align
        x = self.roi_align(x, proposals) 
        x = x.view(x.size(0), -1)
        # Pass through fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        # Get a classification logits
        cls_score = self.cls_head(x)
        # Get prediction deltas
        bbox_pred = self.bbox_head(x)

        if self.training:
            target_deltas, pred_deltas = self.prepare_target_pred_deltas(labels, proposals[:, 1:], bbox_coords, bbox_pred)
            # print("Fast R-CNN target labels:", torch.unique(labels, return_counts=True))
            loss = self.faster_loss(cls_score, labels, pred_deltas, target_deltas) + rpn_loss
            return loss
        else:
            # Okay, now we should post-process the values
            # Apply a threshold for deciding which boxes to keep
            cls_score = torch.softmax(cls_score, dim=1)
            # Apply the threshold
            keep_idxs = cls_score[:, 1] > self.face_confidence_threshold
            cls_score = cls_score[keep_idxs]
            bbox_pred = bbox_pred[keep_idxs]
            proposals = proposals[keep_idxs]

            if len(bbox_pred) == 0:
                # Return empty predictions for all images in batch
                return [{"boxes": torch.zeros((0, 4), device=self.device), 
                        "scores": torch.zeros(0, device=self.device), 
                        "labels": torch.zeros(0, device=self.device)} for _ in range(batch_size)]
            # Inverse transform the bbox_pred to coordinates
            with torch.no_grad(): # Post-processing doesn't need gradients
                source_boxes = proposals[:, 1:] # Shape (N_kept, 4)
                deltas = bbox_pred              # Shape (N_kept, 4)
                proposal_coords = self.apply_deltas_(source_boxes, deltas)
                # Squeeze the batch dimension since apply_deltas_ returns (1, N, 4) but nms expects (N, 4)
                proposal_coords = proposal_coords.squeeze(0)
            # Apply non-maximum suppression
            keep_idxs = nms(proposal_coords, cls_score[:, 1], 0.3)
            cls_score = cls_score[keep_idxs]
            bbox_coords = proposal_coords[keep_idxs]
            batch_indices = proposals[keep_idxs, 0].long()
            # Create a list of dictionaries, one for each image
            results = []
            for i in range(batch_size):
                # Get indices for this batch
                batch_mask = batch_indices == i
                # Create dictionary for this image
                image_result = {
                    "boxes": bbox_coords[batch_mask],
                    "scores": cls_score[batch_mask, 1],  # Use class 1 scores (face)
                    "labels": torch.ones(batch_mask.sum(), dtype=torch.int64, device=self.device)
                }
                results.append(image_result)
            
            return results