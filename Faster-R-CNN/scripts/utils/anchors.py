import numpy as np
import random
import pandas as pd
import random
import os
from typing import List, Dict, Any, Tuple
import torch

def generate_anchors(image_height, image_width, stride, scales, aspect_ratios):
    """
    Generates anchor boxes densely across an image based on feature map grid.

    Args:
        image_height (int): Height of the input image.
        image_width (int): Width of the input image.
        stride (int): Overall stride of the backbone network (e.g., 32 for VGG16).
        scales (list or np.array): Anchor scales (base side lengths like [128, 256, 512]).
        aspect_ratios (list or np.array): Anchor aspect ratios (e.g., [0.5, 1.0, 2.0] for 1:2, 1:1, 2:1).

    Returns:
        np.ndarray: Generated anchor boxes, shape (H' * W' * k, 4),
                    format [x1, y1, x2, y2] in image coordinates.
                    k = len(scales) * len(aspect_ratios).
                    H', W' are feature map dimensions.
    """
    scales = np.asarray(scales)
    aspect_ratios = np.asarray(aspect_ratios)
    num_anchor_types = len(scales) * len(aspect_ratios) # k

    # Calculate feature map size
    feature_height = image_height // stride
    feature_width = image_width // stride

    # --- Generate k base anchor shapes (widths and heights) ---
    # Area = scale^2
    # w * h = scale^2
    # w / h = ratio  => w = h * ratio
    # => (h * ratio) * h = scale^2 => h^2 = scale^2 / ratio => h = scale / sqrt(ratio)
    # => w = scale * sqrt(ratio)
    # Note: Transpose before flatten to get order: (scale1,r1), (scale1,r2), (scale1,r3), (scale2,r1)...
    base_anchor_heights = (scales / np.sqrt(aspect_ratios)[:, np.newaxis]).T.flatten() # Shape: (k,)
    base_anchor_widths = (scales * np.sqrt(aspect_ratios)[:, np.newaxis]).T.flatten() # Shape: (k,)


    # --- Generate grid centers (in image coordinates) ---
    shift_x = (np.arange(0, feature_width) + 0.5) * stride
    shift_y = (np.arange(0, feature_height) + 0.5) * stride
    # create meshgrid
    center_x, center_y = np.meshgrid(shift_x, shift_y) # Shapes: (H', W')

    # Flatten centers to vectors
    centers_x_flat = center_x.flatten() # Shape: (H'*W',)
    centers_y_flat = center_y.flatten() # Shape: (H'*W',)
    num_locations = len(centers_x_flat) # H' * W'

    # --- Combine centers and base anchor shapes ---
    # Expand centers to match the number of anchors (repeat each center k times)
    # Shape: (H'*W', 1) -> (H'*W', k) -> (H'*W'*k,)
    centers_x_expanded = np.repeat(centers_x_flat[:, np.newaxis], num_anchor_types, axis=1).flatten()
    centers_y_expanded = np.repeat(centers_y_flat[:, np.newaxis], num_anchor_types, axis=1).flatten()

    # Tile base anchor shapes to match the number of locations
    # Shape: (k,) -> (1, k) -> (H'*W', k) -> (H'*W'*k,)
    anchor_widths_tiled = np.tile(base_anchor_widths[np.newaxis, :], (num_locations, 1)).flatten()
    anchor_heights_tiled = np.tile(base_anchor_heights[np.newaxis, :], (num_locations, 1)).flatten()

    # --- Calculate anchor coordinates [x1, y1, x2, y2] ---
    # (Subtract/add half width/height from centers)
    x1 = centers_x_expanded - 0.5 * anchor_widths_tiled
    y1 = centers_y_expanded - 0.5 * anchor_heights_tiled
    x2 = centers_x_expanded + 0.5 * anchor_widths_tiled
    y2 = centers_y_expanded + 0.5 * anchor_heights_tiled

    # Stack coordinates together
    anchors = np.stack([x1, y1, x2, y2], axis=1) # Shape: (H'*W'*k, 4)

    return anchors.astype(np.float32)

def compute_iou(boxes: torch.Tensor, mask: torch.Tensor, gt_bboxes: torch.Tensor, gt_mask: torch.Tensor):
    """
    Computes IoU between a set of boxes and batches of padded GT boxes.
    
    Args:
        boxes (Tensor): Tensor of box coordinates of shape (B, N, 4) of format (x1, y1, x2, y2)
        mask (Tensor): Tensor of specifying a padded indexes of boxes, shape (B, N)
        gt_bboxes (Tensor): Padded ground truth boxes per image, shape (B, N_gt_max, 4).
        gt_mask (Tensor): Boolean mask indicating valid GT boxes, shape (B, N_gt_max).
        
    Returns:
        Tensor: IoU matrix. Shape (B, N_A, N_gt_max)
    """
    B, N_gt_max, _ = gt_bboxes.shape
    # A small epsilon for numerical stability (division by zero)
    eps = torch.finfo(gt_bboxes.dtype).eps
    
    # Expand shapes for broadcasting: (B, N_A, 1, 4) and (B, 1, N_gt_max, 4)
    boxes_expanded = boxes.unsqueeze(2)
    gt_boxes_expanded = gt_bboxes.unsqueeze(1)

    # Intersection coordinates (B, N_A, N_gt_max)
    x1 = torch.maximum(boxes_expanded[..., 0], gt_boxes_expanded[..., 0])
    y1 = torch.maximum(boxes_expanded[..., 1], gt_boxes_expanded[..., 1])
    x2 = torch.minimum(boxes_expanded[..., 2], gt_boxes_expanded[..., 2])
    y2 = torch.minimum(boxes_expanded[..., 3], gt_boxes_expanded[..., 3])

    # Intersection area (B, N_A, N_gt_max)
    width = (x2 - x1).clamp(min=0)
    height = (y2 - y1).clamp(min=0)
    intersection = width * height

    # Anchor areas (B, N_A) -> (B, N_A, 1)
    area_anchors = ((boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])).unsqueeze(2)
    area_anchors = area_anchors.clamp(min=eps)

    # GT box areas (B, N_gt_max) -> (B, 1, N_gt_max)
    area_gt_boxes = ((gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (gt_bboxes[..., 3] - gt_bboxes[..., 1])).unsqueeze(1)
    area_gt_boxes = area_gt_boxes.clamp(min=eps) # Avoid division by zero

    # Apply mask to GT areas for padding - prevents NaNs in union for padded boxes
    gt_mask_expanded = gt_mask.unsqueeze(1) # Shape (B, 1, N_gt_max)
    mask = mask.unsqueeze(2) # Shape (B, N, 1)

    # Ensure area_gt_boxes is float if mask is bool
    area_gt_boxes = area_gt_boxes * gt_mask_expanded.float() # Zero out areas for padded boxes
    area_anchors = area_anchors * mask.float() # Zero out areas for padded boxes 

    # Union (B, N_A, N_gt_max)
    union = area_anchors + area_gt_boxes - intersection
    union = union.clamp(min=eps) # Avoid division by zero

    # IoU (B, N_A, N_gt_max)
    iou = intersection / union
    combined_mask = mask & gt_mask_expanded
    # Mask out IoUs corresponding to padded GT boxes
    iou = iou * combined_mask.float()
    return iou

def label_anchors(anchors: torch.Tensor, gt_boxes: torch.Tensor, gt_mask: torch.Tensor, upper_thresh: float, lower_thresh: float):
    """ 
    Assigns labels (1=positive, 0=negative, -1=ignore) to anchors based on IoU with ground truth (GT) bounding boxes (BB). 

    Args:
        anchors: A tensor of coordinates of each anchor, format (x1, y1, x2, y2), shape (N_A, 4)
        gt_boxes: A tensor of coordinates of each GT BB, format (x1, y1, x2, y2), shape (B, N_GT, 4) 
        gt_mask: A boolean tensor specifying which elements of gt_boxes are padded (not real), shape (B, N_GT)

    Returns:
        anchor_labels: A tensor of labels for each anchor, shape (B, N_A)
        best_gt_per_anchor: A tensor of best matching bounding box for each anchor, shape (B, N_A)
    """
    # Get dimensions and device
    B, N_gt, _ = gt_boxes.shape
    N_A, _ = anchors.shape
    device = anchors.device

    # Add a batch dimension to anchor tensor
    anchors_batched = anchors.unsqueeze(0).expand(B, N_A, 4)

    # Create a boolean mask matching the expanded anchors (all True since anchors are not padded)
    mask = torch.ones((B, N_A), dtype=torch.bool, device=device)

    # Compute IoU matrix -- a matrix where each element is IoU of anchor_i (row) and bb_j (column) -- shape (B, N_A, N_GT)
    iou_matrix = compute_iou(anchors_batched, mask, gt_boxes, gt_mask) 

    # Set IoU for padded GT boxes to -1 so they don't influence max calculation negatively, use tensor(-1.0) to explicitly move it to needed device
    iou_matrix = torch.where(gt_mask.unsqueeze(1), iou_matrix, torch.tensor(-1.0, device=device, dtype=iou_matrix.dtype))

    # Find max IoU and corresponding GT index for each anchor
    max_iou_per_anchor, best_gt_per_anchor = iou_matrix.max(dim=2) # (B, N_A)
    
    # --- Assign based on max IoU per GT box (for ensuring each GT is matched) ---
    # This part remains tricky to fully vectorize efficiently without loops or complex ops

    # For each GT BB, find the anchor(s) with the highest IoU
    gt_mask_float = gt_mask.float() # Make it float, so it can be multiplied numerically

    # Find max IoU anchor for each GT
    max_iou_per_gt, _ = iou_matrix.max(dim=1) # (B, N_gt) 

    # Mask out max IoUs for padded GTs
    max_iou_per_gt = max_iou_per_gt * gt_mask_float

    # --- Initialize labels ---
    anchor_labels = torch.full((B, N_A), -1, dtype=torch.long, device=device) # Initially all set to -1, use torch.long dtype as a default for labels

    # --- Assign labels based on thresholds ---
    # Check if maximum IoU for each anchor is bigger/lower than the corresponding threshold
    anchor_labels[max_iou_per_anchor < lower_thresh] = 0 
    anchor_labels[max_iou_per_anchor >= upper_thresh] = 1 

    # --- Additionally assign 1 to anchors that were the best match for each GT bbox. In case the IoU is lower than the threshold  ---
    # Loop approach (less efficient but clear):
    for b in range(B):
        for g in range(N_gt):
            if gt_mask[b, g]: # If it's a real GT box (not padded)
                # Find which anchors have max IoU (or close) for this GT
                current_gt_max_iou = max_iou_per_gt[b, g]
                if current_gt_max_iou > 1e-8: # Check if any anchor matched this GT with a positive IoU
                    # Find the index of the anchor that had the best match with this GT bbox
                    best_anchor_for_gt = torch.where(iou_matrix[b, :, g] >= current_gt_max_iou - 1e-8)[0]
                    # Assing 1 to this anchor, it maybe 1 already then nothing changes
                    anchor_labels[b, best_anchor_for_gt] = 1 

    return anchor_labels, best_gt_per_anchor

def label_proposals(proposals: torch.Tensor, mask: torch.Tensor, gt_boxes: torch.Tensor, gt_mask: torch.Tensor):
    '''
    A function to label a tensor of proposals using IoU thresholding. 

    Args:
        proposals (Tensor): A tensor of proposals, shape: (B, K, 5) where K is the top-K proposals and last dimension equals to 5 to include the batch index
        mask (Tensor): A boolean mask to check which proposals are padded to keep the unified shape, shape: (B, K)
        gt_boxes (Tensor): A tensor of ground truth bounding boxes, shape: (B, N_GT, 4)
        gt_mask (Tensor): A boolean mask to check which bounding boxes are padded to keep the unified shape, shape: (B, N_GT)
    '''
    # Some constants, TODO: make them args to the Faster-R-CNN class
    FG_THR = 0.5 # Foreground threshold
    BG_HI_THR = 0.5 # Background upper threshold
    BG_LO_THR = 0.1 # Backgound lower threshold

    B, N_gt, _ = gt_boxes.shape
    _, K, _ = proposals.shape
    device = proposals.device

    # Compute the IoU matrix, don't use the batch indexing dimension, matrix shape: (B, K, N_GT)
    iou_matrix = compute_iou(proposals[..., 1:], mask, gt_boxes, gt_mask) 
    # Set -1.0 to padded boxes and proposals
    iou_matrix = torch.where(gt_mask.unsqueeze(1), iou_matrix, torch.tensor(-1.0, device=device, dtype=iou_matrix.dtype))
    iou_matrix = torch.where(mask.unsqueeze(2), iou_matrix, torch.tensor(-1.0, device=device, dtype=iou_matrix.dtype))
    # Get maximum IoU and the best matching bounding box per each proposal, both tensors are of shape (B, K)
    max_ious_per_proposal, best_gt_per_proposal = iou_matrix.max(dim=2) 

    # --- Initialize labels ---
    prop_labels = torch.full((B, K), -1, dtype=torch.long, device=device) # Set to -1.0 initially

    # --- Assign labels based on thresholds ---
    # A negative label is assigned to proposals that are having best IoU between BG_LO and BG_HI (quite confident background)
    prop_labels[(max_ious_per_proposal < BG_HI_THR) & (max_ious_per_proposal >= BG_LO_THR)] = 0 
    # A positive label is assigned to proposals with IoU higher than foreground threshold
    prop_labels[max_ious_per_proposal >= FG_THR] = 1 
    # All others are ignored

    # Return labels and potentially the argmax for regression target assignment
    return prop_labels, best_gt_per_proposal