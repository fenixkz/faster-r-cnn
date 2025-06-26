import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataset import get_datasets, FaceDataset, Sample
from model.FasterRCNN import FasterRCNN
from typing import List
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm
import os 
import time
from contextlib import nullcontext

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 640
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # A path to the project directory, upper directory of scripts/ dir

# Set seed for reproducibility
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def collate_fn(batch: List[Sample]):
    '''
    A custom collate function for FaceDataset to work with Dataloader
    '''
    
    # Stack images using torch.stack
    images = torch.stack([item.image for item in batch])

    # Process BBoxes: Pad and Create Mask
    gt_bboxes_list = [item.bboxes for item in batch]   
    max_bboxes = max(boxes.shape[0] for boxes in gt_bboxes_list) # Get a number of maximum gt_bboxes in the batch

    padded_bboxes = []
    masks = []

    # As different samples from the same batch may have different number of bboxes we should pad to make it consistent
    if max_bboxes > 0: 
        for boxes in gt_bboxes_list:
            num_boxes = boxes.shape[0]
            # Pad boxes tensor
            padding_size = max_bboxes - num_boxes
            padded_box = F.pad(boxes, (0, 0, 0, padding_size), mode='constant', value=0) # Pad dim 0 (rows) after current boxes
            padded_bboxes.append(padded_box)

            # Create mask tensor
            mask = torch.zeros(max_bboxes, dtype=torch.bool)
            if num_boxes > 0:
                mask[:num_boxes] = True
            masks.append(mask)
        # Stack padded tensors and masks
        bboxes_tensor = torch.stack(padded_bboxes).float() # Ensure float type
        masks_tensor = torch.stack(masks)
    else: # Handle case where no image in the batch has any boxes
        B = len(batch)
        bboxes_tensor = torch.zeros((B, 0, 4)).float() # Shape (B, 0, 4)
        masks_tensor = torch.zeros((B, 0), dtype=torch.bool) # Shape (B, 0)


    return {'image': images, 'bboxes': bboxes_tensor, 'masks': masks_tensor}

def main():
    # Create a results directory
    os.makedirs(f"{PROJECT_DIR}/results", exist_ok=True)
    # Path where we save results of each try
    path = f"{PROJECT_DIR}/results/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    os.makedirs(path, exist_ok=True)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
    ###### Data preparation
    train_dataset, val_dataset = get_datasets(image_size=IMAGE_HEIGHT)
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn # Larger batch size for eval usually ok
    )
    ###### End of Data Preparation


    ########## DEBUG
    # from torchvision.utils import draw_bounding_boxes
    # from torchvision.transforms.functional import to_pil_image
    # import numpy as np

    # print("Performing a visual sanity check on one batch...")
    # for _ in range(10):
    #     check_batch = next(iter(train_loader))
    #     images = check_batch['image']
    #     gt_bboxes = check_batch['bboxes']
    #     gt_masks = check_batch['masks']

    #     # Get the first image and its boxes from the batch
    #     img_tensor = images[0]
    #     boxes_for_img = gt_bboxes[0][gt_masks[0]] # Use the mask to get only valid boxes

    #     # Un-normalize the image tensor to make it viewable
    #     # NOTE: This assumes your get_datasets() adds normalization.
    #     # If not, you only need to multiply by 255.
    #     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    #     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    #     img_un_norm = img_tensor * std + mean
    #     img_un_norm = torch.clamp(img_un_norm, 0, 1)

    #     # Convert to uint8 for drawing
    #     img_uint8 = (img_un_norm * 255).byte()

    #     # Draw the boxes
    #     img_with_boxes = draw_bounding_boxes(img_uint8, boxes_for_img, colors="red", width=2)

    #     # Display the image
    #     to_pil_image(img_with_boxes).show()

    # # --- HALT THE SCRIPT FOR NOW ---
    # import sys
    # sys.exit("Stopping after sanity check.")

    ################# END OF DEBUG




    ###### Model preparation
    model = FasterRCNN(image_width=IMAGE_WIDTH, 
                       image_height=IMAGE_HEIGHT,
                       resnet_depth=18,
                       num_classes=2,
                       device=device,
                       anchors_mini_batch_size=256,
                       dropout=0).to(device)
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
    # print("Compiling the model... (this may take a moment)")
    # model = torch.compile(model)
    # Use SGD optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9) 
    # A scheduler for LR to reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=3)
    # Use a mixed precision and gradient scaler
    use_amp = True if device == 'cuda' else False

    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    pt_dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device_str, dtype=pt_dtype)

    enable_scaler = use_amp and dtype == 'float16'
    scaler = torch.amp.GradScaler(device, enabled=enable_scaler)
    ####### End of Model Preparation

    ####### Metrics
    map = MeanAveragePrecision()
    num_epochs = 5
    best_map = 0
    train_losses = []
    val_maps = []
    patience = 10
    no_improve = 0
    ####### End of Metrics

    def save_checkpoint(current_epoch: int):
        '''
        A nested function to save checkpoint after each epoch.
        '''
        checkpoint_data = {
                            'model': model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'current_epoch': current_epoch,
                            'best_map': best_map,
                          }
        if use_amp:
            checkpoint_data['scaler'] = scaler.state_dict()
        torch.save(checkpoint_data, os.path.join(path, 'checkpoint.pth'))




    for e in range(num_epochs):
        epoch_loss = 0
        steps = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {e+1}/{num_epochs} [Train]", leave=True)
        model.train()
        for i, batch in enumerate(progress_bar):
            ####### Data preparation
            imgs = batch['image'].to(device) # Get the images
            gt_bboxes = batch['bboxes'].to(device) # Get the ground truth bounding boxes
            gt_masks = batch['masks'].to(device) # Get the ground truth masks
            ####### End of data preparation

            ####### Forwards pass using AMP
            with ctx:
                loss = model(imgs, gt_bboxes, gt_masks)
            ####### End of Forward Pass
            
            ####### Backward pass
            # scaler.scale() is a no-op if scaler is disabled
            scaler.scale(loss).backward()
            # scaler.unscale_ is a no-op if scaler is disabled
            scaler.unscale_(optimizer)
            # A good practice to clamp the maximum gradient to avoid grad explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # scaler.step is a no-op if scaler is disabled
            scaler.step(optimizer)
            # scaler.update is a no-op if scaler is disabled
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ####### End of Backward Pass

            epoch_loss += loss.item()
            steps += 1
            progress_bar.set_postfix({'loss': epoch_loss / steps})

        avg_loss = epoch_loss / steps
        ######## Evaluation per epoch
        with torch.no_grad():
            model.eval()
            val_bar = tqdm(val_loader, desc=f"Epoch {e+1}/{num_epochs} [Val]", leave=True)
            for batch in val_bar:
                imgs = batch['image'].to(device) # Get the images
                gt_bboxes = batch['bboxes'].to(device) # Get the ground truth bounding boxes
                gt_masks = batch['masks'].to(device) # Get the ground truth masks
                
                pred_dict = model(imgs, gt_bboxes, gt_masks)
                batch_size = imgs.shape[0]
                target = []
                for j in range(batch_size):
                    bbox = gt_bboxes[j, gt_masks[j], :]
                    labels = torch.ones(len(bbox), dtype=torch.int64, device=device)
                    target.append({'boxes': bbox, 'labels': labels})
                map.update(pred_dict, target)
            map_metric = map.compute()
            map.reset() # Reset metric for the next epoch! Very important.
        ####### End of Evaluation

        # Save the best model if the best map is achieved
        if best_map < map_metric['map']:
            best_map = map_metric['map'] # Assign a new best mAP 
            torch.save(model.state_dict(), os.path.join(path, 'best_model.pth')) # Save best model
            print(f"New best model saved with best mAP: {best_map:.6f}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                progress_bar.set_description(f"Training (Early stopping at epoch {e+1})")
                progress_bar.close()
                print(f"Stopping training at epoch {e} with best mAP: {best_map} and current mAP {map_metric['map']}")
                break
        
        print(f"Epoch {e+1}/{num_epochs}, Average Loss: {avg_loss}, Validation mAP: {map_metric['map']:.4f}")
        
        ##### Save data for metrics
        train_losses.append(avg_loss)
        val_maps.append(map_metric['map'])
        save_checkpoint(current_epoch=e)
        ###### End of Save data for metrics

        ###### Step the scheduler
        scheduler.step(map_metric['map'])
        ###### End of Step the scheduler

    
if __name__=='__main__':
    set_seed()
    main()