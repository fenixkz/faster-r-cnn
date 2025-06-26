import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import os 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import kagglehub
from dataclasses import dataclass
# Download latest version
path = kagglehub.dataset_download("fareselmenshawii/face-detection-dataset")

BASE_PATH= "/home/fenixkz/.cache/kagglehub/datasets/fareselmenshawii/face-detection-dataset/versions/3" 
# The exact mean and std from the ResNet documentation
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

@dataclass
class Sample:
    image: torch.Tensor
    bboxes: torch.Tensor
    labels: torch.Tensor

class FaceDataset(Dataset):

    def __init__(self, base_path: str, train: bool = True, transform: A.transforms = None):
        super().__init__()
        csv_path = os.path.join(base_path, "csv", "train.csv") if train else os.path.join(base_path, "csv", "val.csv")
        img_dir_path = os.path.join(base_path, "images", "train") if train else os.path.join(base_path, "images", "val")
        self.entries = self.load_csv(csv_path, img_dir_path)
        self.transform = transform 

    def load_csv(self, csv_path: str, img_dir_path: str):
        # Read the .csv into dataframe format
        df: pd.DataFrame = pd.read_csv(csv_path)
        # Prepare a list for image entries
        image_entries = []
        # Group dataframe by names (there can be several bounding boxes in the same image)
        df_grouped = df.groupby('image_name')

        for name, group in df_grouped:
            # Get an absolute path to the image
            img_path = os.path.join(img_dir_path, name)
            # Check if exists
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found during initial load: {img_path}. Skipping.")
                continue
            # Get bounding boxes as a list of np.float32 dtype
            bboxes = group[['x_center', 'y_center', 'width', 'height']].values.astype(np.float32).tolist()
            # Get dimensions of the image (width, height)
            image_w = group['image_width'].values[0]
            image_h = group['image_height'].values[0]
            # Convert each bounding box from (x, y, w, h) format to (x1, y1, x2, y2)
            for bbox in bboxes:
                center_x = bbox[0] * image_w
                center_y = bbox[1] * image_h
                bbox_width = bbox[2] * image_w
                bbox_height = bbox[3] * image_h
                bbox[0] = max(0, min(image_w, center_x - bbox_width / 2))
                bbox[1] = max(0, min(image_h, center_y - bbox_height / 2))
                bbox[2] = max(0, min(image_w, center_x + bbox_width / 2))
                bbox[3] = max(0, min(image_h, center_y + bbox_height / 2))
            # A binary classification, label 1 for face, 0 for background
            labels = [1] * len(bboxes)
            # Finally add an entry to the list
            image_entries.append({
                'name': name, 
                'path': img_path,
                'bboxes': bboxes,
                'labels': labels
            })
        return image_entries

    def __len__(self): # Return size of this dataset
        return len(self.entries)
    
    def __getitem__(self, idx: int):
        # Get an entry with corresponding index
        entry = self.entries[idx]
        # Read an image using PIL in RGB format and np.uint8 dtype 
        # Note: in load_csv we already verified that the image exists
        img = np.array(Image.open(entry['path']).convert("RGB"), dtype=np.uint8) # Note: it loads the image in HWC format
        
        bboxes = entry['bboxes'] # A list of list of floats: List[List[float]]
        labels = entry['labels'] # A list of integers

        # Apply Albumentations transforms if present
        if self.transform:
            try:
                transformed = self.transform(image=img, bboxes=bboxes, class_labels=labels)
                img = transformed['image']
                bboxes = transformed['bboxes'] if 'bboxes' in transformed else []
                labels = transformed['class_labels'] if 'class_labels' in transformed else []
            except Exception as e:
                print(f"Error during transform for image {entry['path']}: {e}")
                return None # Handle transform errors
        
        # Make bboxes a tensor 
        if len(bboxes) > 0:
            final_bboxes = torch.FloatTensor(bboxes)
        else:
            final_bboxes = torch.zeros((0, 4), dtype=torch.float32)
        
        return Sample(img, final_bboxes, torch.tensor(labels, dtype=torch.int64)) # Use int64 for better compatability with loss, etc.
                
# Make a util function to get the train and val datasets
def get_datasets(image_size=None):
    
    # A set of transformations for train data
    train_transform = A.Compose([
        A.Resize(height=image_size, width=image_size) if image_size else A.NoOp(),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomRotate90(p=0.2),
        A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-10, 10), shear=(-5, 5), p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(), # Note: permutes image dimensions to get CHW format
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'], clip=True)) # IMPORTANT to pass correct format of bounding boxes, we use (x1, y1, x2, y2) which is pascal_voc

    # Validation transform, only resize and make a tensor
    val_transform = A.Compose([
        A.Resize(height=image_size, width=image_size) if image_size else A.NoOp(),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(), # Note: permutes image dimensions to get CHW format
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


    train_dataset = FaceDataset(BASE_PATH, train=True, transform=train_transform)
    val_dataset = FaceDataset(BASE_PATH, train=False, transform=val_transform)

    return train_dataset, val_dataset


###### Uncomment for sanity check that the image and bboxes are correctly transformed 

if __name__=='__main__':

    
    _, val_dataset = get_datasets(image_size=640)

    entry = val_dataset[np.random.randint(low=0, high=len(val_dataset))]

    img = entry.image
    # 1. Convert mean and std to tensors and reshape for broadcasting
    # They need to be of shape (C, 1, 1) to work with an image of shape (C, H, W)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    # 2. Apply the unnormalization formula
    img = img * std + mean
    
    # # 3. Clip the values to be in the [0, 1] range
    img = torch.clamp(img, 0, 1)
    # --- END of Unnormalization Logic ---

    # Permute from (C, H, W) to (H, W, C) for numpy/PIL
    img = img.permute(1, 2, 0).numpy() 
    img = (img * 255).astype(np.uint8) 
    img = Image.fromarray(img)

    draw = ImageDraw.Draw(img)
    bboxes = entry.bboxes

    for bbox in bboxes:
        bbox = bbox.numpy()
        # Convert coordinates to integers and list
        bbox = [int(x) for x in bbox]
        draw.rectangle(bbox, outline='red', width=3)
    img.show()