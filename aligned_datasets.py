"""
Dataset classes for aligned MRI-X-ray right hip landmark detection.

This module provides PyTorch dataset classes for the aligned MRI-X-ray dataset where:
- MRI and X-ray images are the same size (aligned in the same coordinate system)
- Both modalities have 4 landmarks (right hip only)
- X-ray images have been transformed to MRI space using landmark-based affine transformation
- Both share the same pixel spacing

Classes:
- AlignedHipXrayDataset: Dataset for aligned X-ray images
- AlignedHipMRIDataset: Dataset for MRI slice 10 images
"""

import os
import json

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from heatmaps_utils import get_scale_factor, keypoints2heatmaps


class AlignedHipXrayDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, phase, image_size=(512, 512), num_channels=1, 
                 fuse_heatmap=False, sigma=8, partition_path="./partitions/xray_mri_hip_partition.json"):
        """
        Aligned Hip X-ray dataset class for loading aligned X-ray images with landmarks.
        
        These X-rays have been aligned to MRI space and contain only right hip landmarks.
        
        Args:
            dataset_path (str): Path to XRAY-MRI-RIGHT-HIP dataset directory
            phase (str): 'train', 'val', 'test', or 'all'
            image_size (tuple): Target image size (height, width)
            num_channels (int): Number of image channels (1 for grayscale, 3 for RGB)
            fuse_heatmap (bool): Whether to fuse heatmaps
            sigma (int): Sigma for Gaussian heatmap generation
            partition_path (str, optional): Path to partition.json file
        """
        self.phase = phase
        self.new_image_size = image_size
        self.dataset_name = 'AlignedHipXray'
        self.num_landmarks = 4  # Right hip only
        
        self.transforms = self.get_transforms()
        self.num_channels = num_channels
        self.fuse_heatmap = fuse_heatmap
        self.sigma = sigma
        
        self.image_dir = os.path.join(dataset_path, 'imgs', 'pngs')
        self.labels_dir = os.path.join(dataset_path, 'annotations', 'XRAY')
        self.pixel_sizes_dir = os.path.join(dataset_path, 'pixel_sizes', 'XRAY')

        # Find all aligned XRAY images
        self.image_files = self._find_aligned_xray_images()
        
        # Handle data splitting
        if os.path.exists(partition_path):
            partition_data = self.load_partition(partition_path)
            print(f"Loaded partition from: {partition_path}")
        else:
            partition_data = self.create_partition(partition_path)
            
        # Map phase names to partition keys
        phase_mapping = {
            'train': 'training',
            'val': 'validation', 
            'test': 'testing'
        }
        
        if self.phase == 'all':
            self.indexes = (partition_data['training'] + 
                            partition_data['validation'] + 
                            partition_data['testing'])
        elif self.phase in phase_mapping:
            partition_key = phase_mapping[self.phase]
            self.indexes = partition_data[partition_key]
        else:
            raise Exception(f"Unknown phase: {phase}")

    def __getitem__(self, index):
        image_key = self.indexes[index]
        image_path = self.image_files[image_key]

        try:
            image = self.read_image(image_path)
        except Exception as e:
            print(f"Failed to load image {image_key} at {image_path}: {e}")
            raise e

        landmarks, ratios = self.read_landmarks(image_key, original_size=(image.shape[0], image.shape[1]))
        pixel_size_x, pixel_size_y = self.read_pixel_size(image_key)
        
        width_scale_factor, height_scale_factor = get_scale_factor(image.shape, self.new_image_size)
        physical_scaling_factor = np.array([width_scale_factor * pixel_size_x, 
                                          height_scale_factor * pixel_size_y])

        try:
            transformed = self.transforms(image=image, keypoints=landmarks)
            transformed_image, transformed_landmarks = transformed['image'], transformed['keypoints']
            
            assert len(transformed_landmarks) == self.num_landmarks, \
                f"Expected {self.num_landmarks} landmarks for {image_key}, got {len(transformed_landmarks)}"
        
        except Exception as e:
            raise e
    
        heatmaps = keypoints2heatmaps(np.array(transformed_landmarks), img_size=self.new_image_size, 
                                    fuse=self.fuse_heatmap, sigma=self.sigma)

        return {
            'name': image_key,
            'image': transformed_image.float(),
            'landmarks': torch.from_numpy(np.array(transformed_landmarks)).double(),
            'heatmaps': torch.from_numpy(np.stack(heatmaps)).float(),
            'physical_scaling_factor': torch.from_numpy(physical_scaling_factor).double()
        }

    def __len__(self):
        return len(self.indexes)
    
    def _find_aligned_xray_images(self):
        """Find all aligned XRAY images in the directory structure."""
        image_files = {}
        
        if not os.path.exists(self.image_dir):
            print(f"Warning: Image directory not found: {self.image_dir}")
            return image_files
        
        # Walk through: PATIENT/VISIT/XRAY/*_aligned.png
        for patient_dir in os.listdir(self.image_dir):
            patient_path = os.path.join(self.image_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            
            for visit_dir in os.listdir(patient_path):
                visit_path = os.path.join(patient_path, visit_dir)
                if not os.path.isdir(visit_path):
                    continue
                
                xray_path = os.path.join(visit_path, 'XRAY')
                if not os.path.exists(xray_path):
                    continue
                
                # Find aligned X-ray images
                for filename in os.listdir(xray_path):
                    if filename.endswith('_aligned.png'):
                        # Key format: PATIENT-VISIT-BASEFILENAME
                        # Remove '_aligned.png' to get base filename
                        base_filename = filename.replace('_aligned.png', '')
                        key = f"{patient_dir}-{visit_dir}-{base_filename}"
                        full_path = os.path.join(xray_path, filename)
                        image_files[key] = full_path
        
        print(f"Found {len(image_files)} aligned X-ray images")
        return image_files

    def read_image(self, path):
        img = Image.open(path)

        if self.num_channels == 3:
            img_array = np.array(img.convert('RGB'), dtype=np.uint8)  # uint8 for proper Albumentations support
        elif self.num_channels == 1:
            img_array = np.array(img.convert('L'), dtype=np.uint8)
        else:
            raise ValueError(f"Invalid num_channels: {self.num_channels}")
        
        return img_array

    def read_landmarks(self, name, original_size):
        path = os.path.join(self.labels_dir, name + '.txt')
        landmarks_coordinates = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        x = float(parts[1])
                        y = float(parts[2])
                        landmarks_coordinates.append([x, y])
        
        ratios = (1.0, 1.0)  # No scaling needed, coordinates are already correct
        return landmarks_coordinates, ratios

    def read_pixel_size(self, name):
        """Read pixel size from corresponding pixel size file."""
        path = os.path.join(self.pixel_sizes_dir, name + '.txt')
        try:
            with open(path, 'r') as f:
                line = f.readline().strip()
                if line.startswith('pixel_size'):
                    parts = line.split()
                    if len(parts) >= 3:
                        pixel_size_x = float(parts[1])
                        pixel_size_y = float(parts[2])
                        return pixel_size_x, pixel_size_y
            # If file format is unexpected, return default
            print(f"Warning: Unexpected format in {path}, using default pixel size")
            return 0.3125, 0.3125
        except FileNotFoundError:
            print(f"Warning: Pixel size file not found: {path}, using default")
            return 0.3125, 0.3125
        except Exception as e:
            print(f"Warning: Error reading pixel size file: {path}, {e}")
            return 0.3125, 0.3125

    def get_transforms(self):
        transforms = [
            A.LongestMaxSize(max_size=self.new_image_size[0]),
            A.PadIfNeeded(min_height=self.new_image_size[0],
                         min_width=self.new_image_size[1],
                         border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(normalization="min_max"),
            ToTensorV2()
        ]   
        
        if self.phase == 'train':
            # Add augmentations for training
            train_transforms = [
                # Step 1: Geometric augmentations (moderate, anatomically plausible)
                A.Affine(
                    scale=(0.95, 1.05),        # Reduced scale variation (±5%)
                    translate_percent=(-0.05, 0.05),  # Small translations (±5%)
                    rotate=(-15, 15),          # Reduced rotation (±15°)
                    shear=(-5, 5),             # Reduced shear (±5°)
                    fit_output=True,          # Keep original size
                    p=0.8
                ),
                
                # Step 2: Intensity augmentations (work on uint8 [0-255])
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,      # MRIs can handle more brightness variation
                    contrast_limit=0.2,
                    p=0.7
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.6
                ),
                A.CLAHE(
                    clip_limit=2.0,
                    tile_grid_size=(8, 8),
                    p=0.3
                ),
                
                # Noise augmentations (work on uint8)
                A.GaussNoise(
                    std_range=(0.02, 0.06),    # Slightly more noise for MRI (normalized)
                    mean_range=(0, 0),         # Zero mean
                    per_channel=False,
                    p=0.35
                ),
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    p=0.25
                ),
                
                # GridDistortion instead of ElasticTransform (better keypoint support)
                A.GridDistortion(
                    num_steps=3,               # Conservative grid distortion
                    distort_limit=0.06,        # Smaller for MRI
                    p=0.15
                )
            ] 

            transforms = train_transforms + transforms

        return A.Compose(transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
 
    def load_partition(self, partition_path):
        """Load train/val/test splits from partition.json file."""
        try:
            with open(partition_path, 'r') as f:
                partition_data = json.load(f)
            
            # Validate partition data
            required_keys = ['training', 'validation', 'testing']
            if not all(key in partition_data for key in required_keys):
                raise ValueError(f"Partition file missing required keys: {required_keys}")
            
            return partition_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Partition file not found: {partition_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in partition file: {e}")
    
    def create_partition(self, partition_path, train_ratio=0.7, val_ratio=0.15):
        """
        Create a new partition.json file with train/val/test splits at the PATIENT level.
        """
        # Group images by patient ID
        patient_to_images = {}
        for image_key in self.image_files.keys():
            patient = image_key.split('-')[0]
            if patient not in patient_to_images:
                patient_to_images[patient] = []
            patient_to_images[patient].append(image_key)
        
        # Get list of unique patients and shuffle
        patients = list(patient_to_images.keys())
        np.random.shuffle(patients)
        
        # Split patients into train/val/test
        n_patients = len(patients)
        train_num = int(train_ratio * n_patients)
        val_num = int(val_ratio * n_patients)
        
        train_patients = patients[:train_num]
        val_patients = patients[train_num:train_num + val_num]
        test_patients = patients[train_num + val_num:]
        
        # Collect all images for each split
        training_files = []
        validation_files = []
        testing_files = []
        
        for patient in train_patients:
            training_files.extend(patient_to_images[patient])
        
        for patient in val_patients:
            validation_files.extend(patient_to_images[patient])
        
        for patient in test_patients:
            testing_files.extend(patient_to_images[patient])
        
        # Create splits
        partition_data = {
            'training': training_files,
            'validation': validation_files,
            'testing': testing_files
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(partition_path), exist_ok=True)
        
        # Save partition file
        with open(partition_path, 'w') as f:
            json.dump(partition_data, f, indent=2)
            
        print(f"Created partition file: {partition_path}")
        print(f"Total unique patients: {n_patients}")
        print(f"Training: {len(train_patients)} patients, {len(partition_data['training'])} images")
        print(f"Validation: {len(val_patients)} patients, {len(partition_data['validation'])} images")
        print(f"Testing: {len(test_patients)} patients, {len(partition_data['testing'])} images")
        
        return partition_data

    def visualize_samples(self, num_samples=6, figsize=(15, 10), random_selection=True, save_path=None):
        """Visualize a grid of samples with landmarks."""
        if num_samples == 'all':
            num_samples = len(self)
        else:
            num_samples = min(num_samples, len(self))
        
        if num_samples == 0:
            print("No samples to visualize")
            return None
        
        # Calculate grid dimensions
        if num_samples <= 6:
            rows, cols = 2, 3
        elif num_samples <= 9:
            rows, cols = 3, 3
        elif num_samples <= 12:
            rows, cols = 3, 4
        elif num_samples <= 16:
            rows, cols = 4, 4
        elif num_samples <= 20:
            rows, cols = 4, 5
        elif num_samples <= 25:
            rows, cols = 5, 5
        else:
            cols = 6
            rows = (num_samples + cols - 1) // cols
        
        if num_samples > 25:
            figsize = (cols * 3, rows * 3)
        elif num_samples > 12:
            figsize = (cols * 3.5, rows * 3.5)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        # Get indices
        if random_selection:
            indices = np.random.choice(len(self), size=num_samples, replace=False)
        else:
            indices = list(range(num_samples))
        
        for i, idx in enumerate(indices):
            sample = self[idx]
            image = sample['image'].numpy()
            if image.shape[0] == 1:
                image = image[0]
            landmarks = sample['landmarks'].numpy()
            
            axes[i].imshow(image, cmap='gray')
            axes[i].scatter(landmarks[:, 0], landmarks[:, 1], 
                          c='cyan', s=100, alpha=0.8, edgecolors='white', linewidth=2)
            for j, (x, y) in enumerate(landmarks):
                axes[i].annotate(f'{j+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=10, color='yellow', weight='bold')
            axes[i].set_title(f"Aligned X-ray\n{sample['name'][:40]}", fontsize=8)
            axes[i].axis('off')
        
        # Turn off remaining axes
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        return fig


class AlignedHipMRIDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, phase, image_size=(512, 512), num_channels=1, 
                 fuse_heatmap=False, sigma=8, partition_path="./partitions/xray_mri_hip_partition.json"):
        """
        Aligned Hip MRI dataset class for loading MRI slice 10 images with landmarks.
        
        These are the original MRI images that the X-rays have been aligned to.
        
        Args:
            dataset_path (str): Path to XRAY-MRI-RIGHT-HIP dataset directory
            phase (str): 'train', 'val', 'test', or 'all'
            image_size (tuple): Target image size (height, width)
            num_channels (int): Number of image channels (1 for grayscale, 3 for RGB)
            fuse_heatmap (bool): Whether to fuse heatmaps
            sigma (int): Sigma for Gaussian heatmap generation
            partition_path (str, optional): Path to partition.json file
        """
        self.phase = phase
        self.new_image_size = image_size
        self.dataset_name = 'AlignedHipMRI'
        self.num_landmarks = 4  # Right hip only
        
        self.transforms = self.get_transforms()
        self.num_channels = num_channels
        self.fuse_heatmap = fuse_heatmap
        self.sigma = sigma
        
        self.image_dir = os.path.join(dataset_path, 'imgs', 'pngs')
        self.labels_dir = os.path.join(dataset_path, 'annotations', 'MRI')
        self.pixel_sizes_dir = os.path.join(dataset_path, 'pixel_sizes', 'MRI')

        # Find all MRI images
        self.image_files = self._find_mri_images()
        
        # Handle data splitting - use same partition as X-ray for consistency
        if os.path.exists(partition_path):
            partition_data = self.load_partition(partition_path)
            print(f"Loaded partition from: {partition_path}")
        else:
            partition_data = self.create_partition(partition_path)
            
        # Map phase names to partition keys
        phase_mapping = {
            'train': 'training',
            'val': 'validation', 
            'test': 'testing'
        }
        
        if self.phase == 'all':
            self.indexes = (partition_data['training'] + 
                            partition_data['validation'] + 
                            partition_data['testing'])
        elif self.phase in phase_mapping:
            partition_key = phase_mapping[self.phase]
            self.indexes = partition_data[partition_key]
        else:
            raise Exception(f"Unknown phase: {phase}")

    def __getitem__(self, index):
        image_key = self.indexes[index]
        image_path = self.image_files[image_key]

        try:
            image = self.read_image(image_path)
        except Exception as e:
            print(f"Failed to load image {image_key} at {image_path}: {e}")
            raise e

        landmarks, ratios = self.read_landmarks(image_key, original_size=(image.shape[0], image.shape[1]))
        pixel_size_x, pixel_size_y = self.read_pixel_size(image_key)
        
        width_scale_factor, height_scale_factor = get_scale_factor(image.shape, self.new_image_size)
        physical_scaling_factor = np.array([width_scale_factor * pixel_size_x, 
                                          height_scale_factor * pixel_size_y])

        try:
            transformed = self.transforms(image=image, keypoints=landmarks)
            transformed_image, transformed_landmarks = transformed['image'], transformed['keypoints']
            
            assert len(transformed_landmarks) == self.num_landmarks, \
                f"Expected {self.num_landmarks} landmarks for {image_key}, got {len(transformed_landmarks)}"
        
        except Exception as e:
            raise e

        heatmaps = keypoints2heatmaps(np.array(transformed_landmarks), img_size=self.new_image_size, 
                                    fuse=self.fuse_heatmap, sigma=self.sigma)

        return {
            'name': image_key,
            'image': transformed_image.float(),
            'landmarks': torch.from_numpy(np.array(transformed_landmarks)).double(),
            'heatmaps': torch.from_numpy(np.stack(heatmaps)).float(),
            'physical_scaling_factor': torch.from_numpy(physical_scaling_factor).double()
        }

    def __len__(self):
        return len(self.indexes)
    
    def _find_mri_images(self):
        """Find all MRI images in the directory structure."""
        image_files = {}
        
        if not os.path.exists(self.image_dir):
            print(f"Warning: Image directory not found: {self.image_dir}")
            return image_files
        
        # Walk through: PATIENT/VISIT/MRI/*.png
        for patient_dir in os.listdir(self.image_dir):
            patient_path = os.path.join(self.image_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue
            
            for visit_dir in os.listdir(patient_path):
                visit_path = os.path.join(patient_path, visit_dir)
                if not os.path.isdir(visit_path):
                    continue
                
                mri_path = os.path.join(visit_path, 'MRI')
                if not os.path.exists(mri_path):
                    continue
                
                # Find MRI images
                for filename in os.listdir(mri_path):
                    if filename.endswith('.png'):
                        # Key format: PATIENT-VISIT-FILENAME (without .png)
                        base_filename = filename.replace('.png', '')
                        key = f"{patient_dir}-{visit_dir}-{base_filename}"
                        full_path = os.path.join(mri_path, filename)
                        image_files[key] = full_path
        
        print(f"Found {len(image_files)} MRI images")
        return image_files

    def read_image(self, path):
        img = Image.open(path)

        if self.num_channels == 3:
            img_array = np.array(img.convert('RGB'), dtype=np.uint8)  # uint8 for proper Albumentations support
        elif self.num_channels == 1:
            img_array = np.array(img.convert('L'), dtype=np.uint8)
        else:
            raise ValueError(f"Invalid num_channels: {self.num_channels}")
        
        return img_array

    def read_landmarks(self, name, original_size):
        path = os.path.join(self.labels_dir, name + '.txt')
        landmarks_coordinates = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        x = float(parts[1])
                        y = float(parts[2])
                        landmarks_coordinates.append([x, y])
        
        ratios = (1.0, 1.0)
        return landmarks_coordinates, ratios

    def read_pixel_size(self, name):
        """Read pixel size from corresponding pixel size file."""
        path = os.path.join(self.pixel_sizes_dir, name + '.txt')
        try:
            with open(path, 'r') as f:
                line = f.readline().strip()
                if line.startswith('pixel_size'):
                    parts = line.split()
                    if len(parts) >= 3:
                        pixel_size_x = float(parts[1])
                        pixel_size_y = float(parts[2])
                        return pixel_size_x, pixel_size_y
            # If file format is unexpected, return default
            print(f"Warning: Unexpected format in {path}, using default pixel size")
            return 0.3125, 0.3125
        except FileNotFoundError:
            print(f"Warning: Pixel size file not found: {path}, using default")
            return 0.3125, 0.3125
        except Exception as e:
            print(f"Warning: Error reading pixel size file: {path}, {e}")
            return 0.3125, 0.3125

    def get_transforms(self):
        transforms = [
            A.LongestMaxSize(max_size=self.new_image_size[0]),
            A.PadIfNeeded(min_height=self.new_image_size[0],
                         min_width=self.new_image_size[1],
                         border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(normalization="min_max"),
            ToTensorV2()
        ]

        if self.phase == 'train':
            # Add augmentations for training
            train_transforms = [
                # Step 1: Geometric augmentations (moderate, anatomically plausible)
                A.Affine(
                    scale=(0.95, 1.05),        # Reduced scale variation (±5%)
                    translate_percent=(-0.05, 0.05),  # Small translations (±5%)
                    rotate=(-15, 15),          # Reduced rotation (±15°)
                    shear=(-5, 5),             # Reduced shear (±5°)
                    fit_output=True,          # Keep original size
                    p=0.7
                ),
                
                # Step 2: Intensity augmentations (work on uint8 [0-255])
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,      
                    contrast_limit=0.2,
                    p=0.6
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=0.6
                ),
                #A.CLAHE(
                #    clip_limit=2.0,
                #    tile_grid_size=(8, 8),
                #    p=0.3
                #),
                
                # Noise augmentations (work on uint8)
                #A.GaussNoise(
                #    std_range=(0.02, 0.06),    # Slightly more noise for MRI (normalized)
                #    mean_range=(0, 0),         # Zero mean
                #    per_channel=False,
                #    p=0.35
                #),
                #A.GaussianBlur(
                #    blur_limit=(3, 5),
                #    p=0.25
                #),
                
                # GridDistortion instead of ElasticTransform (better keypoint support)
                A.GridDistortion(
                    num_steps=3,               # Conservative grid distortion
                    distort_limit=0.06,        # Smaller for MRI
                    p=0.15
                )
            ]

            transforms = train_transforms + transforms

        return A.Compose(transforms, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
 
    def load_partition(self, partition_path):
        """Load train/val/test splits from partition.json file."""
        try:
            with open(partition_path, 'r') as f:
                partition_data = json.load(f)
            
            # Validate partition data
            required_keys = ['training', 'validation', 'testing']
            if not all(key in partition_data for key in required_keys):
                raise ValueError(f"Partition file missing required keys: {required_keys}")
            
            return partition_data
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Partition file not found: {partition_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in partition file: {e}")
    
    def create_partition(self, partition_path, train_ratio=0.7, val_ratio=0.15):
        """
        Create a new partition.json file with train/val/test splits at the PATIENT level.
        """
        # Group images by patient ID
        patient_to_images = {}
        for image_key in self.image_files.keys():
            patient = image_key.split('-')[0]
            if patient not in patient_to_images:
                patient_to_images[patient] = []
            patient_to_images[patient].append(image_key)
        
        # Get list of unique patients and shuffle
        patients = list(patient_to_images.keys())
        np.random.shuffle(patients)
        
        # Split patients into train/val/test
        n_patients = len(patients)
        train_num = int(train_ratio * n_patients)
        val_num = int(val_ratio * n_patients)
        
        train_patients = patients[:train_num]
        val_patients = patients[train_num:train_num + val_num]
        test_patients = patients[train_num + val_num:]
        
        # Collect all images for each split
        training_files = []
        validation_files = []
        testing_files = []
        
        for patient in train_patients:
            training_files.extend(patient_to_images[patient])
        
        for patient in val_patients:
            validation_files.extend(patient_to_images[patient])
        
        for patient in test_patients:
            testing_files.extend(patient_to_images[patient])
        
        # Create splits
        partition_data = {
            'training': training_files,
            'validation': validation_files,
            'testing': testing_files
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(partition_path), exist_ok=True)
        
        # Save partition file
        with open(partition_path, 'w') as f:
            json.dump(partition_data, f, indent=2)
            
        print(f"Created partition file: {partition_path}")
        print(f"Total unique patients: {n_patients}")
        print(f"Training: {len(train_patients)} patients, {len(partition_data['training'])} images")
        print(f"Validation: {len(val_patients)} patients, {len(partition_data['validation'])} images")
        print(f"Testing: {len(test_patients)} patients, {len(partition_data['testing'])} images")
        
        return partition_data

    def visualize_samples(self, num_samples=6, figsize=(15, 10), random_selection=True, save_path=None):
        """Visualize a grid of samples with landmarks."""
        if num_samples == 'all':
            num_samples = len(self)
        else:
            num_samples = min(num_samples, len(self))
        
        if num_samples == 0:
            print("No samples to visualize")
            return None
        
        # Calculate grid dimensions
        if num_samples <= 6:
            rows, cols = 2, 3
        elif num_samples <= 9:
            rows, cols = 3, 3
        elif num_samples <= 12:
            rows, cols = 3, 4
        elif num_samples <= 16:
            rows, cols = 4, 4
        elif num_samples <= 20:
            rows, cols = 4, 5
        elif num_samples <= 25:
            rows, cols = 5, 5
        else:
            cols = 6
            rows = (num_samples + cols - 1) // cols
        
        if num_samples > 25:
            figsize = (cols * 3, rows * 3)
        elif num_samples > 12:
            figsize = (cols * 3.5, rows * 3.5)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        # Get indices
        if random_selection:
            indices = np.random.choice(len(self), size=num_samples, replace=False)
        else:
            indices = list(range(num_samples))
        
        for i, idx in enumerate(indices):
            sample = self[idx]
            image = sample['image'].numpy()
            if image.shape[0] == 1:
                image = image[0]
            landmarks = sample['landmarks'].numpy()
            
            axes[i].imshow(image, cmap='gray')
            axes[i].scatter(landmarks[:, 0], landmarks[:, 1], 
                          c='red', s=100, alpha=0.8, edgecolors='white', linewidth=2)
            for j, (x, y) in enumerate(landmarks):
                axes[i].annotate(f'{j+1}', (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=10, color='yellow', weight='bold')
            axes[i].set_title(f"MRI Slice 10\n{sample['name'][:40]}", fontsize=8)
            axes[i].axis('off')
        
        # Turn off remaining axes
        for i in range(num_samples, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {save_path}")
        
        return fig


def get_aligned_dataset(modality, dataset_path, phase, image_size, image_channels, sigma, partition_path=None):
    """
    Get aligned dataset for specified modality.
    
    Args:
        modality (str): 'mri' or 'xray'
        dataset_path (str): Path to XRAY-MRI-RIGHT-HIP dataset
        phase (str): 'train', 'val', 'test', or 'all'
        image_size (tuple): Target image size
        image_channels (int): Number of image channels
        sigma (int): Sigma for heatmap generation
        partition_path (str, optional): Path to partition file
    
    Returns:
        Dataset instance
    """
    dataset_classes = {
        "xray": AlignedHipXrayDataset,
        "mri": AlignedHipMRIDataset
    }
    
    if modality.lower() not in dataset_classes:
        raise ValueError(f"Modality {modality} not found. Choose 'mri' or 'xray'")
    
    dataset_class = dataset_classes[modality.lower()]
    
    # Set default partition path if not provided
    if partition_path is None:
        partition_path = os.path.join(dataset_path, "partitions", "xray_mri_hip_partition.json")
    
    return dataset_class(dataset_path=dataset_path, phase=phase, image_size=image_size, 
                        num_channels=image_channels, sigma=sigma, partition_path=partition_path)


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, pin_memory, 
                      num_workers, training_samples):
    """
    Create PyTorch dataloaders for train/val/test splits.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for dataloaders
        pin_memory: Whether to use pinned memory
        num_workers: Number of worker processes for data loading
        training_samples: Number of training samples to use ("all" or integer)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if training_samples != "all":
        assert len(train_dataset) >= int(training_samples), \
            "The number of training samples is greater than the number of samples in the dataset"
        train_dataset.indexes = train_dataset.indexes[:int(training_samples)]
    
    return (
        DataLoader(train_dataset, batch_size=batch_size,
                  shuffle=True, pin_memory=pin_memory, num_workers=num_workers,
                  drop_last=True),  # Drop last incomplete batch to avoid BatchNorm issues
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                  pin_memory=pin_memory, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                  pin_memory=pin_memory, num_workers=num_workers)
    )


if __name__ == "__main__":
    import sys
    
    # Dataset configuration
    dataset_path = "./path/to/XRAY-MRI-RIGHT-HIP"
    output_dir = "./path/to/visualizations/aligned_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    # Landmark labels (same for both modalities - right hip only)
    landmark_labels = {
        0: "Right Lateral ACE",
        1: "Right Femur Center", 
        2: "Adjusted Right Neck Point",
        3: "Right Alpha Angle"
    }
    
    print("=" * 80)
    print("COMPREHENSIVE ALIGNED DATASET VISUALIZATION")
    print("=" * 80)
    
    try:
        # Load datasets
        print("\nLoading datasets...")
        mri_test = AlignedHipMRIDataset(dataset_path, phase='test', image_size=(512, 512))
        xray_test = AlignedHipXrayDataset(dataset_path, phase='test', image_size=(512, 512))
        mri_all = AlignedHipMRIDataset(dataset_path, phase='all', image_size=(512, 512))
        xray_all = AlignedHipXrayDataset(dataset_path, phase='all', image_size=(512, 512))
        
        print(f"Test set: {len(mri_test)} pairs")
        print(f"Total: {len(mri_all)} pairs")
        
        # =====================================================================
        # 1. GRID OF VARIOUS PATIENTS WITH ALL VISUALIZATION STYLES
        # =====================================================================
        print("\n" + "=" * 80)
        print("1. Creating comprehensive patient grid with all visualization styles...")
        print("=" * 80)
        
        num_patients = min(5, len(mri_test))
        num_viz_styles = 8  # MRI, X-ray, 50-50 blend, color overlay, difference, checkerboard, transparency, split
        
        fig = plt.figure(figsize=(32, 6*num_patients))
        
        for patient_idx in range(num_patients):
            mri_sample = mri_test[patient_idx]
            xray_sample = xray_test[patient_idx]
            
            # Extract data
            mri_img = mri_sample['image'].squeeze().numpy()
            mri_landmarks = mri_sample['landmarks'].numpy()
            xray_img = xray_sample['image'].squeeze().numpy()
            xray_landmarks = xray_sample['landmarks'].numpy()
            
            # Normalize for visualization
            mri_norm = (mri_img - mri_img.min()) / (mri_img.max() - mri_img.min() + 1e-8)
            xray_norm = (xray_img - xray_img.min()) / (xray_img.max() - xray_img.min() + 1e-8)
            
            row = patient_idx
            patient_name = mri_sample['name']  # Get patient-visit name
            
            # Column 0: MRI
            ax = plt.subplot(num_patients, num_viz_styles, row * num_viz_styles + 1)
            ax.imshow(mri_img, cmap='gray')
            ax.scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                      c='red', s=30, marker='x', linewidths=1.5, alpha=0.9)
            if row == 0:
                ax.set_title('MRI', fontsize=12, fontweight='bold', pad=15)
            else:
                # Add patient name above each MRI image
                ax.set_title(patient_name[:45], fontsize=7, fontweight='bold', pad=5)
            # Show row number as ylabel
            ax.set_ylabel(f'Row {row+1}', fontsize=9, fontweight='bold', rotation=0, 
                         ha='right', va='center', labelpad=10)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Column 1: Aligned X-ray
            ax = plt.subplot(num_patients, num_viz_styles, row * num_viz_styles + 2)
            ax.imshow(xray_img, cmap='gray')
            ax.scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                      c='cyan', s=30, marker='x', linewidths=1.5, alpha=0.9)
            if row == 0:
                ax.set_title('Aligned X-ray', fontsize=12, fontweight='bold', pad=15)
            else:
                # Add patient name above each X-ray image
                ax.set_title(patient_name[:45], fontsize=7, fontweight='bold', pad=5)
            ax.axis('off')
            
            # Column 2: 50-50 Blend
            ax = plt.subplot(num_patients, num_viz_styles, row * num_viz_styles + 3)
            blended = 0.5 * mri_norm + 0.5 * xray_norm
            ax.imshow(blended, cmap='gray')
            ax.scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                      c='red', s=25, marker='o', linewidths=1, alpha=0.8)
            ax.scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                      c='cyan', s=25, marker='s', linewidths=1, alpha=0.8)
            if row == 0:
                ax.set_title('50-50 Blend', fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Column 3: Color Overlay (MRI=red, X-ray=cyan)
            ax = plt.subplot(num_patients, num_viz_styles, row * num_viz_styles + 4)
            overlay_rgb = np.zeros((*mri_img.shape, 3))
            overlay_rgb[:, :, 0] = mri_norm * 0.8
            overlay_rgb[:, :, 1] = xray_norm * 0.5 + mri_norm * 0.3
            overlay_rgb[:, :, 2] = xray_norm * 0.8
            ax.imshow(overlay_rgb)
            ax.scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                      c='red', s=25, marker='o', linewidths=1, alpha=0.8)
            ax.scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                      c='cyan', s=25, marker='s', linewidths=1, alpha=0.8)
            if row == 0:
                ax.set_title('Color Overlay\n(MRI=Red, X-ray=Cyan)', fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # Column 4: Absolute Difference
            ax = plt.subplot(num_patients, num_viz_styles, row * num_viz_styles + 5)
            difference = np.abs(mri_norm - xray_norm)
            im = ax.imshow(difference, cmap='hot')
            ax.scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                      c='lime', s=25, marker='o', linewidths=1, alpha=0.8)
            ax.scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                      c='lime', s=25, marker='s', linewidths=1, alpha=0.8)
            if row == 0:
                ax.set_title('Absolute Difference', fontsize=12, fontweight='bold')
                # Add colorbar only for first row
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Intensity Diff', fontsize=8)
            ax.axis('off')
            
            # Column 5: Checkerboard
            ax = plt.subplot(num_patients, num_viz_styles, row * num_viz_styles + 6)
            checker_size = 40
            h, w = mri_img.shape[:2]
            checker = np.zeros((h, w))
            for i in range(0, h, checker_size):
                for j in range(0, w, checker_size):
                    if (i // checker_size + j // checker_size) % 2 == 0:
                        checker[i:i+checker_size, j:j+checker_size] = 1
            checkerboard = np.where(checker, mri_norm, xray_norm)
            ax.imshow(checkerboard, cmap='gray')
            ax.scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                      c='red', s=25, marker='o', linewidths=1, alpha=0.8)
            ax.scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                      c='cyan', s=25, marker='s', linewidths=1, alpha=0.8)
            if row == 0:
                ax.set_title('Checkerboard\n(40px tiles)', fontsize=11, fontweight='bold')
            ax.axis('off')
            
            # Column 6: X-ray @ 50% over MRI
            ax = plt.subplot(num_patients, num_viz_styles, row * num_viz_styles + 7)
            ax.imshow(mri_img, cmap='gray')
            ax.imshow(xray_img, cmap='gray', alpha=0.5)
            ax.scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                      c='red', s=25, marker='o', linewidths=1, alpha=0.8)
            ax.scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                      c='cyan', s=25, marker='s', linewidths=1, alpha=0.8)
            if row == 0:
                ax.set_title('X-ray @ 50% Opacity', fontsize=12, fontweight='bold')
            ax.axis('off')
            
            # Column 7: Split View
            ax = plt.subplot(num_patients, num_viz_styles, row * num_viz_styles + 8)
            split = w // 2
            combined = np.copy(mri_norm)
            combined[:, split:] = xray_norm[:, split:]
            ax.imshow(combined, cmap='gray')
            ax.axvline(x=split, color='lime', linewidth=2, linestyle='--', alpha=0.8)
            ax.scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                      c='red', s=25, marker='o', linewidths=1, alpha=0.8)
            ax.scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                      c='cyan', s=25, marker='s', linewidths=1, alpha=0.8)
            # Add MRI/X-ray labels on split view
            ax.text(split//2, h-20, 'MRI', fontsize=10, color='red', 
                   weight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            ax.text(split + split//2, h-20, 'X-ray', fontsize=10, color='cyan',
                   weight='bold', ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            if row == 0:
                ax.set_title('Split View\n(MRI | X-ray)', fontsize=11, fontweight='bold')
            ax.axis('off')
        
        # Add legend for marker shapes at the bottom
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=10, label='MRI Landmarks', markeredgecolor='red', linewidth=0),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='cyan',
                   markersize=10, label='X-ray Landmarks', markeredgecolor='cyan', linewidth=0)
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
                  fontsize=12, frameon=True, fancybox=True, shadow=True,
                  bbox_to_anchor=(0.5, -0.005))
        
        plt.tight_layout(rect=[0, 0.015, 1, 1])  # Make room for legend at bottom
        grid_path = os.path.join(output_dir, 'comprehensive_patient_grid.png')
        plt.savefig(grid_path, dpi=200, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"✓ Saved comprehensive patient grid: {grid_path}")
        
        # =====================================================================
        # 2. ALIGNMENT ERROR ANALYSIS
        # =====================================================================
        print("\n" + "=" * 80)
        print("2. Computing alignment error analysis...")
        print("=" * 80)
        
        all_errors = []
        per_landmark_errors = [[], [], [], []]
        pair_errors = []
        
        for i in range(len(mri_all)):
            mri_sample = mri_all[i]
            xray_sample = xray_all[i]
            
            mri_landmarks = mri_sample['landmarks'].numpy()
            xray_landmarks = xray_sample['landmarks'].numpy()
            
            landmark_errors = np.linalg.norm(mri_landmarks - xray_landmarks, axis=1)
            mean_error = np.mean(landmark_errors)
            
            all_errors.extend(landmark_errors)
            for j in range(4):
                per_landmark_errors[j].append(landmark_errors[j])
            pair_errors.append((i, mri_sample['name'], mean_error))
        
        all_errors = np.array(all_errors)
        
        # Create error analysis plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Overall distribution
        axes[0].hist(all_errors, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(all_errors), color='red', linestyle='--', linewidth=2.5, 
                       label=f'Mean: {np.mean(all_errors):.2f}px')
        axes[0].axvline(np.median(all_errors), color='green', linestyle='--', linewidth=2.5,
                       label=f'Median: {np.median(all_errors):.2f}px')
        axes[0].set_xlabel('Alignment Error (pixels)', fontsize=13)
        axes[0].set_ylabel('Frequency', fontsize=13)
        axes[0].set_title('Overall Landmark Alignment Error Distribution', fontsize=15, weight='bold')
        axes[0].legend(fontsize=12)
        axes[0].grid(alpha=0.3)
        
        # Per-landmark comparison
        landmark_names_short = ['L1: Lateral ACE', 'L2: Femur Center', 'L3: Neck Point', 'L4: Alpha Angle']
        landmark_means = [np.mean(per_landmark_errors[j]) for j in range(4)]
        landmark_stds = [np.std(per_landmark_errors[j]) for j in range(4)]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        x = np.arange(4)
        bars = axes[1].bar(x, landmark_means, yerr=landmark_stds, color=colors, alpha=0.7,
                          capsize=6, edgecolor='black', linewidth=1.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(landmark_names_short, fontsize=11, rotation=15, ha='right')
        axes[1].set_ylabel('Mean Alignment Error (pixels)', fontsize=13)
        axes[1].set_title('Mean Error by Landmark', fontsize=15, weight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, landmark_means, landmark_stds)):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + std + 1,
                        f'{mean:.2f}±{std:.2f}',
                        ha='center', va='bottom', fontsize=10, weight='bold')
        
        plt.tight_layout()
        error_path = os.path.join(output_dir, 'alignment_error_analysis.png')
        plt.savefig(error_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved alignment error analysis: {error_path}")
        
        print(f"\nAlignment Statistics:")
        print(f"  Mean error: {np.mean(all_errors):.2f} ± {np.std(all_errors):.2f} pixels")
        print(f"  Median error: {np.median(all_errors):.2f} pixels")
        print(f"  Range: [{np.min(all_errors):.2f}, {np.max(all_errors):.2f}] pixels")
        
        # =====================================================================
        # 3. BEST AND WORST ALIGNED PAIRS
        # =====================================================================
        print("\n" + "=" * 80)
        print("3. Creating best and worst alignment visualizations...")
        print("=" * 80)
        
        pair_errors.sort(key=lambda x: x[2])
        
        best_indices = [pair_errors[i][0] for i in range(min(3, len(pair_errors)))]
        worst_indices = [pair_errors[-(i+1)][0] for i in range(min(3, len(pair_errors)))]
        
        # Best alignments
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('BEST ALIGNED PAIRS', fontsize=18, weight='bold', y=0.995)
        
        for plot_idx, data_idx in enumerate(best_indices):
            mri_sample = mri_all[data_idx]
            xray_sample = xray_all[data_idx]
            
            mri_img = mri_sample['image'].squeeze().numpy()
            mri_landmarks = mri_sample['landmarks'].numpy()
            xray_img = xray_sample['image'].squeeze().numpy()
            xray_landmarks = xray_sample['landmarks'].numpy()
            
            errors = np.linalg.norm(mri_landmarks - xray_landmarks, axis=1)
            mean_error = np.mean(errors)
            patient_name = mri_sample['name']
            
            mri_norm = (mri_img - mri_img.min()) / (mri_img.max() - mri_img.min() + 1e-8)
            xray_norm = (xray_img - xray_img.min()) / (xray_img.max() - xray_img.min() + 1e-8)
            
            # MRI
            axes[plot_idx, 0].imshow(mri_img, cmap='gray')
            axes[plot_idx, 0].scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                                     c='red', s=100, marker='x', linewidths=2)
            # Add landmark numbers
            for j in range(4):
                axes[plot_idx, 0].text(mri_landmarks[j, 0]+10, mri_landmarks[j, 1]-10, 
                                      f'L{j+1}', fontsize=9, color='yellow', weight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
            axes[plot_idx, 0].set_title(f'MRI (Rank #{plot_idx+1})\n{patient_name[:40]}', 
                                       fontsize=10, weight='bold')
            axes[plot_idx, 0].axis('off')
            
            # X-ray
            axes[plot_idx, 1].imshow(xray_img, cmap='gray')
            axes[plot_idx, 1].scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                                     c='cyan', s=100, marker='x', linewidths=2)
            # Add landmark numbers
            for j in range(4):
                axes[plot_idx, 1].text(xray_landmarks[j, 0]+10, xray_landmarks[j, 1]-10,
                                      f'L{j+1}', fontsize=9, color='yellow', weight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
            axes[plot_idx, 1].set_title(f'X-ray (Error: {mean_error:.2f}px)', fontsize=11, weight='bold')
            axes[plot_idx, 1].axis('off')
            
            # Overlay
            axes[plot_idx, 2].imshow(mri_img, cmap='gray', alpha=0.6)
            axes[plot_idx, 2].imshow(xray_img, cmap='hot', alpha=0.4)
            for j in range(4):
                axes[plot_idx, 2].plot([mri_landmarks[j, 0], xray_landmarks[j, 0]],
                                      [mri_landmarks[j, 1], xray_landmarks[j, 1]],
                                      'g--', linewidth=2, alpha=0.7)
            axes[plot_idx, 2].scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                                     c='red', s=80, marker='o', linewidths=2)
            axes[plot_idx, 2].scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                                     c='cyan', s=80, marker='s', linewidths=2)
            # Add landmark numbers on overlay
            for j in range(4):
                mid_x = (mri_landmarks[j, 0] + xray_landmarks[j, 0]) / 2
                mid_y = (mri_landmarks[j, 1] + xray_landmarks[j, 1]) / 2
                axes[plot_idx, 2].text(mid_x, mid_y, f'L{j+1}', fontsize=9, color='lime',
                                      weight='bold', ha='center',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            axes[plot_idx, 2].set_title('Overlay', fontsize=11, weight='bold')
            axes[plot_idx, 2].axis('off')
            
            # Error bars
            axes[plot_idx, 3].barh(range(4), errors, color=colors, alpha=0.7, edgecolor='black')
            axes[plot_idx, 3].set_yticks(range(4))
            axes[plot_idx, 3].set_yticklabels([f'L{i+1}' for i in range(4)])
            axes[plot_idx, 3].set_xlabel('Error (px)', fontsize=10)
            axes[plot_idx, 3].set_title('Per-Landmark Error', fontsize=11, weight='bold')
            axes[plot_idx, 3].grid(axis='x', alpha=0.3)
            for i, err in enumerate(errors):
                axes[plot_idx, 3].text(err + 0.2, i, f'{err:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        best_path = os.path.join(output_dir, 'best_alignments.png')
        plt.savefig(best_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved best alignments: {best_path}")
        
        # Worst alignments
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        fig.suptitle('WORST ALIGNED PAIRS', fontsize=18, weight='bold', y=0.995)
        
        for plot_idx, data_idx in enumerate(worst_indices):
            mri_sample = mri_all[data_idx]
            xray_sample = xray_all[data_idx]
            
            mri_img = mri_sample['image'].squeeze().numpy()
            mri_landmarks = mri_sample['landmarks'].numpy()
            xray_img = xray_sample['image'].squeeze().numpy()
            xray_landmarks = xray_sample['landmarks'].numpy()
            
            errors = np.linalg.norm(mri_landmarks - xray_landmarks, axis=1)
            mean_error = np.mean(errors)
            patient_name = mri_sample['name']
            
            mri_norm = (mri_img - mri_img.min()) / (mri_img.max() - mri_img.min() + 1e-8)
            xray_norm = (xray_img - xray_img.min()) / (xray_img.max() - xray_img.min() + 1e-8)
            
            # MRI
            axes[plot_idx, 0].imshow(mri_img, cmap='gray')
            axes[plot_idx, 0].scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                                     c='red', s=100, marker='x', linewidths=2)
            # Add landmark numbers
            for j in range(4):
                axes[plot_idx, 0].text(mri_landmarks[j, 0]+10, mri_landmarks[j, 1]-10,
                                      f'L{j+1}', fontsize=9, color='yellow', weight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
            axes[plot_idx, 0].set_title(f'MRI (Rank #{len(pair_errors)-plot_idx})\n{patient_name[:40]}', 
                                       fontsize=10, weight='bold')
            axes[plot_idx, 0].axis('off')
            
            # X-ray
            axes[plot_idx, 1].imshow(xray_img, cmap='gray')
            axes[plot_idx, 1].scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                                     c='cyan', s=100, marker='x', linewidths=2)
            # Add landmark numbers
            for j in range(4):
                axes[plot_idx, 1].text(xray_landmarks[j, 0]+10, xray_landmarks[j, 1]-10,
                                      f'L{j+1}', fontsize=9, color='yellow', weight='bold',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.6))
            axes[plot_idx, 1].set_title(f'X-ray (Error: {mean_error:.2f}px)', fontsize=11, weight='bold')
            axes[plot_idx, 1].axis('off')
            
            # Overlay
            axes[plot_idx, 2].imshow(mri_img, cmap='gray', alpha=0.6)
            axes[plot_idx, 2].imshow(xray_img, cmap='hot', alpha=0.4)
            for j in range(4):
                axes[plot_idx, 2].plot([mri_landmarks[j, 0], xray_landmarks[j, 0]],
                                      [mri_landmarks[j, 1], xray_landmarks[j, 1]],
                                      'g--', linewidth=2, alpha=0.7)
            axes[plot_idx, 2].scatter(mri_landmarks[:, 0], mri_landmarks[:, 1],
                                     c='red', s=80, marker='o', linewidths=2)
            axes[plot_idx, 2].scatter(xray_landmarks[:, 0], xray_landmarks[:, 1],
                                     c='cyan', s=80, marker='s', linewidths=2)
            # Add landmark numbers on overlay
            for j in range(4):
                mid_x = (mri_landmarks[j, 0] + xray_landmarks[j, 0]) / 2
                mid_y = (mri_landmarks[j, 1] + xray_landmarks[j, 1]) / 2
                axes[plot_idx, 2].text(mid_x, mid_y, f'L{j+1}', fontsize=9, color='lime',
                                      weight='bold', ha='center',
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            axes[plot_idx, 2].set_title('Overlay', fontsize=11, weight='bold')
            axes[plot_idx, 2].axis('off')
            
            # Error bars
            axes[plot_idx, 3].barh(range(4), errors, color=colors, alpha=0.7, edgecolor='black')
            axes[plot_idx, 3].set_yticks(range(4))
            axes[plot_idx, 3].set_yticklabels([f'L{i+1}' for i in range(4)])
            axes[plot_idx, 3].set_xlabel('Error (px)', fontsize=10)
            axes[plot_idx, 3].set_title('Per-Landmark Error', fontsize=11, weight='bold')
            axes[plot_idx, 3].grid(axis='x', alpha=0.3)
            for i, err in enumerate(errors):
                axes[plot_idx, 3].text(err + 0.5, i, f'{err:.2f}', va='center', fontsize=9)
        
        plt.tight_layout()
        worst_path = os.path.join(output_dir, 'worst_alignments.png')
        plt.savefig(worst_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved worst alignments: {worst_path}")
        
        # Summary
        print("\n" + "=" * 80)
        print("VISUALIZATION COMPLETE")
        print("=" * 80)
        print(f"\nGenerated visualizations:")
        print(f"  1. {grid_path}")
        print(f"  2. {error_path}")
        print(f"  3. {best_path}")
        print(f"  4. {worst_path}")
        print(f"\nBest 3 pairs (lowest error):")
        for i, (idx, name, error) in enumerate(pair_errors[:3]):
            print(f"  {i+1}. {name[:60]} - {error:.2f} px")
        print(f"\nWorst 3 pairs (highest error):")
        for i, (idx, name, error) in enumerate(pair_errors[-3:]):
            print(f"  {i+1}. {name[:60]} - {error:.2f} px")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
