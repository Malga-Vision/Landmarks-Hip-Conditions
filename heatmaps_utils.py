"""Utility functions for heatmap generation and landmark extraction.

This module provides functions for:
- Converting keypoints to Gaussian heatmaps
- Extracting landmarks from heatmaps
- Scaling and fusing heatmaps
- Visualizing landmarks and heatmaps
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import torch


def get_scale_factor(original_image_shape, downsampled_image_shape):
    width_aspect_ratio = original_image_shape[1] / downsampled_image_shape[1]
    height_aspect_ratio = original_image_shape[0] / downsampled_image_shape[0]
    
    return width_aspect_ratio, height_aspect_ratio
    
def fuse_heatmaps(heatmaps):
    return np.maximum.reduce(heatmaps)

def keypoints2heatmaps(landmarks, img_size, fuse=False, sigma=0):
    heatmaps = []
    for i in range(len(landmarks)):
        heatmap = np.zeros(img_size, dtype=np.float32)
        x, y = landmarks[i].astype(int)
        heatmap[y, x] = 1.0
        heatmap = gaussian_filter(heatmap, sigma=sigma) if sigma > 0 else heatmap
        heatmap = np.where(heatmap < 0.5 * heatmap.max(), 0, 1)
        assert is_binary_image(heatmap), f"Heatmap Image is not binary: {np.unique(heatmap)}"
        heatmaps.append(heatmap)
    
    return fuse_heatmaps(heatmaps) if fuse else np.array(heatmaps)
    
    
def is_binary_image(image):
    return len(np.unique(image)) == 2


def get_mode_probability(heatmap):
    return np.max(heatmap)

def get_hottest_point(heatmap):
    # Find the index of the maximum value
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # Get the maximum value and its neighbors
    max_val = heatmap[y, x]
    
    # Check if the maximum is at the edge of the heatmap
    if x > 0 and x < heatmap.shape[1] - 1 and y > 0 and y < heatmap.shape[0] - 1:
        # Quadratic interpolation in both x and y directions
        dx = 0.5 * (heatmap[y, x+1] - heatmap[y, x-1]) / (2 * max_val - heatmap[y, x+1] - heatmap[y, x-1])
        dy = 0.5 * (heatmap[y+1, x] - heatmap[y-1, x]) / (2 * max_val - heatmap[y+1, x] - heatmap[y-1, x])
    else:
        dx, dy = 0, 0
        
    return x + dx, y + dy


def extract_landmark(heatmap, use_argmax=True):

    if use_argmax:
        # Argmax method
        x, y = get_hottest_point(heatmap)

    else:
        # Convert heatmap to binary image
        binary_heatmap = np.where(heatmap < 0.5 * heatmap.max(), 0, 1)
        assert is_binary_image(binary_heatmap), f"Image is not binary: {np.unique(binary_heatmap)}"
        
        # Convert to uint8
        binary_heatmap = (binary_heatmap * 255).astype(np.uint8)
        
        # FindContours method        
        contours, _ = cv2.findContours(binary_heatmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(max_contour)

            if M['m00'] != 0:
                x = M['m10'] / M['m00']
                y = M['m01'] / M['m00']
            else:
                raise ValueError(f"Zero division error during moment calculation for landmark extraction: {M}")
        else:
            raise ValueError(f"No contours found to extract landmark")
        
    return np.array([x, y], dtype=np.float64)
    

def save_landmarks_heatmaps(batch_images, batch_keypoints, batch_heatmaps, batch_predicted_heatmaps, batch_names, save_dir):
    os.makedirs(os.path.join(save_dir, "plot_landmarks"), exist_ok=True)
    
    # Define color maps
    cmap_yellow = LinearSegmentedColormap.from_list('my_cmap_yellow', [(0, (0, 0, 0, 0)), (1, 'yellow')], N=256)
    cmap_green = LinearSegmentedColormap.from_list('my_cmap_green', [(0, (0, 0, 0, 0)), (1, 'green')], N=256)
    cmap_red = LinearSegmentedColormap.from_list('my_cmap_red', [(0, (0, 0, 0, 0)), (1, 'red')], N=256)

    image_size = batch_images.shape[2:]
    num_landmarks = batch_keypoints.shape[1]
    assert num_landmarks == batch_predicted_heatmaps.shape[1], f"Number of landmarks mismatch: {num_landmarks} vs {batch_predicted_heatmaps.shape[1]}"
    
    for i in range(len(batch_images)):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5)) 
        
        original_image = batch_images[i].transpose((1, 2, 0))
        
        # Ground truth
        keypoints = batch_keypoints[i]
        heatmap = fuse_heatmaps(batch_heatmaps[i])
        
        # Predictions
        predicted_heatmaps = batch_predicted_heatmaps[i]
        predicted_keypoints = np.array([extract_landmark(hm) for hm in predicted_heatmaps])
        predicted_heatmap = keypoints2heatmaps(predicted_keypoints, image_size, fuse=True)
        
        # Visualize the keypoints
        axs[0].set_title("Keypoints", fontsize=12, pad=5)  
        axs[0].imshow(original_image, cmap='gray')
        axs[0].scatter(keypoints[:, 0], keypoints[:, 1], c='green', s=0.25, label='Ground Truth', marker='o')
        axs[0].scatter(predicted_keypoints[:, 0], predicted_keypoints[:, 1], c='red', s=0.25, label='Prediction', marker='o')
        axs[0].legend(fontsize=8, loc='upper right')  
        
        # Visualize the heatmaps
        axs[1].set_title("Heatmaps", fontsize=12, pad=5)  
        axs[1].imshow(original_image, cmap='gray')
        axs[1].imshow(heatmap, cmap=cmap_green, alpha=1, label='Ground Truth')
        axs[1].imshow(predicted_heatmap, cmap=cmap_red, alpha=1, label='Prediction')
        
        # Visualize the predictions
        axs[2].set_title("Prediction", fontsize=12, pad=5)  
        axs[2].imshow(original_image, cmap='gray')        
        normalized_heatmaps = predicted_heatmaps / np.max(predicted_heatmaps, axis=(1, 2), keepdims=True)
        squashed_heatmaps = np.max(normalized_heatmaps, axis=0)
        #squashed_heatmaps = fuse_heatmaps(predicted_heatmaps)
        axs[2].imshow(squashed_heatmaps, cmap='inferno', alpha=0.4, label='Prediction')
        
        for ax in axs:
            ax.set_axis_off()
        
        plt.subplots_adjust(top=0.85, bottom=0.05, left=0.05, right=0.95, wspace=0.05) 
        fig.suptitle(f"Image: {batch_names[i]}", fontsize=14, y=0.98)  
        fig.savefig(os.path.join(save_dir, "plot_landmarks", f"{batch_names[i]}_landmarks.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
