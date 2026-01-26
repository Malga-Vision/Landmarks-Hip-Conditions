"""Deep learning training and evaluation utilities for landmark detection.

This module contains functions for training, validation, evaluation, and visualization
of deep learning models for anatomical landmark detection in medical images.
"""

import os
import json
import logging
import traceback

import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from heatmaps_utils import get_mode_probability, extract_landmark
from setup import two_d_softmax, CustomNLLLoss
from metrics import (
    compute_euclidean_distance, compute_mean_std, compute_sdr, calculate_ere,
    calculate_alpha_angle, calculate_lce_angle, draw_alpha_angle, draw_lce_angle,
    compute_angle_statistics, plot_combined_bland_altman, plot_bland_altman,
    visualize_hip_angles, visualize_hip_angles_mri
)


def extract_patient_visit(name):
    """Extract patient ID and visit from filename."""
    base = os.path.basename(str(name))
    parts = base.replace('.', '_').split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    else:
        return base


def visualise_grid(images, heatmap_stack, predicted_landmarks, ground_truth_landmarks, grid_distances, num_samples=8, cols=4, column_titles=None):
    """
    Visualizes a grid of images with predicted heatmaps and landmarks in two rows.
    First row shows heatmaps overlaid on images, second row shows landmarks with connections.
    
    Args:
        images: Batch of images (N, C, H, W)
        heatmap_stack: Predicted heatmaps (N, num_landmarks, H, W)
        predicted_landmarks: Predicted landmark positions (N, num_landmarks, 2) in pixel space
        ground_truth_landmarks: Ground truth landmarks (N, num_landmarks, 2) in pixel space
        grid_distances: List of lists containing distances in mm for each landmark (N samples, each with num_landmarks distances)
        num_samples: Number of samples to display
        cols: Number of columns in the grid
        column_titles: Optional list of titles for each column
    """
    num_samples = min(num_samples, images.shape[0])
    
    # Create figure with 2 rows: heatmaps and landmarks
    fig, axes = plt.subplots(2, cols, figsize=(4*cols, 8))
    if cols == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(min(num_samples, cols)):
        # First row: Heatmaps
        ax_heatmap = axes[0, i]
        image = images[i, 0]  # Assuming single channel
        ax_heatmap.imshow(image, cmap='gray')

        # Display heatmaps
        normalized_heatmaps = heatmap_stack[i] / np.max(heatmap_stack[i], axis=(1, 2), keepdims=True)
        squashed_heatmaps = np.max(normalized_heatmaps, axis=0)
        ax_heatmap.imshow(squashed_heatmaps, cmap='inferno', alpha=0.6)

        # Use patient+visit as column title if provided
        if column_titles is not None and i < len(column_titles):
            ax_heatmap.set_title(f'{column_titles[i]}', fontsize=10)
        else:
            ax_heatmap.set_title(f'Heatmaps - Sample {i+1}', fontsize=10)
        ax_heatmap.axis('off')

        # Second row: Landmarks
        ax_landmarks = axes[1, i]
        ax_landmarks.imshow(image, cmap='gray')

        # Get predicted and ground truth landmarks for this sample
        predicted_landmark_positions = predicted_landmarks[i]
        ground_truth_landmark_position = ground_truth_landmarks[i]

        # Separate predicted landmarks by error distance (threshold: 2mm)
        threshold = 2.0  # mm
        good_predictions = []
        bad_predictions = []
        
        for j in range(len(grid_distances[i])):
            distance = grid_distances[i][j]
            if distance <= threshold:
                good_predictions.append(predicted_landmark_positions[j])
            else:
                bad_predictions.append(predicted_landmark_positions[j])
        
        good_predictions = np.array(good_predictions)
        bad_predictions = np.array(bad_predictions)

        # Plot ground truth landmarks in blue (zorder=1, bottom layer)
        ax_landmarks.scatter(ground_truth_landmark_position[:, 0], ground_truth_landmark_position[:, 1], 
                           color='blue', s=20, label='Ground Truth', zorder=1)
        
        # Plot good predictions (≤2mm) in green (zorder=2)
        if len(good_predictions) > 0:
            ax_landmarks.scatter(good_predictions[:, 0], good_predictions[:, 1], 
                               color='green', s=20, label='Pred (≤2mm)', zorder=2)
        
        # Plot bad predictions (>2mm) in red (zorder=2)
        if len(bad_predictions) > 0:
            ax_landmarks.scatter(bad_predictions[:, 0], bad_predictions[:, 1], 
                               color='red', s=20, label='Pred (>2mm)', zorder=2)
        
        # Draw yellow connection lines between ground truth and predictions (zorder=3, top layer)
        for j in range(len(predicted_landmark_positions)):
            ax_landmarks.plot([ground_truth_landmark_position[j, 0], predicted_landmark_positions[j, 0]],
                            [ground_truth_landmark_position[j, 1], predicted_landmark_positions[j, 1]],
                            color='yellow', alpha=0.6, linewidth=1.5, zorder=3)

        if column_titles is not None and i < len(column_titles):
            ax_landmarks.set_title(f'{column_titles[i]}', fontsize=10)
        else:
            ax_landmarks.set_title(f'Landmarks - Sample {i+1}', fontsize=10)
        ax_landmarks.axis('off')

        # Add legend only to the first subplot of landmarks row
        if i == 0:
            ax_landmarks.legend(fontsize=8, loc='upper right')
    
    # Hide empty subplots if num_samples < cols
    for i in range(min(num_samples, cols), cols):
        axes[0, i].axis('off')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig


def save_grid_results(output_dir, images, heatmaps, predicted_landmarks, ground_truth_landmarks, image_names, num_landmarks=4):
    """
    Save a 1x3 grid per sample containing: heatmap overlay, alpha-angle visualization and LCE visualization.
    Uses the same drawing functions as the angle_visualizations to ensure consistent style.

    Args:
        output_dir: directory to save images
        images: numpy array (N, C, H, W)
        heatmaps: numpy array (N, num_landmarks, H, W)
        predicted_landmarks: array-like (N, num_landmarks, 2)
        ground_truth_landmarks: array-like (N, num_landmarks, 2)
        image_names: list of image identifiers
        num_landmarks: number of landmarks (4 for MRI, 8 for X-ray)
    """
    os.makedirs(output_dir, exist_ok=True)

    N = images.shape[0]
    for i in range(N):
        try:
            img = images[i, 0]
            hm = heatmaps[i]
            # normalize heatmaps per-channel then squashed overlay
            denom = np.max(hm, axis=(1, 2), keepdims=True) + 1e-12
            normalized = hm / denom
            squashed = np.max(normalized, axis=0)

            pred_lm = np.array(predicted_landmarks[i])
            gt_lm = np.array(ground_truth_landmarks[i])

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Heatmap overlay
            ax0 = axes[0]
            ax0.imshow(img, cmap='gray')
            ax0.imshow(squashed, cmap='inferno', alpha=0.6)
            ax0.set_title('Heatmaps', fontsize=12, fontweight='bold')
            ax0.axis('off')

            # Alpha angle visualization (middle)
            ax1 = axes[1]
            ax1.imshow(img, cmap='gray')
            
            if num_landmarks == 8:
                # X-ray: right hip uses indices 1,2,3 ; left hip uses 5,6,7
                try:
                    r_pred_alpha = calculate_alpha_angle(pred_lm[1], pred_lm[2], pred_lm[3])
                    r_gt_alpha = calculate_alpha_angle(gt_lm[1], gt_lm[2], gt_lm[3])
                    
                    # Draw GT (red)
                    draw_alpha_angle(ax1, gt_lm[1], gt_lm[2], gt_lm[3], color='red', linewidth=2, label='Ground Truth')
                    # Draw Pred (lime)
                    draw_alpha_angle(ax1, pred_lm[1], pred_lm[2], pred_lm[3], color='lime', linewidth=2, label='Predicted')
                    
                    ax1.set_title(f'Alpha Angle - Right Hip\nPredicted: {r_pred_alpha:.1f}° | GT: {r_gt_alpha:.1f}°',
                                fontsize=12, fontweight='bold')
                except Exception:
                    pass
                
                # Left hip if available
                try:
                    l_pred_alpha = calculate_alpha_angle(pred_lm[5], pred_lm[6], pred_lm[7])
                    l_gt_alpha = calculate_alpha_angle(gt_lm[5], gt_lm[6], gt_lm[7])
                    
                    # Draw left hip angles (without label to avoid duplicate legend)
                    draw_alpha_angle(ax1, gt_lm[5], gt_lm[6], gt_lm[7], color='red', linewidth=2)
                    draw_alpha_angle(ax1, pred_lm[5], pred_lm[6], pred_lm[7], color='lime', linewidth=2)
                    
                    # Update title to include left hip
                    ax1.set_title(f'Alpha Angles\nRight - Pred: {r_pred_alpha:.1f}° | GT: {r_gt_alpha:.1f}°\n' +
                                f'Left - Pred: {l_pred_alpha:.1f}° | GT: {l_gt_alpha:.1f}°',
                                fontsize=11, fontweight='bold')
                except Exception:
                    pass
            else:
                # MRI: right hip only, indices 1,2,3 for alpha
                try:
                    pred_alpha = calculate_alpha_angle(pred_lm[1], pred_lm[2], pred_lm[3])
                    gt_alpha = calculate_alpha_angle(gt_lm[1], gt_lm[2], gt_lm[3])
                    
                    # Draw GT (red)
                    draw_alpha_angle(ax1, gt_lm[1], gt_lm[2], gt_lm[3], color='red', linewidth=2, label='Ground Truth')
                    # Draw Pred (lime)
                    draw_alpha_angle(ax1, pred_lm[1], pred_lm[2], pred_lm[3], color='lime', linewidth=2, label='Predicted')
                    
                    ax1.set_title(f'Alpha Angle - Right Hip\nPredicted: {pred_alpha:.1f}° | GT: {gt_alpha:.1f}°',
                                fontsize=12, fontweight='bold')
                except Exception:
                    pass
            
            ax1.axis('off')
            ax1.legend(loc='upper right', fontsize=10)

            # LCE visualization (rightmost)
            ax2 = axes[2]
            ax2.imshow(img, cmap='gray')
            
            if num_landmarks == 8:
                # X-ray: right hip
                try:
                    r_pred_lce = calculate_lce_angle(pred_lm[1], pred_lm[0])
                    r_gt_lce = calculate_lce_angle(gt_lm[1], gt_lm[0])
                    
                    # Draw GT (red)
                    draw_lce_angle(ax2, gt_lm[1], gt_lm[0], color='red', linewidth=2, label='Ground Truth')
                    # Draw Pred (lime)
                    draw_lce_angle(ax2, pred_lm[1], pred_lm[0], color='lime', linewidth=2, label='Predicted')
                    
                    ax2.set_title(f'LCE Angle - Right Hip\nPredicted: {r_pred_lce:.1f}° | GT: {r_gt_lce:.1f}°',
                                fontsize=12, fontweight='bold')
                except Exception:
                    pass
                
                # Left hip
                try:
                    l_pred_lce = calculate_lce_angle(pred_lm[5], pred_lm[4])
                    l_gt_lce = calculate_lce_angle(gt_lm[5], gt_lm[4])
                    
                    # Draw left hip angles
                    draw_lce_angle(ax2, gt_lm[5], gt_lm[4], color='red', linewidth=2)
                    draw_lce_angle(ax2, pred_lm[5], pred_lm[4], color='lime', linewidth=2)
                    
                    # Update title to include left hip
                    ax2.set_title(f'LCE Angles\nRight - Pred: {r_pred_lce:.1f}° | GT: {r_gt_lce:.1f}°\n' +
                                f'Left - Pred: {l_pred_lce:.1f}° | GT: {l_gt_lce:.1f}°',
                                fontsize=11, fontweight='bold')
                except Exception:
                    pass
            else:
                # MRI: right hip only
                try:
                    pred_lce = calculate_lce_angle(pred_lm[1], pred_lm[0])
                    gt_lce = calculate_lce_angle(gt_lm[1], gt_lm[0])
                    
                    # Draw GT (red)
                    draw_lce_angle(ax2, gt_lm[1], gt_lm[0], color='red', linewidth=2, label='Ground Truth')
                    # Draw Pred (lime)
                    draw_lce_angle(ax2, pred_lm[1], pred_lm[0], color='lime', linewidth=2, label='Predicted')
                    
                    ax2.set_title(f'LCE Angle - Right Hip\nPredicted: {pred_lce:.1f}° | GT: {gt_lce:.1f}°',
                                fontsize=12, fontweight='bold')
                except Exception:
                    pass
            
            ax2.axis('off')
            ax2.legend(loc='upper right', fontsize=10)

            plt.tight_layout()
            base_name = os.path.basename(str(image_names[i]))
            save_name = os.path.join(output_dir, f"grid_result_{i+1}_{base_name}.png")
            fig.savefig(save_name, dpi=300, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            # don't fail whole evaluation if saving one sample fails
            try:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed saving grid_result for sample {i}: {e}")
            except Exception:
                pass
            continue


## -----------------------------------------------------------------------------------------------------------------##
##                                               EARLY STOPPING                                                     ##
## -----------------------------------------------------------------------------------------------------------------##


class EarlyStopping:
    """Early stops training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, delta=0, save_dir=None, verbose=True, filename=None):
        self.patience = patience
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.save_dir = save_dir
        self.verbose = verbose
        self.filename = filename

    def __call__(self, val_loss, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, epoch):

        if self.best_val_loss is None or val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            self.val_loss_min = val_loss
            
            # Save the best model
            save_model(
                save_path=self.save_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=loss_fn,
                results=results,
                epochs_without_improvement=self.counter,
                best_val_loss=val_loss,
                epoch=epoch,
                called_by_early_stopping=True
            )
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True


def save_model(save_path, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, best_val_loss, epoch, called_by_early_stopping=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if called_by_early_stopping:
        checkpoint_path = os.path.join(save_path, "best_checkpoint.pt")
    else:
        checkpoint_path = os.path.join(save_path, f"checkpoint_epoch{epoch}.pt")

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_fn': loss_fn.state_dict(),
        'results': results,
        'epochs_without_improvement': epochs_without_improvement,
        'best_val_loss': best_val_loss,
        'epoch': epoch
    }, checkpoint_path)

    
def load_model(load_path, model, optimizer, scheduler, loss_fn, device):
    checkpoint = torch.load(load_path, map_location=torch.device(device), weights_only=False)

    if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)  # Move the model to the specified device

    if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if 'loss_fn' in checkpoint: loss_fn.load_state_dict(checkpoint['loss_fn'])

    # Load other values only if they exist in the checkpoint
    start_epoch = checkpoint.get('epoch', 0) + 1
    results = checkpoint.get('results', None)
    epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
    best_val_loss = checkpoint.get('best_val_loss', None)
    print(f"Model loaded from {load_path} | Starting from epoch {start_epoch} | Best validation loss: {best_val_loss} | Epochs without improvement: {epochs_without_improvement}")
    del checkpoint
    return model, optimizer, scheduler, loss_fn, start_epoch, results, epochs_without_improvement, best_val_loss


def train_step(model: torch.nn.Module,
               device: torch.device,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               gradient_accumulation_steps: int = 1):
    
    model.train()
    train_loss = 0.0

    for batch, data in enumerate(dataloader):
        X = data['image'].to(device)
        y = data['heatmaps'].to(device)

        y_pred = model(X)
        
        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss = loss_fn(y_pred, y)
        elif isinstance(loss_fn, (CustomNLLLoss)):
            y_pred = two_d_softmax(y_pred)
            loss = loss_fn(y_pred, y)
        else:
            raise ValueError(f"Unsupported loss function {type(loss_fn).__name__}")

        loss = loss / gradient_accumulation_steps
        loss.backward()
        train_loss += loss.item()

        if np.isnan(loss.item()) or np.isinf(loss.item()):
            print(f"Gradient explosion detected in batch {batch + 1} | Loss: {loss.item()} | Applying gradient clipping...")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if ((batch + 1) % gradient_accumulation_steps == 0) or (batch + 1 == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

        del X, y, y_pred, loss, data

    train_loss /= len(dataloader)
    return train_loss


def validate_step(model: torch.nn.Module,
                  device: torch.device,
                  dataloader: torch.utils.data.DataLoader,
                  loss_fn: torch.nn.Module):
    model.eval()
    val_loss = 0.0
    all_radial_errors = []
    all_expected_radial_errors = []
    
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            X = data['image'].to(device)
            y = data['heatmaps'].to(device)
            gt_landmarks = data['landmarks']
            scaling_factors = data['physical_scaling_factor']

            y_pred = model(X)
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(y_pred, y)
            elif isinstance(loss_fn, (CustomNLLLoss)):
                y_pred = two_d_softmax(y_pred)
                loss = loss_fn(y_pred, y)
            else:
                raise ValueError(f"Unsupported loss function {type(loss_fn).__name__}")

            # Compute radial errors for each landmark            
            for i in range(y_pred.shape[0]):
                for j in range(y_pred.shape[1]):
                    pred_landmark = extract_landmark(y_pred[i, j].detach().cpu().numpy())
                    gt_landmark = gt_landmarks[i, j].detach().cpu().numpy()
                    scale = scaling_factors[i].detach().cpu().numpy()

                    distance = compute_euclidean_distance(pred_landmark * scale, gt_landmark * scale)
                    #expected_distance = calculate_ere(y_pred[i, j].detach().cpu().numpy(), pred_landmark * scale, scale)
                    expected_distance = 0
                    all_radial_errors.append(distance)
                    all_expected_radial_errors.append(expected_distance)

            val_loss += loss.item()
            del X, y, y_pred, loss, gt_landmarks, scaling_factors, pred_landmark, gt_landmark, scale

    val_loss /= len(dataloader)
    mre, _ = compute_mean_std(all_radial_errors)
    ere, _ = compute_mean_std(all_expected_radial_errors)

    return val_loss, mre, ere
## -----------------------------------------------------------------------------------------------------------------##
##                                           TRAINING + VALIDATION PART                                             ##
## -----------------------------------------------------------------------------------------------------------------##

def train_and_validate(model: torch.nn.Module,
                       device: torch.device,
                       train_dataloader: torch.utils.data.DataLoader,
                       val_dataloader: torch.utils.data.DataLoader,
                       optimizer: torch.optim.Optimizer,
                       scheduler: torch.optim.lr_scheduler,
                       loss_fn: torch.nn.Module,
                       epochs: int = 10,
                       save_path: str = None,
                       patience: int = 10,
                       gradient_accumulation_steps: int = 1,
                       continue_training: bool = False,
                       logger: logging.Logger = None):
    
    if continue_training:
        model_path = os.path.join(save_path, "best_checkpoint.pt")
        model, optimizer, scheduler, loss_fn, start_epoch, results, epochs_without_improvement, best_val_loss = load_model(model_path, model, optimizer, scheduler, loss_fn, device)
    else:
        results = {"train_loss": [], "val_loss": [], "val_mre": [], "val_ere": []}
        start_epoch = 1
        best_val_loss = float("inf")
        epochs_without_improvement = 0

    # Create EarlyStopping instance
    early_stopping = EarlyStopping(patience=patience, save_dir=save_path, verbose=False)
    early_stopping.counter = epochs_without_improvement
    early_stopping.best_val_loss = best_val_loss
    model = model.to(device)

    # Create tqdm progress bar for epochs
    epoch_pbar = tqdm(range(start_epoch, epochs + 1), desc="Training")
    
    # Loop through training and validating steps for a number of epochs
    for epoch in epoch_pbar:

        train_loss = train_step(model, device, train_dataloader, loss_fn, optimizer, gradient_accumulation_steps)
        val_loss, val_mre, val_ere = validate_step(model, device, val_dataloader, loss_fn)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        #logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.7f} | Validation Loss: {val_loss:.7f} | Validation MRE: {val_mre:.2f} | Validation ERE: {val_ere:.2f}")
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["val_mre"].append(val_mre)
        results["val_ere"].append(val_ere)
        
        # Update tqdm progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.6f}',
            'Val Loss': f'{val_loss:.6f}',
            'MRE': f'{val_mre:.3f}',
            #'ERE': f'{val_ere:.3f}',
            'LR': optimizer.param_groups[0]['lr']
        })
        # Check for early stopping
        early_stopping(val_loss, model, optimizer, scheduler, loss_fn, results, epochs_without_improvement, epoch)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    del epoch_pbar, early_stopping, train_loss, val_loss, val_mre, model, optimizer, scheduler, loss_fn, device, train_dataloader, val_dataloader

    return results



## -----------------------------------------------------------------------------------------------------------------##
##                                         TEST-TIME AUGMENTATION FUNCTIONS                                         ##
## -----------------------------------------------------------------------------------------------------------------##

def apply_tta_transforms(image_tensor, landmarks_tensor=None):
    """
    Apply test-time augmentation transforms to an image tensor.
    These transforms mirror the training augmentations for better consistency.
    
    Args:
        image_tensor: Input image tensor of shape (C, H, W), normalized in [0, 1] range
        landmarks_tensor: Optional landmarks tensor of shape (num_landmarks, 2) in pixel coordinates
        
    Returns:
        List of tuples (augmented_image, transform_info) where transform_info contains
        the type of transform and parameters needed for inverse transformation
    """
    augmented_samples = []
    C, H, W = image_tensor.shape
    
    # Original image (always included)
    augmented_samples.append((image_tensor, {'type': 'original'}))
    
    # Horizontal flip (mirrors training flip probability)
    #flipped_img = TF.hflip(image_tensor)
    #augmented_samples.append((flipped_img, {'type': 'hflip', 'width': W}))
    
    # Translations: ±5 pixels in x and y (matches training range of ±5%)
    # Note: Using affine transform for translation since TF.translate doesn't exist
    for dx in [-5, 5]:
        for dy in [-5, 5]:
            # affine(img, angle, translate, scale, shear)
            translated_img = TF.affine(image_tensor, angle=0, translate=[dx, dy], scale=1.0, shear=0)
            augmented_samples.append((translated_img, {'type': 'translate', 'dx': dx, 'dy': dy}))

    # Rotation augmentations: -10, -5, +5, +10 degrees (matches training range of ±10°)
    # Using affine for consistency
    for angle in [-10, -5, 5, 10]:
        rotated_img = TF.affine(image_tensor, angle=angle, translate=[0, 0], scale=1.0, shear=0,
                                interpolation=TF.InterpolationMode.BILINEAR)
        augmented_samples.append((rotated_img, {'type': 'rotate', 'angle': angle, 'center': (W/2, H/2)}))
    
    # Scale variations: 0.95x and 1.05x (matches training scale range of 0.95-1.05)
    # Using affine for consistency - note: affine scale is around center
    for scale in [0.95, 1.05]:
        scaled_img = TF.affine(image_tensor, angle=0, translate=[0, 0], scale=scale, shear=0,
                              interpolation=TF.InterpolationMode.BILINEAR)
        augmented_samples.append((scaled_img, {'type': 'scale', 'scale': scale, 'orig_size': (H, W)}))
    
    # Brightness adjustments (mirrors training brightness/contrast)
    # Slight brightness variations: ±10%
    for brightness_factor in [0.9, 1.1]:
        brightened_img = TF.adjust_brightness(image_tensor, brightness_factor)
        # Clamp to valid range [0, 1]
        brightened_img = torch.clamp(brightened_img, 0, 1)
        augmented_samples.append((brightened_img, {'type': 'brightness', 'factor': brightness_factor}))
    
    # Contrast adjustments (mirrors training contrast augmentation)
    # Slight contrast variations: ±10%
    for contrast_factor in [0.9, 1.1]:
        contrasted_img = TF.adjust_contrast(image_tensor, contrast_factor)
        # Clamp to valid range [0, 1]
        contrasted_img = torch.clamp(contrasted_img, 0, 1)
        augmented_samples.append((contrasted_img, {'type': 'contrast', 'factor': contrast_factor}))
    
    # Gamma adjustments (mirrors training gamma augmentation)
    # Gamma values: 0.9 and 1.1 (conservative, within training range)
    for gamma in [0.9, 1.1]:
        gamma_corrected = TF.adjust_gamma(image_tensor, gamma)
        # Clamp to valid range [0, 1]
        gamma_corrected = torch.clamp(gamma_corrected, 0, 1)
        augmented_samples.append((gamma_corrected, {'type': 'gamma', 'gamma': gamma}))
    
    return augmented_samples


def inverse_transform_heatmaps(heatmaps, transform_info):
    """
    Apply inverse transformation to heatmaps to bring them back to original image space.
    
    Args:
        heatmaps: Predicted heatmaps tensor of shape (num_landmarks, H, W)
        transform_info: Dictionary containing transform type and parameters
        
    Returns:
        Transformed heatmaps in original image space
    """
    transform_type = transform_info['type']
    
    if transform_type == 'original':
        return heatmaps
    
    elif transform_type == 'hflip':
        # Flip heatmaps back horizontally
        return torch.flip(heatmaps, dims=[2])

    elif transform_type == 'translate':
        # Translate heatmaps back by negative offset
        dx = -transform_info['dx']
        dy = -transform_info['dy']
        return TF.affine(heatmaps, angle=0, translate=[dx, dy], scale=1.0, shear=0)

    elif transform_type == 'rotate':
        # Rotate back by negative angle using affine
        angle = -transform_info['angle']
        return TF.affine(heatmaps, angle=angle, translate=[0, 0], scale=1.0, shear=0,
                        interpolation=TF.InterpolationMode.BILINEAR)
    
    elif transform_type == 'scale':
        # Scale back using inverse scale with affine
        scale = transform_info['scale']
        inverse_scale = 1.0 / scale
        return TF.affine(heatmaps, angle=0, translate=[0, 0], scale=inverse_scale, shear=0,
                        interpolation=TF.InterpolationMode.BILINEAR)
    
    elif transform_type in ['brightness', 'contrast', 'gamma']:
        # Intensity transforms don't affect heatmap spatial structure
        # Heatmaps are predictions, not affected by input intensity changes
        return heatmaps
    
    return heatmaps


def inverse_transform_landmarks(landmarks, transform_info):
    """
    Apply inverse transformation to landmarks to bring them back to original image space.
    
    Args:
        landmarks: Predicted landmarks array of shape (num_landmarks, 2) in pixel coordinates
        transform_info: Dictionary containing transform type and parameters
        
    Returns:
        Transformed landmarks in original image space
    """
    transform_type = transform_info['type']
    landmarks = landmarks.copy()
    
    if transform_type == 'original':
        return landmarks
    
    elif transform_type == 'hflip':
        # Flip x-coordinates
        width = transform_info['width']
        landmarks[:, 0] = width - landmarks[:, 0]
        return landmarks
    
    elif transform_type == 'translate':
        # Translate landmarks back
        dx = -transform_info['dx']
        dy = -transform_info['dy']
        landmarks[:, 0] += dx
        landmarks[:, 1] += dy
        return landmarks
    
    elif transform_type == 'rotate':
        # Rotate landmarks back
        angle = -transform_info['angle']
        center = transform_info['center']
        angle_rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Translate to origin
        landmarks_centered = landmarks - center
        
        # Rotate
        x_new = landmarks_centered[:, 0] * cos_a - landmarks_centered[:, 1] * sin_a
        y_new = landmarks_centered[:, 0] * sin_a + landmarks_centered[:, 1] * cos_a
        
        # Translate back
        landmarks[:, 0] = x_new + center[0]
        landmarks[:, 1] = y_new + center[1]
        
        return landmarks
    
    elif transform_type == 'scale':
        # With affine, scale is around center, so we just inverse scale from center
        scale = transform_info['scale']
        H, W = transform_info['orig_size']
        center = np.array([W/2, H/2])
        
        # Translate to origin, inverse scale, translate back
        landmarks_centered = landmarks - center
        landmarks_centered = landmarks_centered / scale
        landmarks = landmarks_centered + center
        
        return landmarks
    
    elif transform_type in ['brightness', 'contrast', 'gamma']:
        # Intensity transforms don't affect landmark spatial positions
        return landmarks
    
    return landmarks


## -----------------------------------------------------------------------------------------------------------------##
##                                                  EVALUATION PART                                                 ##
## -----------------------------------------------------------------------------------------------------------------##
def evaluate(model: torch.nn.Module,
                        device: torch.device,
                        test_dataloader: torch.utils.data.DataLoader,
                        loss_fn: torch.nn.Module,
                        num_landmarks: int,
                        output_path: str,
                        logger: logging.Logger = None,
                        visualize_angles: bool = True,
                        use_tta: bool = False):
    """
    Enhanced evaluation function that includes hip angle calculations (alpha and LCE angles).
    
    Args:
        model: PyTorch model to evaluate
        device: Device to run evaluation on
        test_dataloader: DataLoader for test data
        loss_fn: Loss function
        num_landmarks: Number of landmarks (should be 8 for hip X-rays)
        output_path: Path to save results
        logger: Logger for output
        visualize_angles: Whether to save angle visualizations
        use_tta: Whether to use test-time augmentation (default: False)
        
    Returns:
        Dictionary containing all evaluation metrics including angles
    """
    tta_status = "WITH TTA" if use_tta else "WITHOUT TTA"
    print(f"\nSTART - EVALUATE FUNCTION WITH ANGLES ({tta_status})")
    if use_tta:
        logger.info("Test-Time Augmentation (TTA) is ENABLED")
        logger.info("TTA will apply 19 augmentations per image (mirroring training augmentations)")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "angle_visualizations"), exist_ok=True)
    model = model.to(device)
    model.eval()
    
    # Log dataset type info
    dataset_type = "X-ray (bilateral, 8 landmarks)" if num_landmarks == 8 else "MRI (right hip only, 4 landmarks)"
    logger.info(f"Evaluating {dataset_type}")
    
    # Metrics tracking
    metrics = {
        'losses': [],
        'radial_errors': [],
        'expected_radial_errors': [],
        'mode_probabilities': [],
        'pixel_size': [],
        # Angle-specific metrics
        'predicted_landmarks_all': [],
        'ground_truth_landmarks_all': [],
        'image_names': []
    }
    
    # Store data for grid visualization
    grid_viz_data = {
        'images': [],
        'heatmaps': [],
        'ground_truth_landmarks': [],  # Will store in pixel space
        'predicted_landmarks': [],      # Will store in pixel space
        'distances': [],                # Will store distances in mm per sample
    }
    
    pbar = tqdm(test_dataloader, desc="Evaluating", unit="batch")
    sigma = test_dataloader.dataset.sigma
    use_argmax = False if sigma > 1 else True
    
    with torch.no_grad():
        for data in pbar:
            try:
                images_name = data['name']
                X = data['image'].to(device)
                y = data['heatmaps'].to(device)
                ground_truth_landmarks = data['landmarks']
                scaling_factors = data['physical_scaling_factor']
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

            if use_tta:
                # Test-Time Augmentation: Process each image in the batch separately
                batch_size = X.shape[0]
                y_pred_tta = []
                
                for i in range(batch_size):
                    single_image = X[i]  # Shape: (C, H, W)
                    
                    # Apply TTA transforms
                    augmented_samples = apply_tta_transforms(single_image)
                    
                    # Run inference on all augmented samples
                    tta_predictions = []
                    for aug_img, transform_info in augmented_samples:
                        # Add batch dimension and move to device
                        aug_img_batch = aug_img.unsqueeze(0).to(device)
                        
                        # Get prediction
                        aug_pred = model(aug_img_batch)
                        
                        # Apply softmax if needed
                        if isinstance(loss_fn, (CustomNLLLoss)):
                            aug_pred = two_d_softmax(aug_pred)
                        
                        # Remove batch dimension
                        aug_pred = aug_pred.squeeze(0)  # Shape: (num_landmarks, H, W)
                        
                        # Transform heatmaps back to original space
                        aug_pred_transformed = inverse_transform_heatmaps(aug_pred, transform_info)
                        tta_predictions.append(aug_pred_transformed)
                    
                    # Average all TTA predictions
                    avg_pred = torch.stack(tta_predictions).mean(dim=0)
                    y_pred_tta.append(avg_pred)
                
                # Stack back into batch
                y_pred = torch.stack(y_pred_tta)
            else:
                # Standard inference without TTA
                y_pred = model(X)
                if isinstance(loss_fn, (CustomNLLLoss)):
                    y_pred = two_d_softmax(y_pred)
            
            # Compute loss (always on the final averaged predictions)
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss = loss_fn(y_pred, y)
            elif isinstance(loss_fn, (CustomNLLLoss)):
                # y_pred already has softmax applied if using these losses
                loss = loss_fn(y_pred, y)
            else:
                raise ValueError(f"Unsupported loss function {type(loss_fn).__name__}")
            
            metrics['losses'].append(loss.item())
            
            # Move tensors to CPU and convert to numpy for metric calculations
            images_np = X.cpu().numpy()
            predicted_heatmaps_np = y_pred.cpu().numpy()
            ground_truth_heatmaps_np = y.cpu().numpy()
            ground_truth_landmarks_np = ground_truth_landmarks.cpu().numpy()
            physical_scaling_factor_np = scaling_factors.cpu().numpy()
            
            # Collect data for grid visualization from all batches
            if len(grid_viz_data['images']) == 0:
                grid_viz_data['images'] = images_np
                grid_viz_data['heatmaps'] = predicted_heatmaps_np
            else:
                grid_viz_data['images'] = np.concatenate([grid_viz_data['images'], images_np], axis=0)
                grid_viz_data['heatmaps'] = np.concatenate([grid_viz_data['heatmaps'], predicted_heatmaps_np], axis=0)

            # Compute metrics for each sample
            for i in range(predicted_heatmaps_np.shape[0]):
                predicted_landmarks = np.zeros((num_landmarks, 2), dtype=np.float64)
                sample_distances = []  # Store distances for all landmarks in this sample
                
                # Extract predicted landmarks
                for j in range(num_landmarks):
                    predicted_landmark = extract_landmark(predicted_heatmaps_np[i, j], use_argmax=use_argmax)
                    predicted_landmarks[j] = predicted_landmark
                    
                    target_point = ground_truth_landmarks_np[i, j]
                    scale_factor = physical_scaling_factor_np[i]
    
                    predicted_landmark_scaled = predicted_landmark * scale_factor
                    target_point_scaled = target_point * scale_factor
                    
                    radial_error = compute_euclidean_distance(predicted_landmark_scaled, target_point_scaled)
                    expected_radial_error = calculate_ere(predicted_heatmaps_np[i, j], predicted_landmark_scaled, scale_factor)
                    mode_prob = get_mode_probability(predicted_heatmaps_np[i, j])

                    # Store metrics
                    metrics['radial_errors'].append(radial_error)
                    metrics['expected_radial_errors'].append(expected_radial_error)
                    metrics['mode_probabilities'].append(mode_prob)
                    metrics['pixel_size'].append(scale_factor)
                    
                    # Collect distance for this landmark
                    sample_distances.append(radial_error)
                
                # Store data for grid visualization (all in pixel space for visualization)
                grid_viz_data['ground_truth_landmarks'].append(ground_truth_landmarks_np[i])  # Already in pixel space
                grid_viz_data['predicted_landmarks'].append(predicted_landmarks)  # In pixel space
                grid_viz_data['distances'].append(sample_distances)  # Distances in mm
                
                # Store landmarks for angle calculation
                metrics['predicted_landmarks_all'].append(predicted_landmarks)
                metrics['ground_truth_landmarks_all'].append(ground_truth_landmarks_np[i])
                metrics['image_names'].append(images_name[i] if isinstance(images_name, list) else images_name)
                                    
            # Update tqdm progress bar
            pbar.set_postfix({
                'Loss': f"{np.mean(metrics['losses']):.4f}",
                'MRE': f"{np.mean(metrics['radial_errors']):.4f}",
            })
    
    pbar.close()
    
    # ========================================
    # GENERATE VISUALIZATION GRID
    # ========================================
    try:
        logger.info("Generating visualization grids for all inference images...")
        images = grid_viz_data['images']
        heatmaps = grid_viz_data['heatmaps']
        # Convert lists to numpy arrays for proper indexing
        ground_truth_landmarks = np.array(grid_viz_data['ground_truth_landmarks'])  # Shape: (N, num_landmarks, 2)
        predicted_landmarks = np.array(grid_viz_data['predicted_landmarks'])      # Shape: (N, num_landmarks, 2)
        distances = grid_viz_data['distances']  # List of lists: N samples, each with num_landmarks distances
        image_names = metrics['image_names']

        total = images.shape[0]
        cols = 4
        num_samples_per_grid = cols
        num_grids = (total + num_samples_per_grid - 1) // num_samples_per_grid
        for grid_idx in range(num_grids):
            start = grid_idx * num_samples_per_grid
            end = min(start + num_samples_per_grid, total)
            grid_images = images[start:end]
            grid_heatmaps = heatmaps[start:end]
            grid_predicted_landmarks = predicted_landmarks[start:end]
            grid_ground_truth_landmarks = ground_truth_landmarks[start:end]
            grid_distances = distances[start:end]
            grid_titles = [extract_patient_visit(n) for n in image_names[start:end]]
            fig = visualise_grid(
                grid_images,
                grid_heatmaps,
                grid_predicted_landmarks,
                grid_ground_truth_landmarks,
                grid_distances,
                num_samples=end-start,
                cols=cols,
                column_titles=grid_titles
            )
            viz_path = os.path.join(output_path, f"predictions_grid_{grid_idx+1}.png")
            fig.savefig(viz_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Visualization grid saved to: {viz_path}")
    except Exception as e:
        logger.warning(f"Failed to generate visualization grids: {e}")
        import traceback
        logger.warning(f"Traceback: {traceback.format_exc()}")
    
    # ========================================
    # GENERATE ANGLE VISUALIZATION GRIDS
    # ========================================
    if visualize_angles:
        try:
            logger.info("Generating angle visualization grids...")
            angle_viz_output = os.path.join(output_path, "angle_visualizations")
            os.makedirs(angle_viz_output, exist_ok=True)
            
            # Get the data needed for angle visualization
            images = grid_viz_data['images']
            predicted_landmarks_for_angles = np.array(metrics['predicted_landmarks_all'])
            ground_truth_landmarks_for_angles = np.array(metrics['ground_truth_landmarks_all'])
            image_names = metrics['image_names']

            # Select appropriate visualization function based on number of landmarks
            if num_landmarks == 8:
                viz_func = visualize_hip_angles
            elif num_landmarks == 4:
                viz_func = visualize_hip_angles_mri
            else:
                raise ValueError(f"Unsupported number of landmarks: {num_landmarks}")
            
            # Create grids of angle visualizations
            total = images.shape[0]
            cols = 4
            num_samples_per_grid = cols
            num_grids = (total + num_samples_per_grid - 1) // num_samples_per_grid
            
            for grid_idx in range(num_grids):
                start = grid_idx * num_samples_per_grid
                end = min(start + num_samples_per_grid, total)
                num_in_grid = end - start
                
                # Create figure with 2 rows: alpha angles (row 0) and LCE angles (row 1)
                fig, axes = plt.subplots(2, num_in_grid, figsize=(5*num_in_grid, 10))
                if num_in_grid == 1:
                    axes = axes.reshape(2, 1)
                
                for col_idx, sample_idx in enumerate(range(start, end)):
                    img = images[sample_idx, 0]  # Assuming single channel
                    pred_lm = predicted_landmarks_for_angles[sample_idx]
                    gt_lm = ground_truth_landmarks_for_angles[sample_idx]
                    title = extract_patient_visit(image_names[sample_idx])
                    
                    # Row 0: Alpha angles (GT and Pred overlaid)
                    ax_alpha = axes[0, col_idx]
                    ax_alpha.imshow(img, cmap='gray')
                    
                    if num_landmarks == 8:
                        # Calculate angles
                        pred_right_alpha = calculate_alpha_angle(pred_lm[1], pred_lm[2], pred_lm[3])
                        gt_right_alpha = calculate_alpha_angle(gt_lm[1], gt_lm[2], gt_lm[3])
                        pred_left_alpha = calculate_alpha_angle(pred_lm[5], pred_lm[6], pred_lm[7])
                        gt_left_alpha = calculate_alpha_angle(gt_lm[5], gt_lm[6], gt_lm[7])
                        
                        # Determine colors based on error threshold
                        right_alpha_diff = abs(gt_right_alpha - pred_right_alpha)
                        left_alpha_diff = abs(gt_left_alpha - pred_left_alpha)
                        right_color = 'red' if right_alpha_diff > 10 else 'green'
                        left_color = 'red' if left_alpha_diff > 10 else 'green'
                        
                        # Draw GT alpha angles in blue
                        ax_alpha.plot([gt_lm[2, 0], gt_lm[1, 0]], [gt_lm[2, 1], gt_lm[1, 1]],
                                    color='blue', linewidth=2, label='GT')
                        ax_alpha.plot([gt_lm[1, 0], gt_lm[3, 0]], [gt_lm[1, 1], gt_lm[3, 1]],
                                    color='blue', linewidth=2)
                        ax_alpha.plot([gt_lm[6, 0], gt_lm[5, 0]], [gt_lm[6, 1], gt_lm[5, 1]],
                                    color='blue', linewidth=2)
                        ax_alpha.plot([gt_lm[5, 0], gt_lm[7, 0]], [gt_lm[5, 1], gt_lm[7, 1]],
                                    color='blue', linewidth=2)
                        
                        # Draw GT landmarks in blue
                        ax_alpha.scatter([gt_lm[1, 0], gt_lm[2, 0], gt_lm[3, 0]], 
                                       [gt_lm[1, 1], gt_lm[2, 1], gt_lm[3, 1]], 
                                       color='blue', s=20, marker='o', zorder=10)
                        ax_alpha.scatter([gt_lm[5, 0], gt_lm[6, 0], gt_lm[7, 0]], 
                                       [gt_lm[5, 1], gt_lm[6, 1], gt_lm[7, 1]], 
                                       color='blue', s=20, marker='o', zorder=10)
                        
                        # Draw Pred alpha angles in red/green based on error
                        right_label = f'Pred Right (Δ>{10}°)' if right_alpha_diff > 10 else f'Pred Right (Δ≤{10}°)'
                        left_label = f'Pred Left (Δ>{10}°)' if left_alpha_diff > 10 else f'Pred Left (Δ≤{10}°)'
                        ax_alpha.plot([pred_lm[2, 0], pred_lm[1, 0]], [pred_lm[2, 1], pred_lm[1, 1]],
                                    color=right_color, linewidth=2, label=right_label)
                        ax_alpha.plot([pred_lm[1, 0], pred_lm[3, 0]], [pred_lm[1, 1], pred_lm[3, 1]],
                                    color=right_color, linewidth=2)
                        ax_alpha.plot([pred_lm[6, 0], pred_lm[5, 0]], [pred_lm[6, 1], pred_lm[5, 1]],
                                    color=left_color, linewidth=2, label=left_label)
                        ax_alpha.plot([pred_lm[5, 0], pred_lm[7, 0]], [pred_lm[5, 1], pred_lm[7, 1]],
                                    color=left_color, linewidth=2)
                        
                        # Draw Pred landmarks in red/green based on error
                        ax_alpha.scatter([pred_lm[1, 0], pred_lm[2, 0], pred_lm[3, 0]], 
                                       [pred_lm[1, 1], pred_lm[2, 1], pred_lm[3, 1]], 
                                       color=right_color, s=20, marker='o', zorder=11)
                        ax_alpha.scatter([pred_lm[5, 0], pred_lm[6, 0], pred_lm[7, 0]], 
                                       [pred_lm[5, 1], pred_lm[6, 1], pred_lm[7, 1]], 
                                       color=left_color, s=20, marker='o', zorder=11)
                        
                        ax_alpha.set_title(f'{title} - Alpha Angles\n' + 
                                         f'Right: GT={gt_right_alpha:.1f}° Pred={pred_right_alpha:.1f}° (Δ={right_alpha_diff:.1f}°)\n' +
                                         f'Left: GT={gt_left_alpha:.1f}° Pred={pred_left_alpha:.1f}° (Δ={left_alpha_diff:.1f}°)',
                                         fontsize=8)
                    else:  # MRI with 4 landmarks (right hip only)
                        pred_alpha = calculate_alpha_angle(pred_lm[1], pred_lm[2], pred_lm[3])
                        gt_alpha = calculate_alpha_angle(gt_lm[1], gt_lm[2], gt_lm[3])
                        
                        # Determine color based on error threshold
                        alpha_diff = abs(gt_alpha - pred_alpha)
                        pred_color = 'red' if alpha_diff > 10 else 'green'
                        
                        # Draw GT alpha angle in blue
                        ax_alpha.plot([gt_lm[2, 0], gt_lm[1, 0]], [gt_lm[2, 1], gt_lm[1, 1]],
                                    color='blue', linewidth=2, label='GT')
                        ax_alpha.plot([gt_lm[1, 0], gt_lm[3, 0]], [gt_lm[1, 1], gt_lm[3, 1]],
                                    color='blue', linewidth=2)
                        
                        # Draw GT landmarks in blue
                        ax_alpha.scatter([gt_lm[1, 0], gt_lm[2, 0], gt_lm[3, 0]], 
                                       [gt_lm[1, 1], gt_lm[2, 1], gt_lm[3, 1]], 
                                       color='blue', s=20, marker='o', zorder=10)
                        
                        # Draw Pred alpha angle in red/green
                        pred_label = f'Pred (Δ>{10}°)' if alpha_diff > 10 else f'Pred (Δ≤{10}°)'
                        ax_alpha.plot([pred_lm[2, 0], pred_lm[1, 0]], [pred_lm[2, 1], pred_lm[1, 1]],
                                    color=pred_color, linewidth=2, label=pred_label)
                        ax_alpha.plot([pred_lm[1, 0], pred_lm[3, 0]], [pred_lm[1, 1], pred_lm[3, 1]],
                                    color=pred_color, linewidth=2)
                        
                        # Draw Pred landmarks in red/green
                        ax_alpha.scatter([pred_lm[1, 0], pred_lm[2, 0], pred_lm[3, 0]], 
                                       [pred_lm[1, 1], pred_lm[2, 1], pred_lm[3, 1]], 
                                       color=pred_color, s=20, marker='o', zorder=11)
                        
                        ax_alpha.set_title(f'{title} - Alpha Angle\nGT={gt_alpha:.1f}° Pred={pred_alpha:.1f}° (Δ={alpha_diff:.1f}°)',
                                         fontsize=9)
                    
                    ax_alpha.axis('off')
                    ax_alpha.legend(fontsize=7, loc='upper right')
                    
                    # Row 1: LCE angles (GT and Pred overlaid)
                    ax_lce = axes[1, col_idx]
                    ax_lce.imshow(img, cmap='gray')
                    
                    if num_landmarks == 8:
                        # Calculate angles
                        pred_right_lce = calculate_lce_angle(pred_lm[1], pred_lm[0])
                        gt_right_lce = calculate_lce_angle(gt_lm[1], gt_lm[0])
                        pred_left_lce = calculate_lce_angle(pred_lm[5], pred_lm[4])
                        gt_left_lce = calculate_lce_angle(gt_lm[5], gt_lm[4])
                        
                        # Determine colors based on error threshold
                        right_lce_diff = abs(gt_right_lce - pred_right_lce)
                        left_lce_diff = abs(gt_left_lce - pred_left_lce)
                        right_color = 'red' if right_lce_diff > 10 else 'green'
                        left_color = 'red' if left_lce_diff > 10 else 'green'
                        
                        # Draw GT LCE angles in blue
                        # Vertical reference lines
                        ax_lce.plot([gt_lm[1, 0], gt_lm[1, 0]], 
                                  [gt_lm[1, 1], gt_lm[1, 1] - 50],
                                  color='blue', linewidth=2, linestyle='--', alpha=0.7)
                        # Line to acetabulum
                        ax_lce.plot([gt_lm[1, 0], gt_lm[0, 0]], [gt_lm[1, 1], gt_lm[0, 1]],
                                  color='blue', linewidth=2, label='GT')
                        
                        # Left hip
                        ax_lce.plot([gt_lm[5, 0], gt_lm[5, 0]], 
                                  [gt_lm[5, 1], gt_lm[5, 1] - 50],
                                  color='blue', linewidth=2, linestyle='--', alpha=0.7)
                        ax_lce.plot([gt_lm[5, 0], gt_lm[4, 0]], [gt_lm[5, 1], gt_lm[4, 1]],
                                  color='blue', linewidth=2)
                        
                        # Draw GT landmarks in blue
                        ax_lce.scatter([gt_lm[1, 0], gt_lm[0, 0]], 
                                     [gt_lm[1, 1], gt_lm[0, 1]], 
                                     color='blue', s=20, marker='o', zorder=10)
                        ax_lce.scatter([gt_lm[5, 0], gt_lm[4, 0]], 
                                     [gt_lm[5, 1], gt_lm[4, 1]], 
                                     color='blue', s=20, marker='o', zorder=10)
                        
                        # Draw Pred LCE angles in red/green based on error
                        # Vertical reference lines
                        right_label = f'Pred Right (Δ>{10}°)' if right_lce_diff > 10 else f'Pred Right (Δ≤{10}°)'
                        left_label = f'Pred Left (Δ>{10}°)' if left_lce_diff > 10 else f'Pred Left (Δ≤{10}°)'
                        ax_lce.plot([pred_lm[1, 0], pred_lm[1, 0]], 
                                  [pred_lm[1, 1], pred_lm[1, 1] - 50],
                                  color=right_color, linewidth=2, linestyle='--', alpha=0.7)
                        # Line to acetabulum
                        ax_lce.plot([pred_lm[1, 0], pred_lm[0, 0]], [pred_lm[1, 1], pred_lm[0, 1]],
                                  color=right_color, linewidth=2, label=right_label)
                        
                        # Left hip
                        ax_lce.plot([pred_lm[5, 0], pred_lm[5, 0]], 
                                  [pred_lm[5, 1], pred_lm[5, 1] - 50],
                                  color=left_color, linewidth=2, linestyle='--', alpha=0.7)
                        ax_lce.plot([pred_lm[5, 0], pred_lm[4, 0]], [pred_lm[5, 1], pred_lm[4, 1]],
                                  color=left_color, linewidth=2, label=left_label)
                        
                        # Draw Pred landmarks in red/green based on error
                        ax_lce.scatter([pred_lm[1, 0], pred_lm[0, 0]], 
                                     [pred_lm[1, 1], pred_lm[0, 1]], 
                                     color=right_color, s=20, marker='o', zorder=11)
                        ax_lce.scatter([pred_lm[5, 0], pred_lm[4, 0]], 
                                     [pred_lm[5, 1], pred_lm[4, 1]], 
                                     color=left_color, s=20, marker='o', zorder=11)
                        
                        ax_lce.set_title(f'{title} - LCE Angles\n' + 
                                       f'Right: GT={gt_right_lce:.1f}° Pred={pred_right_lce:.1f}° (Δ={right_lce_diff:.1f}°)\n' +
                                       f'Left: GT={gt_left_lce:.1f}° Pred={pred_left_lce:.1f}° (Δ={left_lce_diff:.1f}°)',
                                       fontsize=8)
                    else:  # MRI
                        pred_lce = calculate_lce_angle(pred_lm[1], pred_lm[0])
                        gt_lce = calculate_lce_angle(gt_lm[1], gt_lm[0])
                        
                        # Determine color based on error threshold
                        lce_diff = abs(gt_lce - pred_lce)
                        pred_color = 'red' if lce_diff > 10 else 'green'
                        
                        # Draw GT LCE angle in blue
                        ax_lce.plot([gt_lm[1, 0], gt_lm[1, 0]], 
                                  [gt_lm[1, 1], gt_lm[1, 1] - 50],
                                  color='blue', linewidth=2, linestyle='--', alpha=0.7)
                        ax_lce.plot([gt_lm[1, 0], gt_lm[0, 0]], [gt_lm[1, 1], gt_lm[0, 1]],
                                  color='blue', linewidth=2, label='GT')
                        
                        # Draw GT landmarks in blue
                        ax_lce.scatter([gt_lm[1, 0], gt_lm[0, 0]], 
                                     [gt_lm[1, 1], gt_lm[0, 1]], 
                                     color='blue', s=20, marker='o', zorder=10)
                        
                        # Draw Pred LCE angle in red/green
                        pred_label = f'Pred (Δ>{10}°)' if lce_diff > 10 else f'Pred (Δ≤{10}°)'
                        ax_lce.plot([pred_lm[1, 0], pred_lm[1, 0]], 
                                  [pred_lm[1, 1], pred_lm[1, 1] - 50],
                                  color=pred_color, linewidth=2, linestyle='--', alpha=0.7)
                        ax_lce.plot([pred_lm[1, 0], pred_lm[0, 0]], [pred_lm[1, 1], pred_lm[0, 1]],
                                  color=pred_color, linewidth=2, label=pred_label)
                        
                        # Draw Pred landmarks in red/green
                        ax_lce.scatter([pred_lm[1, 0], pred_lm[0, 0]], 
                                     [pred_lm[1, 1], pred_lm[0, 1]], 
                                     color=pred_color, s=20, marker='o', zorder=11)
                        
                        ax_lce.set_title(f'{title} - LCE Angle\nGT={gt_lce:.1f}° Pred={pred_lce:.1f}° (Δ={lce_diff:.1f}°)',
                                       fontsize=9)
                    
                    ax_lce.axis('off')
                    ax_lce.legend(fontsize=7, loc='upper right')
                
                plt.tight_layout()
                viz_path = os.path.join(angle_viz_output, f"angles_grid_{grid_idx+1}.png")
                fig.savefig(viz_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Angle visualization grid {grid_idx+1}/{num_grids} saved to: {viz_path}")
                
        except Exception as e:
            logger.warning(f"Failed to generate angle visualization grids: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
    
    # Convert to numpy arrays
    radial_errors = np.array(metrics['radial_errors']).flatten()
    expected_radial_errors = np.array(metrics['expected_radial_errors']).flatten()
    mode_probabilities = np.array(metrics['mode_probabilities']).flatten()
    pixel_size = np.mean(metrics['pixel_size'])
    predicted_landmarks_all = np.array(metrics['predicted_landmarks_all'])  # (N, 8, 2)
    ground_truth_landmarks_all = np.array(metrics['ground_truth_landmarks_all'])  # (N, 8, 2)

    # Save combined grid results (heatmap + alpha + lce) per sample
    try:
        grid_results_dir = os.path.join(output_path, 'grid_results')
        save_grid_results(grid_results_dir, grid_viz_data['images'], grid_viz_data['heatmaps'],
                          np.array(grid_viz_data['predicted_landmarks']), np.array(grid_viz_data['ground_truth_landmarks']),
                          metrics['image_names'], num_landmarks=num_landmarks)
        logger.info(f"Saved grid_results to: {grid_results_dir}")
    except Exception as e:
        logger.warning(f"Failed to save grid_results: {e}")
        logger.warning(f"Traceback: {traceback.format_exc()}")
    
    # ========================================
    # CALCULATE HIP ANGLES WITH PROPER FILENAME TRACKING
    # ========================================
    logger.info("\n" + "="*70)
    logger.info("HIP ANGLE ANALYSIS")
    logger.info("="*70)
    
    # Check if we have X-ray data (8 landmarks) or MRI data (4 landmarks)
    has_both_hips = (num_landmarks == 8)
    
    # Calculate angles for all samples
    if has_both_hips:
        # X-ray dataset with both left and right hips
        pred_right_alpha, pred_left_alpha = calculate_alpha_angles_batch(predicted_landmarks_all)
        gt_right_alpha, gt_left_alpha = calculate_alpha_angles_batch(ground_truth_landmarks_all)
        pred_right_lce, pred_left_lce = calculate_lce_angles_batch(predicted_landmarks_all)
        gt_right_lce, gt_left_lce = calculate_lce_angles_batch(ground_truth_landmarks_all)
    else:
        # MRI dataset with only right hip (4 landmarks)
        pred_right_alpha = np.array([calculate_alpha_angle(
            femur_head_center=lm[1],
            neck_orientation_point=lm[2],
            lateral_cam_point=lm[3]
        ) for lm in predicted_landmarks_all])
        gt_right_alpha = np.array([calculate_alpha_angle(
            femur_head_center=lm[1],
            neck_orientation_point=lm[2],
            lateral_cam_point=lm[3]
        ) for lm in ground_truth_landmarks_all])
        pred_right_lce = np.array([calculate_lce_angle(
            femur_head_center=lm[1],
            lateral_ace_point=lm[0]
        ) for lm in predicted_landmarks_all])
        gt_right_lce = np.array([calculate_lce_angle(
            femur_head_center=lm[1],
            lateral_ace_point=lm[0]
        ) for lm in ground_truth_landmarks_all])
        # Create empty arrays for left hip
        pred_left_alpha = np.array([])
        gt_left_alpha = np.array([])
        pred_left_lce = np.array([])
        gt_left_lce = np.array([])
    
    # Create properly aligned filename arrays for each hip
    # CRITICAL: X-ray images have TWO hips (right and left), MRI has only right hip
    # We need to track which filename corresponds to which hip measurement
    image_names_flat = np.array(metrics['image_names'])
    
    # Create tuples of (label_with_side, original_filename) for proper tracking
    hip_info_right = [(f"{name}_RIGHT", name) for name in image_names_flat]
    
    if has_both_hips:
        # X-ray: combine both hips
        hip_info_left = [(f"{name}_LEFT", name) for name in image_names_flat]
        # Combine angles and filenames in the SAME order: all rights, then all lefts
        pred_alpha_all = np.concatenate([pred_right_alpha, pred_left_alpha])
        gt_alpha_all = np.concatenate([gt_right_alpha, gt_left_alpha])
        pred_lce_all = np.concatenate([pred_right_lce, pred_left_lce])
        gt_lce_all = np.concatenate([gt_right_lce, gt_left_lce])
        hip_info_all = hip_info_right + hip_info_left
    else:
        # MRI: only right hip
        pred_alpha_all = pred_right_alpha
        gt_alpha_all = gt_right_alpha
        pred_lce_all = pred_right_lce
        gt_lce_all = gt_right_lce
        hip_info_all = hip_info_right
    
    # Verify alignment - CRITICAL CHECK
    expected_hips = len(image_names_flat) * 2 if has_both_hips else len(image_names_flat)
    if len(hip_info_all) != len(pred_alpha_all):
        logger.error(f"CRITICAL: Mismatch between hip info ({len(hip_info_all)}) and angles ({len(pred_alpha_all)})")
        logger.error(f"This will cause incorrect filename associations!")
        logger.error(f"Number of images: {len(image_names_flat)}, Expected hips: {expected_hips}")
        raise ValueError("Hip info and angle arrays are misaligned!")
    else:
        hip_type = "bilateral (left + right)" if has_both_hips else "unilateral (right only)"
        logger.info(f"✓ Hip info alignment verified: {len(hip_info_all)} hips from {len(image_names_flat)} images ({hip_type})")
    
    # Compute angle statistics
    alpha_stats = compute_angle_statistics(pred_alpha_all, gt_alpha_all)
    lce_stats = compute_angle_statistics(pred_lce_all, gt_lce_all)
    
    # Log alpha angle results
    dataset_type_str = "X-ray (bilateral)" if has_both_hips else "MRI (right hip only)"
    logger.info(f"\nDataset: {dataset_type_str}")
    logger.info(f"\nAlpha (α) Angle Results:")
    logger.info(f"  Mean Difference: {alpha_stats['mean_difference']:.2f}° ± {alpha_stats['std_difference']:.2f}°")
    logger.info(f"  Median Difference: {alpha_stats['median_difference']:.2f}°")
    logger.info(f"  ICC Score: {alpha_stats['icc']:.3f}")
    logger.info(f"  Correlation: {alpha_stats['correlation']:.3f}")
    
    # Log LCE angle results
    logger.info(f"\nLateral Center Edge (LCE) Angle Results:")
    logger.info(f"  Mean Difference: {lce_stats['mean_difference']:.2f}° ± {lce_stats['std_difference']:.2f}°")
    logger.info(f"  Median Difference: {lce_stats['median_difference']:.2f}°")
    logger.info(f"  ICC Score: {lce_stats['icc']:.3f}")
    logger.info(f"  Correlation: {lce_stats['correlation']:.3f}")
    
    # Diagnostic accuracy (cam impingement)
    cam_threshold = 65.0
    pred_cam_diagnoses = pred_alpha_all > cam_threshold
    gt_cam_diagnoses = gt_alpha_all > cam_threshold
    cam_agreement = np.mean(pred_cam_diagnoses == gt_cam_diagnoses)
    
    # Diagnostic accuracy (pincer impingement)
    pincer_threshold = 40.0
    pred_pincer_diagnoses = pred_lce_all > pincer_threshold
    gt_pincer_diagnoses = gt_lce_all > pincer_threshold
    pincer_agreement = np.mean(pred_pincer_diagnoses == gt_pincer_diagnoses)

    # Count diagnostics
    total_hips = len(pred_alpha_all)
    correct_no_cam = np.sum((~pred_cam_diagnoses) & (~gt_cam_diagnoses))  # True Negatives (TN)
    correct_cam = np.sum(pred_cam_diagnoses & gt_cam_diagnoses)  # True Positives (TP)
    missed_cam = np.sum((~pred_cam_diagnoses) & gt_cam_diagnoses)  # False Negatives (FN)
    false_cam = np.sum(pred_cam_diagnoses & (~gt_cam_diagnoses))  # False Positives (FP)
    
    # Calculate diagnostic performance metrics
    # Confusion matrix: TP, FP, FN, TN
    TP = int(correct_cam)
    FP = int(false_cam)
    FN = int(missed_cam)
    TN = int(correct_no_cam)
    
    # Sensitivity (Recall, True Positive Rate): TP / (TP + FN)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    
    # Specificity (True Negative Rate): TN / (TN + FP)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    
    # Precision (Positive Predictive Value): TP / (TP + FP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    
    # Negative Predictive Value (NPV): TN / (TN + FN)
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    
    # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    
    # Calculate diagnostic performance metrics for PINCER impingement
    total_hips_pincer = len(pred_lce_all)
    correct_no_pincer = np.sum((~pred_pincer_diagnoses) & (~gt_pincer_diagnoses))  # True Negatives (TN)
    correct_pincer = np.sum(pred_pincer_diagnoses & gt_pincer_diagnoses)  # True Positives (TP)
    missed_pincer = np.sum((~pred_pincer_diagnoses) & gt_pincer_diagnoses)  # False Negatives (FN)
    false_pincer = np.sum(pred_pincer_diagnoses & (~gt_pincer_diagnoses))  # False Positives (FP)
    
    # Pincer confusion matrix: TP, FP, FN, TN
    TP_pincer = int(correct_pincer)
    FP_pincer = int(false_pincer)
    FN_pincer = int(missed_pincer)
    TN_pincer = int(correct_no_pincer)
    
    # Pincer diagnostic metrics
    sensitivity_pincer = TP_pincer / (TP_pincer + FN_pincer) if (TP_pincer + FN_pincer) > 0 else 0.0
    specificity_pincer = TN_pincer / (TN_pincer + FP_pincer) if (TN_pincer + FP_pincer) > 0 else 0.0
    precision_pincer = TP_pincer / (TP_pincer + FP_pincer) if (TP_pincer + FP_pincer) > 0 else 0.0
    npv_pincer = TN_pincer / (TN_pincer + FN_pincer) if (TN_pincer + FN_pincer) > 0 else 0.0
    f1_score_pincer = 2 * (precision_pincer * sensitivity_pincer) / (precision_pincer + sensitivity_pincer) if (precision_pincer + sensitivity_pincer) > 0 else 0.0

    # Save detailed CAM diagnosis information with CORRECT filename associations
    txt_save_path = os.path.join(output_path, 'cam_impingement_errors.txt')
    with open(txt_save_path, 'w') as f:
        f.write('='*80 + '\n')
        f.write('CAM IMPINGEMENT DIAGNOSTIC ANALYSIS\n')
        f.write('='*80 + '\n')
        f.write(f'Threshold: α > {cam_threshold}°\n')
        f.write(f'Total hips evaluated: {total_hips}\n')
        f.write(f'Diagnostic accuracy: {cam_agreement*100:.1f}%\n')
        f.write('='*80 + '\n\n')
        
        f.write('CONFUSION MATRIX\n')
        f.write('-'*80 + '\n')
        f.write(f'{"":>20} | {"Predicted CAM":>15} | {"Predicted NO CAM":>18}\n')
        f.write('-'*80 + '\n')
        f.write(f'{"GT CAM":>20} | {TP:>15} (TP) | {FN:>18} (FN)\n')
        f.write(f'{"GT NO CAM":>20} | {FP:>15} (FP) | {TN:>18} (TN)\n')
        f.write('-'*80 + '\n\n')
        
        f.write('DIAGNOSTIC PERFORMANCE METRICS\n')
        f.write('-'*80 + '\n')
        f.write(f'Sensitivity (Recall/TPR):        {sensitivity*100:.2f}%  (TP / [TP + FN])\n')
        f.write(f'Specificity (TNR):                {specificity*100:.2f}%  (TN / [TN + FP])\n')
        f.write(f'Precision (PPV):                  {precision*100:.2f}%  (TP / [TP + FP])\n')
        f.write(f'Negative Predictive Value (NPV):  {npv*100:.2f}%  (TN / [TN + FN])\n')
        f.write(f'F1 Score:                         {f1_score:.3f}\n')
        f.write(f'Accuracy:                         {cam_agreement*100:.2f}%  ([TP + TN] / Total)\n')
        f.write('-'*80 + '\n\n')
        
        f.write('INTERPRETATION GUIDE\n')
        f.write('-'*80 + '\n')
        f.write('- Sensitivity: Of all true CAM cases, what % did we detect?\n')
        f.write('- Specificity: Of all true NO-CAM cases, what % did we correctly identify?\n')
        f.write('- Precision (PPV): Of all predicted CAM cases, what % were correct?\n')
        f.write('- NPV: Of all predicted NO-CAM cases, what % were correct?\n')
        f.write('='*80 + '\n\n')
        
        f.write(f'MISSED CAM IMPINGEMENT (GT=cam, Pred=not cam): {missed_cam} hips\n')
        f.write('-'*80 + '\n')
        f.write(f'{"Hip ID":<45} | {"GT α":>8} | {"Pred α":>8} | {"Δα":>8}\n')
        f.write('-'*80 + '\n')
        for i, (hip_label, base_name) in enumerate(hip_info_all):
            if (~pred_cam_diagnoses[i]) and gt_cam_diagnoses[i]:
                angle_diff = abs(pred_alpha_all[i] - gt_alpha_all[i])
                f.write(f'{hip_label:<45} | {gt_alpha_all[i]:>7.2f}° | {pred_alpha_all[i]:>7.2f}° | {angle_diff:>7.2f}°\n')
        
        f.write(f'\n\nFALSE POSITIVE CAM (GT=not cam, Pred=cam): {false_cam} hips\n')
        f.write('-'*80 + '\n')
        f.write(f'{"Hip ID":<45} | {"GT α":>8} | {"Pred α":>8} | {"Δα":>8}\n')
        f.write('-'*80 + '\n')
        for i, (hip_label, base_name) in enumerate(hip_info_all):
            if pred_cam_diagnoses[i] and (~gt_cam_diagnoses[i]):
                angle_diff = abs(pred_alpha_all[i] - gt_alpha_all[i])
                f.write(f'{hip_label:<45} | {gt_alpha_all[i]:>7.2f}° | {pred_alpha_all[i]:>7.2f}° | {angle_diff:>7.2f}°\n')
        
        f.write(f'\n\nCORRECT CAM DIAGNOSIS: {correct_cam} hips\n')
        f.write('-'*80 + '\n')
        f.write(f'{"Hip ID":<45} | {"GT α":>8} | {"Pred α":>8} | {"Δα":>8}\n')
        f.write('-'*80 + '\n')
        for i, (hip_label, base_name) in enumerate(hip_info_all):
            if pred_cam_diagnoses[i] and gt_cam_diagnoses[i]:
                angle_diff = abs(pred_alpha_all[i] - gt_alpha_all[i])
                f.write(f'{hip_label:<45} | {gt_alpha_all[i]:>7.2f}° | {pred_alpha_all[i]:>7.2f}° | {angle_diff:>7.2f}°\n')
        
        f.write(f'\n\nCORRECT NO CAM DIAGNOSIS: {correct_no_cam} hips\n')
        f.write('-'*80 + '\n')
        f.write(f'{"Hip ID":<45} | {"GT α":>8} | {"Pred α":>8} | {"Δα":>8}\n')
        f.write('-'*80 + '\n')
        for i, (hip_label, base_name) in enumerate(hip_info_all):
            if (~pred_cam_diagnoses[i]) and (~gt_cam_diagnoses[i]):
                angle_diff = abs(pred_alpha_all[i] - gt_alpha_all[i])
                f.write(f'{hip_label:<45} | {gt_alpha_all[i]:>7.2f}° | {pred_alpha_all[i]:>7.2f}° | {angle_diff:>7.2f}°\n')
    
    # Save detailed PINCER diagnosis information with CORRECT filename associations
    txt_save_path_pincer = os.path.join(output_path, 'pincer_impingement_errors.txt')
    with open(txt_save_path_pincer, 'w') as f:
        f.write('='*80 + '\n')
        f.write('PINCER IMPINGEMENT DIAGNOSTIC ANALYSIS\n')
        f.write('='*80 + '\n')
        f.write(f'Threshold: LCE > {pincer_threshold}°\n')
        f.write(f'Total hips evaluated: {total_hips_pincer}\n')
        f.write(f'Diagnostic accuracy: {pincer_agreement*100:.1f}%\n')
        f.write('='*80 + '\n\n')
        
        f.write('CONFUSION MATRIX\n')
        f.write('-'*80 + '\n')
        f.write(f'{"":>20} | {"Predicted PINCER":>18} | {"Predicted NO PINCER":>21}\n')
        f.write('-'*80 + '\n')
        f.write(f'{"GT PINCER":>20} | {TP_pincer:>18} (TP) | {FN_pincer:>21} (FN)\n')
        f.write(f'{"GT NO PINCER":>20} | {FP_pincer:>18} (FP) | {TN_pincer:>21} (TN)\n')
        f.write('-'*80 + '\n\n')
        
        f.write('DIAGNOSTIC PERFORMANCE METRICS\n')
        f.write('-'*80 + '\n')
        f.write(f'Sensitivity (Recall/TPR):        {sensitivity_pincer*100:.2f}%  (TP / [TP + FN])\n')
        f.write(f'Specificity (TNR):                {specificity_pincer*100:.2f}%  (TN / [TN + FP])\n')
        f.write(f'Precision (PPV):                  {precision_pincer*100:.2f}%  (TP / [TP + FP])\n')
        f.write(f'Negative Predictive Value (NPV):  {npv_pincer*100:.2f}%  (TN / [TN + FN])\n')
        f.write(f'F1 Score:                         {f1_score_pincer:.3f}\n')
        f.write(f'Accuracy:                         {pincer_agreement*100:.2f}%  ([TP + TN] / Total)\n')
        f.write('-'*80 + '\n\n')
        
        f.write('INTERPRETATION GUIDE\n')
        f.write('-'*80 + '\n')
        f.write('- Sensitivity: Of all true PINCER cases, what % did we detect?\n')
        f.write('- Specificity: Of all true NO-PINCER cases, what % did we correctly identify?\n')
        f.write('- Precision (PPV): Of all predicted PINCER cases, what % were correct?\n')
        f.write('- NPV: Of all predicted NO-PINCER cases, what % were correct?\n')
        f.write('='*80 + '\n\n')
        
        f.write(f'MISSED PINCER IMPINGEMENT (GT=pincer, Pred=not pincer): {missed_pincer} hips\n')
        f.write('-'*80 + '\n')
        f.write(f'{"Hip ID":<45} | {"GT LCE":>8} | {"Pred LCE":>10} | {"ΔLCE":>8}\n')
        f.write('-'*80 + '\n')
        for i, (hip_label, base_name) in enumerate(hip_info_all):
            if (~pred_pincer_diagnoses[i]) and gt_pincer_diagnoses[i]:
                f.write(f'{hip_label:<45} | {gt_lce_all[i]:>8.2f} | {pred_lce_all[i]:>10.2f} | {abs(pred_lce_all[i]-gt_lce_all[i]):>8.2f}\n')
        
        f.write(f'\n\nFALSE POSITIVE PINCER (GT=not pincer, Pred=pincer): {false_pincer} hips\n')
        f.write('-'*80 + '\n')
        f.write(f'{"Hip ID":<45} | {"GT LCE":>8} | {"Pred LCE":>10} | {"ΔLCE":>8}\n')
        f.write('-'*80 + '\n')
        for i, (hip_label, base_name) in enumerate(hip_info_all):
            if pred_pincer_diagnoses[i] and (~gt_pincer_diagnoses[i]):
                f.write(f'{hip_label:<45} | {gt_lce_all[i]:>8.2f} | {pred_lce_all[i]:>10.2f} | {abs(pred_lce_all[i]-gt_lce_all[i]):>8.2f}\n')
        
        f.write(f'\n\nCORRECT PINCER DIAGNOSIS: {correct_pincer} hips\n')
        f.write('-'*80 + '\n')
        f.write(f'{"Hip ID":<45} | {"GT LCE":>8} | {"Pred LCE":>10} | {"ΔLCE":>8}\n')
        f.write('-'*80 + '\n')
        for i, (hip_label, base_name) in enumerate(hip_info_all):
            if pred_pincer_diagnoses[i] and gt_pincer_diagnoses[i]:
                f.write(f'{hip_label:<45} | {gt_lce_all[i]:>8.2f} | {pred_lce_all[i]:>10.2f} | {abs(pred_lce_all[i]-gt_lce_all[i]):>8.2f}\n')
        
        f.write(f'\n\nCORRECT NO PINCER DIAGNOSIS: {correct_no_pincer} hips\n')
        f.write('-'*80 + '\n')
        f.write(f'{"Hip ID":<45} | {"GT LCE":>8} | {"Pred LCE":>10} | {"ΔLCE":>8}\n')
        f.write('-'*80 + '\n')
        for i, (hip_label, base_name) in enumerate(hip_info_all):
            if (~pred_pincer_diagnoses[i]) and (~gt_pincer_diagnoses[i]):
                f.write(f'{hip_label:<45} | {gt_lce_all[i]:>8.2f} | {pred_lce_all[i]:>10.2f} | {abs(pred_lce_all[i]-gt_lce_all[i]):>8.2f}\n')
    
    logger.info(f"\nCam Impingement Diagnosis (α > {cam_threshold}°):")
    logger.info(f"  Total hips in test set: {total_hips}")
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"    True Positives (TP):  {TP} (correctly detected CAM)")
    logger.info(f"    False Positives (FP): {FP} (incorrectly diagnosed CAM)")
    logger.info(f"    False Negatives (FN): {FN} (missed CAM)")
    logger.info(f"    True Negatives (TN):  {TN} (correctly identified no CAM)")
    logger.info(f"\n  Diagnostic Performance Metrics:")
    logger.info(f"    Accuracy:     {cam_agreement*100:.2f}%")
    logger.info(f"    Sensitivity:  {sensitivity*100:.2f}%  (ability to detect CAM when present)")
    logger.info(f"    Specificity:  {specificity*100:.2f}%  (ability to rule out CAM when absent)")
    logger.info(f"    Precision (PPV): {precision*100:.2f}%  (confidence in CAM diagnosis)")
    logger.info(f"    NPV:          {npv*100:.2f}%  (confidence in no-CAM diagnosis)")
    logger.info(f"    F1 Score:     {f1_score:.3f}")
    
    logger.info(f"\nPincer Impingement Diagnosis (LCE > {pincer_threshold}°):")
    logger.info(f"  Total hips in test set: {total_hips_pincer}")
    logger.info(f"\n  Confusion Matrix:")
    logger.info(f"    True Positives (TP):  {TP_pincer} (correctly detected PINCER)")
    logger.info(f"    False Positives (FP): {FP_pincer} (incorrectly diagnosed PINCER)")
    logger.info(f"    False Negatives (FN): {FN_pincer} (missed PINCER)")
    logger.info(f"    True Negatives (TN):  {TN_pincer} (correctly identified no PINCER)")
    logger.info(f"\n  Diagnostic Performance Metrics:")
    logger.info(f"    Accuracy:     {pincer_agreement*100:.2f}%")
    logger.info(f"    Sensitivity:  {sensitivity_pincer*100:.2f}%  (ability to detect PINCER when present)")
    logger.info(f"    Specificity:  {specificity_pincer*100:.2f}%  (ability to rule out PINCER when absent)")
    logger.info(f"    Precision (PPV): {precision_pincer*100:.2f}%  (confidence in PINCER diagnosis)")
    logger.info(f"    NPV:          {npv_pincer*100:.2f}%  (confidence in no-PINCER diagnosis)")
    logger.info(f"    F1 Score:     {f1_score_pincer:.3f}")
    
    # Cumulative angle difference analysis
    alpha_diffs = np.abs(pred_alpha_all - gt_alpha_all)
    lce_diffs = np.abs(pred_lce_all - gt_lce_all)
    logger.info(f"\nCumulative Alpha Angle Differences:")
    for threshold in [10, 20, 30, 40, 50]:
        percentage = np.mean(alpha_diffs > threshold) * 100
        logger.info(f"  > {threshold}°: {percentage:.1f}%")
    
    logger.info(f"\nCumulative LCE Angle Differences:")
    for threshold in [5, 10, 15, 20, 25]:
        percentage = np.mean(lce_diffs > threshold) * 100
        logger.info(f"  > {threshold}°: {percentage:.1f}%")
    
    # ========================================
    # BLAND-ALTMAN PLOTS
    # ========================================
    logger.info("\nGenerating Bland-Altman plots...")
    bland_altman_stats = {}
    
    try:
        # Bland-Altman for landmark localization (radial errors)
        bland_altman_landmarks_path = os.path.join(output_path, 'bland_altman_landmarks_radial.png')
        ba_stats = plot_bland_altman(
            gt_values=np.zeros_like(radial_errors),  # Ground truth has 0 error by definition
            pred_values=radial_errors,
            title='Bland-Altman Plot: Landmark Localization (Radial Errors)',
            xlabel='Mean Radial Error (mm)',
            ylabel='Radial Error (mm)',
            save_path=bland_altman_landmarks_path
        )
        bland_altman_stats['landmarks_radial'] = ba_stats
        logger.info(f"  ✓ Radial error BA: bias={ba_stats['mean_diff']:.2f}mm, LoA=[{ba_stats['lower_loa']:.2f}, {ba_stats['upper_loa']:.2f}], outliers={ba_stats['outliers_pct']:.1f}%")
    except Exception as e:
        logger.warning(f"  Failed to generate landmark radial BA plot: {e}")
    
    # Bland-Altman for X and Y coordinates separately
    try:
        # Extract X coordinates from landmarks
        pred_landmarks_all = np.array(metrics['predicted_landmarks_all'])  # (N, num_landmarks, 2)
        gt_landmarks_all = np.array(metrics['ground_truth_landmarks_all'])
        
        pred_x = pred_landmarks_all[:, :, 0].flatten()
        gt_x = gt_landmarks_all[:, :, 0].flatten()
        pred_y = pred_landmarks_all[:, :, 1].flatten()
        gt_y = gt_landmarks_all[:, :, 1].flatten()
        
        # X coordinate BA
        bland_altman_x_path = os.path.join(output_path, 'bland_altman_landmarks_x.png')
        ba_stats_x = plot_bland_altman(
            gt_values=gt_x,
            pred_values=pred_x,
            title='Bland-Altman Plot: X Coordinates',
            xlabel='Mean X (pixels)',
            ylabel='Difference (Pred - GT) X (pixels)',
            save_path=bland_altman_x_path
        )
        bland_altman_stats['landmarks_x'] = ba_stats_x
        logger.info(f"  ✓ X-coord BA: bias={ba_stats_x['mean_diff']:.2f}px, LoA=[{ba_stats_x['lower_loa']:.2f}, {ba_stats_x['upper_loa']:.2f}], outliers={ba_stats_x['outliers_pct']:.1f}%")
        
        # Y coordinate BA
        bland_altman_y_path = os.path.join(output_path, 'bland_altman_landmarks_y.png')
        ba_stats_y = plot_bland_altman(
            gt_values=gt_y,
            pred_values=pred_y,
            title='Bland-Altman Plot: Y Coordinates',
            xlabel='Mean Y (pixels)',
            ylabel='Difference (Pred - GT) Y (pixels)',
            save_path=bland_altman_y_path
        )
        bland_altman_stats['landmarks_y'] = ba_stats_y
        logger.info(f"  ✓ Y-coord BA: bias={ba_stats_y['mean_diff']:.2f}px, LoA=[{ba_stats_y['lower_loa']:.2f}, {ba_stats_y['upper_loa']:.2f}], outliers={ba_stats_y['outliers_pct']:.1f}%")
    except Exception as e:
        logger.warning(f"  Failed to generate X/Y coordinate BA plots: {e}")
    
    # Per-landmark Bland-Altman plots
    try:
        landmark_names = [
            "Right_Lateral_Ace", "Right_Femur_Center", "Right_Neck_Point", "Right_Cam_Point",
            "Left_Lateral_Ace", "Left_Femur_Center", "Left_Neck_Point", "Left_Cam_Point"
        ]
        
        per_landmark_dir = os.path.join(output_path, 'bland_altman_per_landmark')
        os.makedirs(per_landmark_dir, exist_ok=True)
        
        for lm_idx in range(min(num_landmarks, len(landmark_names))):
            lm_name = landmark_names[lm_idx]
            pred_lm = pred_landmarks_all[:, lm_idx, :]  # (N, 2)
            gt_lm = gt_landmarks_all[:, lm_idx, :]
            
            # Calculate radial errors for this landmark
            lm_radial_errors = np.sqrt(np.sum((pred_lm - gt_lm)**2, axis=1))
            
            ba_lm_path = os.path.join(per_landmark_dir, f'bland_altman_{lm_name}.png')
            ba_stats_lm = plot_bland_altman(
                gt_values=np.zeros_like(lm_radial_errors),
                pred_values=lm_radial_errors,
                title=f'Bland-Altman: {lm_name.replace("_", " ")}',
                xlabel='Mean Radial Error (mm)',
                ylabel='Radial Error (mm)',
                save_path=ba_lm_path
            )
            bland_altman_stats[f'landmark_{lm_idx}_{lm_name}'] = ba_stats_lm
        
        logger.info(f"  ✓ Per-landmark BA plots saved to: {per_landmark_dir}")
    except Exception as e:
        logger.warning(f"  Failed to generate per-landmark BA plots: {e}")
    
    try:
        # Bland-Altman for Alpha angles
        bland_altman_alpha_path = os.path.join(output_path, 'bland_altman_alpha_angle.png')
        ba_stats_alpha = plot_bland_altman(
            gt_values=gt_alpha_all,
            pred_values=pred_alpha_all,
            title='Bland-Altman Plot: Alpha Angle',
            xlabel='Mean Alpha Angle (°)',
            ylabel='Difference (Predicted - Ground Truth) (°)',
            save_path=bland_altman_alpha_path,
            clinical_threshold=cam_threshold
        )
        bland_altman_stats['alpha_angle'] = ba_stats_alpha
        logger.info(f"  ✓ Alpha angle BA: bias={ba_stats_alpha['mean_diff']:.2f}°, LoA=[{ba_stats_alpha['lower_loa']:.2f}, {ba_stats_alpha['upper_loa']:.2f}], outliers={ba_stats_alpha['outliers_pct']:.1f}%")
    except Exception as e:
        logger.warning(f"  Failed to generate alpha angle BA plot: {e}")
    
    try:
        # Bland-Altman for LCE angles
        bland_altman_lce_path = os.path.join(output_path, 'bland_altman_lce_angle.png')
        ba_stats_lce = plot_bland_altman(
            gt_values=gt_lce_all,
            pred_values=pred_lce_all,
            title='Bland-Altman Plot: LCE Angle',
            xlabel='Mean LCE Angle (°)',
            ylabel='Difference (Predicted - Ground Truth) (°)',
            save_path=bland_altman_lce_path,
            clinical_threshold=pincer_threshold
        )
        bland_altman_stats['lce_angle'] = ba_stats_lce
        logger.info(f"  ✓ LCE angle BA: bias={ba_stats_lce['mean_diff']:.2f}°, LoA=[{ba_stats_lce['lower_loa']:.2f}, {ba_stats_lce['upper_loa']:.2f}], outliers={ba_stats_lce['outliers_pct']:.1f}%")
    except Exception as e:
        logger.warning(f"  Failed to generate LCE angle BA plot: {e}")
    
    try:
        # Combined horizontal Bland-Altman plot for Alpha and LCE angles
        bland_altman_combined_path = os.path.join(output_path, 'bland_altman_combined_alpha_lce.png')
        ba_stats_alpha_combined, ba_stats_lce_combined = plot_combined_bland_altman(
            gt_alpha=gt_alpha_all,
            pred_alpha=pred_alpha_all,
            gt_lce=gt_lce_all,
            pred_lce=pred_lce_all,
            cam_threshold=cam_threshold,
            pincer_threshold=pincer_threshold,
            save_path=bland_altman_combined_path
        )
        bland_altman_stats['alpha_angle_combined'] = ba_stats_alpha_combined
        bland_altman_stats['lce_angle_combined'] = ba_stats_lce_combined
        logger.info(f"  ✓ Combined Alpha + LCE BA plot saved to: {bland_altman_combined_path}")
    except Exception as e:
        logger.warning(f"  Failed to generate combined alpha + LCE BA plot: {e}")
    
    # Save Bland-Altman statistics to text file
    try:
        ba_stats_file = os.path.join(output_path, 'bland_altman_statistics.txt')
        with open(ba_stats_file, 'w') as f:
            f.write('='*80 + '\n')
            f.write('BLAND-ALTMAN ANALYSIS STATISTICS\n')
            f.write('='*80 + '\n\n')
            f.write('Interpretation Guide:\n')
            f.write('  - Bias (Mean Difference): Average difference between predicted and ground truth.\n')
            f.write('    Close to 0 indicates no systematic error.\n')
            f.write('  - LoA (Limits of Agreement): Bias ± 1.96*SD, contains ~95% of differences.\n')
            f.write('    Narrow LoA indicates better precision/reproducibility.\n')
            f.write('  - Outliers: Percentage of points outside LoA (expected ~5% for normal distribution).\n')
            f.write('  - Proportional Bias p-value: If p < 0.05, difference varies with magnitude.\n')
            f.write('='*80 + '\n\n')
            
            for metric_name, stats in bland_altman_stats.items():
                f.write(f'{metric_name.upper().replace("_", " ")}\n')
                f.write('-'*80 + '\n')
                f.write(f'  N samples:           {stats["n_samples"]}\n')
                f.write(f'  Bias (mean diff):    {stats["mean_diff"]:.3f}\n')
                f.write(f'  SD of differences:   {stats["std_diff"]:.3f}\n')
                f.write(f'  Lower LoA:           {stats["lower_loa"]:.3f}\n')
                f.write(f'  Upper LoA:           {stats["upper_loa"]:.3f}\n')
                f.write(f'  LoA width:           {stats["upper_loa"] - stats["lower_loa"]:.3f}\n')
                f.write(f'  Outliers:            {stats["outliers_count"]} ({stats["outliers_pct"]:.2f}%)\n')
                if stats["proportional_bias_p"] is not None:
                    f.write(f'  Proportional bias:   p={stats["proportional_bias_p"]:.4f}')
                    if stats["proportional_bias_p"] < 0.05:
                        f.write(' [SIGNIFICANT - error varies with magnitude]')
                    f.write('\n')
                f.write('\n')
        
        logger.info(f"  ✓ Bland-Altman statistics saved to: {ba_stats_file}")
    except Exception as e:
        logger.warning(f"  Failed to save BA statistics: {e}")
    
    logger.info("="*70 + "\nMatrix:")
    logger.info(f"    True Positives (TP):  {TP} (correctly detected CAM)")
    logger.info(f"    False Positives (FP): {FP} (incorrectly diagnosed CAM)")
    logger.info(f"    False Negatives (FN): {FN} (missed CAM)")
    logger.info(f"    True Negatives (TN):  {TN} (correctly identified no CAM)")
    logger.info(f"\n  Diagnostic Performance Metrics:")
    logger.info(f"    Accuracy:     {cam_agreement*100:.2f}%")
    logger.info(f"    Sensitivity:  {sensitivity*100:.2f}%  (ability to detect CAM when present)")
    logger.info(f"    Specificity:  {specificity*100:.2f}%  (ability to rule out CAM when absent)")
    logger.info(f"    Precision (PPV): {precision*100:.2f}%  (confidence in CAM diagnosis)")
    logger.info(f"    NPV:          {npv*100:.2f}%  (confidence in no-CAM diagnosis)")
    logger.info(f"    F1 Score:     {f1_score:.3f}")
    
    # Cumulative angle difference analysis
    alpha_diffs = np.abs(pred_alpha_all - gt_alpha_all)
    logger.info(f"\nCumulative Alpha Angle Differences:")
    for threshold in [10, 20, 30, 40, 50]:
        percentage = np.mean(alpha_diffs > threshold) * 100
        logger.info(f"  > {threshold}°: {percentage:.1f}%")
    
    logger.info("="*70 + "\n")
    
    # ========================================
    # STANDARD METRICS
    # ========================================
    average_loss = np.mean(metrics['losses'])
    mre_for_landmarks = np.mean(radial_errors.reshape(-1, num_landmarks), axis=0)
    mre_for_landmarks = [round(x, 2) for x in mre_for_landmarks.tolist()]
    mre_std_for_landmarks = np.std(radial_errors.reshape(-1, num_landmarks), axis=0)
    mre_std_for_landmarks = [round(x, 2) for x in mre_std_for_landmarks.tolist()]
    median_re_for_landmarks = np.median(radial_errors.reshape(-1, num_landmarks), axis=0)
    median_re_for_landmarks = [round(x, 2) for x in median_re_for_landmarks.tolist()]
    mre = np.mean(radial_errors)
    median_re = np.median(radial_errors)
    average_ere = np.mean(expected_radial_errors)
    sdr_statistics, thresholds = compute_sdr(radial_errors)
    
    # Per-landmark MRE for each anatomical point
    landmark_names = [
        "Right Lateral Ace.", "Right Femur Center", "Right Neck Point", "Right Cam Point",
        "Left Lateral Ace.", "Left Femur Center", "Left Neck Point", "Left Cam Point"
    ]
    
    logger.info(f"Standard Evaluation Metrics:")
    logger.info(f"Average Loss: {average_loss:.3f}")
    logger.info(f"Mean Radial Error (MRE): {mre:.3f} mm")
    logger.info(f"Median Radial Error: {median_re:.3f} mm")
    logger.info(f"\nMRE per Landmark (Mean ± Std | Median):")
    for name, mre_val, std_val, median_val in zip(landmark_names, mre_for_landmarks, mre_std_for_landmarks, median_re_for_landmarks):
        logger.info(f"  {name}: {mre_val:.2f} ± {std_val:.2f} | {median_val:.2f} mm")
    logger.info(f"\nSuccessful Detection Rate (SDR):")
    for threshold, sdr in zip(thresholds, sdr_statistics):
        logger.info(f"  @ {threshold:.1f}mm: {sdr:.2f}%")
    
    logger.info(f"Expected Radial Error (ERE): {average_ere:.3f}")
    
    
    # Compile results
    results = {
        'loss': average_loss,
        'mre': mre,
        'median_re': median_re,
        'ere': average_ere,
        'mre_for_landmarks': mre_for_landmarks,
        'median_re_for_landmarks': median_re_for_landmarks,
        'std_for_landmarks': mre_std_for_landmarks,
        'sdr': (sdr_statistics, thresholds),
        'pixel_size': pixel_size,
        'use_tta': use_tta,  # Record whether TTA was used
        # Hip angle results
        'alpha_angle': {
            'mean_diff': alpha_stats['mean_difference'],
            'std_diff': alpha_stats['std_difference'],
            'median_diff': alpha_stats['median_difference'],
            'icc': alpha_stats['icc'],
            'correlation': alpha_stats['correlation'],
            'predicted_right': pred_right_alpha.tolist(),
            'predicted_left': pred_left_alpha.tolist(),
            'ground_truth_right': gt_right_alpha.tolist(),
            'ground_truth_left': gt_left_alpha.tolist()
        },
        'lce_angle': {
            'mean_diff': lce_stats['mean_difference'],
            'std_diff': lce_stats['std_difference'],
            'median_diff': lce_stats['median_difference'],
            'icc': lce_stats['icc'],
            'correlation': lce_stats['correlation'],
            'predicted_right': pred_right_lce.tolist(),
            'predicted_left': pred_left_lce.tolist(),
            'ground_truth_right': gt_right_lce.tolist(),
            'ground_truth_left': gt_left_lce.tolist()
        },
        'cam_diagnosis': {
            'threshold': cam_threshold,
            'accuracy': cam_agreement,
            'total_hips': total_hips,
            'confusion_matrix': {
                'true_positives': TP,
                'false_positives': FP,
                'false_negatives': FN,
                'true_negatives': TN
            },
            'performance_metrics': {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'ppv': precision,  # PPV is same as precision
                'npv': npv,
                'f1_score': f1_score
            },
        },
        'pincer_diagnosis': {
            'threshold': pincer_threshold,
            'accuracy': pincer_agreement,
            'total_hips': total_hips_pincer,
            'confusion_matrix': {
                'true_positives': TP_pincer,
                'false_positives': FP_pincer,
                'false_negatives': FN_pincer,
                'true_negatives': TN_pincer
            },
            'performance_metrics': {
                'sensitivity': sensitivity_pincer,
                'specificity': specificity_pincer,
                'precision': precision_pincer,
                'ppv': precision_pincer,
                'npv': npv_pincer,
                'f1_score': f1_score_pincer
            },
            'correct_no_pincer': TN_pincer,
            'correct_pincer': TP_pincer,
            'missed_pincer': FN_pincer,
            'false_positive': FP_pincer
        },
        'bland_altman': bland_altman_stats
    }
    
    # Save detailed results to JSON
    results_file = os.path.join(output_path, 'evaluation_results_detailed.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nDetailed results saved to: {results_file}")
    
    # Log TTA status
    if use_tta:
        logger.info("\n" + "="*70)
        logger.info("TEST-TIME AUGMENTATION (TTA) WAS USED")
        logger.info("="*70)
    
    return results