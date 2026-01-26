"""Evaluation metrics for landmark detection and hip angle calculation.

This module provides:
- Landmark localization metrics (MRE, SDR, ERE)
- Hip angle calculations (Alpha angle for CAM, LCE angle for Pincer impingement)
- Diagnostic metrics (sensitivity, specificity, ICC)
- Bland-Altman agreement analysis
- Visualization functions for hip angles

References:
    Clinical thresholds based on standard FAI diagnostic criteria.
"""

from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt


def compute_sdr(distance_list, thresholds=[2, 2.5, 3, 4, 6, 9, 10]):
    distance_array = np.array(distance_list)
    successful_detection_rates = []
    for threshold in thresholds:
        sdr = 100 * (np.sum(distance_array <= threshold) / len(distance_array))
        successful_detection_rates.append(sdr)
    return successful_detection_rates, thresholds


# -----------------------------------------------------------------------------------------------------------------##
#                                                  Expected Radial Error (ERE)                                       ##
# -----------------------------------------------------------------------------------------------------------------##

def calculate_ere(heatmap, predicted_point_scaled, pixel_size, significant_pixel_cutoff=0.05):
    normalized_heatmap = heatmap / np.max(heatmap)
    normalized_heatmap = np.where(normalized_heatmap > significant_pixel_cutoff, normalized_heatmap, 0)
    normalized_heatmap /= (np.sum(normalized_heatmap) + 1e-10)
    indices = np.argwhere(normalized_heatmap)
    ere = 0
    for twod_idx in indices:
        scaled_idx = np.flip(twod_idx) * pixel_size
        dist = np.linalg.norm(predicted_point_scaled - scaled_idx)
        ere += dist * normalized_heatmap[twod_idx[0], twod_idx[1]]
    return ere

def compute_euclidean_distance(predicted_point: np.ndarray, target_point: np.ndarray) -> float: 
    euclidean_dist = np.linalg.norm(predicted_point - target_point)
    return euclidean_dist

def compute_mean_std(distance_list):
    means = np.mean(distance_list)
    std_devs = np.std(distance_list)
    return means, std_devs


# -----------------------------------------------------------------------------------------------------------------#
#                                           ALPHA ANGLE CALCULATION                                                #
# -----------------------------------------------------------------------------------------------------------------#

def calculate_alpha_angle(femur_head_center: np.ndarray, 
                         neck_orientation_point: np.ndarray,
                         lateral_cam_point: np.ndarray) -> float:
    """
    Calculate the alpha (α) angle for a single hip.
    
    The α-angle measures cam impingement and is the angle between:
    1. The neck axis: vector pointing FROM neck point TO femur center
    2. The cam axis: vector pointing FROM femur center TO cam point
    
    The angle returned is the exterior angle (180° - interior angle).
    
    Args:
        femur_head_center: (x, y) coordinates of femur head center
        neck_orientation_point: (x, y) coordinates along the neck
        lateral_cam_point: (x, y) coordinates where femur head becomes non-spherical
        
    Returns:
        float: Alpha angle in degrees
        
    Reference:
        Threshold: > 70° for men, > 61° for women (average ~65°)
    """
    # neck_axis points FROM neck point TO femur center
    neck_axis = femur_head_center - neck_orientation_point
    
    # cam_axis points FROM femur center TO cam point  
    cam_axis = lateral_cam_point - femur_head_center
    
    # Calculate angle between vectors
    dot_product = np.dot(neck_axis, cam_axis)
    norm_product = np.linalg.norm(neck_axis) * np.linalg.norm(cam_axis)
    
    if norm_product == 0:
        return 0.0
    
    cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    alpha_angle_rad = np.arccos(cos_angle)
    alpha_angle_deg = np.degrees(alpha_angle_rad)
    
    # Return the exterior angle (clinical measurement)
    return 180.0 - alpha_angle_deg


def calculate_alpha_angles_batch(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate alpha angles for a batch of images with both left and right hips.
    
    Args:
        landmarks: Array of shape (batch_size, 8, 2) with ordering:
            [0] right_lateral_ace
            [1] right_femur_center
            [2] adjusted_right_neck_point
            [3] right_alpha_angle (right CAM point)
            [4] left_lateral_ace
            [5] left_femur_center
            [6] adjusted_left_neck_point
            [7] left_alpha_angle (left CAM point)
            
    Returns:
        Tuple of (right_alpha_angles, left_alpha_angles), each of shape (batch_size,)
    """
    batch_size = landmarks.shape[0]
    right_alpha_angles = np.zeros(batch_size)
    left_alpha_angles = np.zeros(batch_size)
    
    for i in range(batch_size):
        # Right hip: use indices [1] (center), [2] (neck), [3] (cam)
        right_alpha_angles[i] = calculate_alpha_angle(
            femur_head_center=landmarks[i, 1],       # right_femur_center
            neck_orientation_point=landmarks[i, 2],  # adjusted_right_neck_point
            lateral_cam_point=landmarks[i, 3]        # right_alpha_angle (CAM)
        )
        
        # Left hip: use indices [5] (center), [6] (neck), [7] (cam)
        left_alpha_angles[i] = calculate_alpha_angle(
            femur_head_center=landmarks[i, 5],       # left_femur_center
            neck_orientation_point=landmarks[i, 6],  # adjusted_left_neck_point
            lateral_cam_point=landmarks[i, 7]        # left_alpha_angle (CAM)
        )
    
    return right_alpha_angles, left_alpha_angles


# -----------------------------------------------------------------------------------------------------------------#
#                                           LCE ANGLE CALCULATION                                                  #
# -----------------------------------------------------------------------------------------------------------------#

def calculate_lce_angle(femur_head_center: np.ndarray,
                       lateral_ace_point: np.ndarray) -> float:
    """
    Calculate the Lateral Center Edge (LCE) angle for a single hip.
    
    The LCE-angle measures pincer impingement and is the angle between:
    1. The vertical axis pointing UPWARD
    2. The line from femur head center to lateral acetabulum point
    
    Args:
        femur_head_center: (x, y) coordinates of femur head center
        lateral_ace_point: (x, y) coordinates of lateral acetabulum edge
        
    Returns:
        float: LCE angle in degrees
        
    Reference:
        Threshold: > 40° for pincer impingement
    """
    # In image coordinates, y increases downward, so upward is [0, -1]
    vertical_vector = np.array([0, -1])
    
    # Vector from femur head center to lateral acetabulum point
    ace_vector = lateral_ace_point - femur_head_center
    
    # Calculate angle between vectors
    dot_product = np.dot(vertical_vector, ace_vector)
    norm_product = np.linalg.norm(vertical_vector) * np.linalg.norm(ace_vector)
    
    if norm_product == 0:
        return 0.0
    
    cos_angle = np.clip(dot_product / norm_product, -1.0, 1.0)
    lce_angle_rad = np.arccos(cos_angle)
    lce_angle_deg = np.degrees(lce_angle_rad)
    
    return lce_angle_deg


def calculate_lce_angles_batch(landmarks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate LCE angles for a batch of images with both left and right hips.
    
    Args:
        landmarks: Array of shape (batch_size, 8, 2)
            
    Returns:
        Tuple of (right_lce_angles, left_lce_angles), each of shape (batch_size,)
    """
    batch_size = landmarks.shape[0]
    right_lce_angles = np.zeros(batch_size)
    left_lce_angles = np.zeros(batch_size)
    
    for i in range(batch_size):
        # Right hip: use indices [1] (center), [0] (lateral ace)
        right_lce_angles[i] = calculate_lce_angle(
            femur_head_center=landmarks[i, 1],  # right_femur_center
            lateral_ace_point=landmarks[i, 0]   # right_lateral_ace
        )
        
        # Left hip: use indices [5] (center), [4] (lateral ace)
        left_lce_angles[i] = calculate_lce_angle(
            femur_head_center=landmarks[i, 5],  # left_femur_center
            lateral_ace_point=landmarks[i, 4]   # left_lateral_ace
        )
    
    return right_lce_angles, left_lce_angles


# -----------------------------------------------------------------------------------------------------------------#
#                                           VISUALIZATION FUNCTIONS                                                #
# -----------------------------------------------------------------------------------------------------------------#

def draw_alpha_angle(ax, femur_head_center: np.ndarray,
                    neck_orientation_point: np.ndarray,
                    lateral_cam_point: np.ndarray,
                    color: str = 'lime',
                    linewidth: float = 2,
                    label: str = None):
    """
    Draw alpha angle lines on a matplotlib axis.
    
    Draws:
    1. Line from neck point TO femur center (neck axis)
    2. Line from femur center TO cam point (cam axis)
    The angle between these is the alpha angle.
    """
    # Draw line from neck point to femur center
    ax.plot([neck_orientation_point[0], femur_head_center[0]],
            [neck_orientation_point[1], femur_head_center[1]],
            color=color, linewidth=linewidth, label=label)
    
    # Draw line from femur center to cam point
    ax.plot([femur_head_center[0], lateral_cam_point[0]],
            [femur_head_center[1], lateral_cam_point[1]],
            color=color, linewidth=linewidth)
    
    # Draw marker at the femur center
    ax.scatter(femur_head_center[0], femur_head_center[1], 
              color=color, s=50, marker='o', zorder=5)


def draw_lce_angle(ax, femur_head_center: np.ndarray,
                  lateral_ace_point: np.ndarray,
                  color: str = 'lime',
                  linewidth: float = 2,
                  vertical_line_length: float = 50,
                  label: str = None):
    """
    Draw LCE angle lines on a matplotlib axis.
    
    Draws:
    1. Vertical line pointing UPWARD from femur center (reference)
    2. Line from femur center to lateral acetabulum point
    The angle between these is the LCE angle.
    """
    # Draw vertical line UPWARD (negative y direction in image coords)
    ax.plot([femur_head_center[0], femur_head_center[0]],
            [femur_head_center[1], femur_head_center[1] - vertical_line_length],
            color=color, linewidth=linewidth, linestyle='--', alpha=0.7)
    
    # Draw line from femur center to lateral acetabulum point
    ax.plot([femur_head_center[0], lateral_ace_point[0]],
            [femur_head_center[1], lateral_ace_point[1]],
            color=color, linewidth=linewidth, label=label)
    
    # Draw marker at the femur center
    ax.scatter(femur_head_center[0], femur_head_center[1], 
              color=color, s=50, marker='o', zorder=5)


def visualize_hip_angles_mri(image: np.ndarray,
                            predicted_landmarks: np.ndarray,
                            ground_truth_landmarks: np.ndarray = None,
                            save_path: str = None,
                            show_alpha: bool = True,
                            show_lce: bool = True) -> plt.Figure:
    """
    Visualize hip angles (alpha and/or LCE) on an MRI image with 4 landmarks (right hip only).
    
    MRI landmark ordering (4 landmarks):
      [0] right_lateral_ace
      [1] right_femur_center
      [2] adjusted_right_neck_point
      [3] right_alpha_angle (right CAM point)
    
    Args:
        image: MRI image array
        predicted_landmarks: Predicted landmarks (4, 2) for right hip
        ground_truth_landmarks: Optional ground truth landmarks (4, 2)
        save_path: Optional path to save the figure
        show_alpha: Whether to show alpha angles
        show_lce: Whether to show LCE angles
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2 if (show_alpha and show_lce) else 1, 
                            figsize=(15, 8) if (show_alpha and show_lce) else (8, 8))
    
    if not (show_alpha and show_lce):
        axes = [axes]
    
    # Calculate angles using single hip calculation
    pred_right_alpha = calculate_alpha_angle(
        femur_head_center=predicted_landmarks[1],
        neck_orientation_point=predicted_landmarks[2],
        lateral_cam_point=predicted_landmarks[3]
    )
    pred_right_lce = calculate_lce_angle(
        femur_head_center=predicted_landmarks[1],
        lateral_ace_point=predicted_landmarks[0]
    )
    
    if ground_truth_landmarks is not None:
        gt_right_alpha = calculate_alpha_angle(
            femur_head_center=ground_truth_landmarks[1],
            neck_orientation_point=ground_truth_landmarks[2],
            lateral_cam_point=ground_truth_landmarks[3]
        )
        gt_right_lce = calculate_lce_angle(
            femur_head_center=ground_truth_landmarks[1],
            lateral_ace_point=ground_truth_landmarks[0]
        )
    
    # Display image
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(2)
    else:
        image = image
    
    # Plot alpha angles
    if show_alpha:
        ax = axes[0]
        ax.imshow(image, cmap='gray')
        ax.set_title(f'Alpha Angle - Right Hip\nPredicted: {pred_right_alpha:.1f}°' + 
                    (f' | GT: {gt_right_alpha:.1f}°' if ground_truth_landmarks is not None else ''),
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Draw predicted alpha angle (green)
        draw_alpha_angle(ax,
                        femur_head_center=predicted_landmarks[1],
                        neck_orientation_point=predicted_landmarks[2],
                        lateral_cam_point=predicted_landmarks[3],
                        color='lime',
                        linewidth=2,
                        label='Predicted')
        
        # Draw ground truth alpha angle (red) if available
        if ground_truth_landmarks is not None:
            draw_alpha_angle(ax,
                           femur_head_center=ground_truth_landmarks[1],
                           neck_orientation_point=ground_truth_landmarks[2],
                           lateral_cam_point=ground_truth_landmarks[3],
                           color='red',
                           linewidth=2,
                           label='Ground Truth')
        
        ax.legend(loc='upper right', fontsize=10)
    
    # Plot LCE angles
    if show_lce:
        ax = axes[1] if show_alpha else axes[0]
        ax.imshow(image, cmap='gray')
        ax.set_title(f'LCE Angle - Right Hip\nPredicted: {pred_right_lce:.1f}°' + 
                    (f' | GT: {gt_right_lce:.1f}°' if ground_truth_landmarks is not None else ''),
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Draw predicted LCE angle (green)
        draw_lce_angle(ax,
                      femur_head_center=predicted_landmarks[1],
                      lateral_ace_point=predicted_landmarks[0],
                      color='lime',
                      linewidth=2,
                      label='Predicted')
        
        # Draw ground truth LCE angle (red) if available
        if ground_truth_landmarks is not None:
            draw_lce_angle(ax,
                          femur_head_center=ground_truth_landmarks[1],
                          lateral_ace_point=ground_truth_landmarks[0],
                          color='red',
                          linewidth=2,
                          label='Ground Truth')
        
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def visualize_hip_angles(image: np.ndarray,
                        predicted_landmarks: np.ndarray,
                        ground_truth_landmarks: np.ndarray = None,
                        save_path: str = None,
                        show_alpha: bool = True,
                        show_lce: bool = True) -> plt.Figure:
    """
    Visualize hip angles (alpha and/or LCE) on an X-ray image.
    
    Args:
        image: X-ray image array
        predicted_landmarks: Predicted landmarks (8, 2) with your landmark order
        ground_truth_landmarks: Optional ground truth landmarks (8, 2)
        save_path: Optional path to save the figure
        show_alpha: Whether to show alpha angles
        show_lce: Whether to show LCE angles
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(1, 2 if (show_alpha and show_lce) else 1, 
                            figsize=(15, 8) if (show_alpha and show_lce) else (8, 8))
    
    if not (show_alpha and show_lce):
        axes = [axes]
    
    # Calculate angles
    pred_right_alpha, pred_left_alpha = calculate_alpha_angles_batch(predicted_landmarks[np.newaxis, :])
    pred_right_lce, pred_left_lce = calculate_lce_angles_batch(predicted_landmarks[np.newaxis, :])
    
    pred_right_alpha, pred_left_alpha = pred_right_alpha[0], pred_left_alpha[0]
    pred_right_lce, pred_left_lce = pred_right_lce[0], pred_left_lce[0]
    
    if ground_truth_landmarks is not None:
        gt_right_alpha, gt_left_alpha = calculate_alpha_angles_batch(ground_truth_landmarks[np.newaxis, :])
        gt_right_lce, gt_left_lce = calculate_lce_angles_batch(ground_truth_landmarks[np.newaxis, :])
        gt_right_alpha, gt_left_alpha = gt_right_alpha[0], gt_left_alpha[0]
        gt_right_lce, gt_left_lce = gt_right_lce[0], gt_left_lce[0]
    
    # Display image
    if image.ndim == 3 and image.shape[0] == 1:
        image_display = image[0]
    elif image.ndim == 3 and image.shape[2] == 1:
        image_display = image[:, :, 0]
    else:
        image_display = image
    
    # Plot alpha angles
    if show_alpha:
        ax_idx = 0 if (show_alpha and show_lce) else 0
        axes[ax_idx].imshow(image_display, cmap='gray')
        axes[ax_idx].set_title('Alpha (α) Angles', fontsize=14, fontweight='bold')
        
        # Ground truth (if available)
        if ground_truth_landmarks is not None:
            # Right hip: [1] center, [2] neck, [3] cam
            draw_alpha_angle(axes[ax_idx], 
                           ground_truth_landmarks[1], ground_truth_landmarks[2], ground_truth_landmarks[3],
                           color='lime', label='Ground Truth')
            # Left hip: [5] center, [6] neck, [7] cam
            draw_alpha_angle(axes[ax_idx],
                           ground_truth_landmarks[5], ground_truth_landmarks[6], ground_truth_landmarks[7],
                           color='lime')
        
        # Predictions
        # Right hip: [1] center, [2] neck, [3] cam
        draw_alpha_angle(axes[ax_idx],
                       predicted_landmarks[1], predicted_landmarks[2], predicted_landmarks[3],
                       color='red', label='Prediction')
        # Left hip: [5] center, [6] neck, [7] cam
        draw_alpha_angle(axes[ax_idx],
                       predicted_landmarks[5], predicted_landmarks[6], predicted_landmarks[7],
                       color='red')
        
        # Add angle text
        text_str = f"Right α: {pred_right_alpha:.1f}°\nLeft α: {pred_left_alpha:.1f}°"
        if ground_truth_landmarks is not None:
            text_str += f"\n\nGT Right: {gt_right_alpha:.1f}°\nGT Left: {gt_left_alpha:.1f}°"
        axes[ax_idx].text(0.02, 0.98, text_str, transform=axes[ax_idx].transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[ax_idx].legend(loc='upper right')
        axes[ax_idx].axis('off')
    
    # Plot LCE angles
    if show_lce:
        ax_idx = 1 if (show_alpha and show_lce) else 0
        axes[ax_idx].imshow(image_display, cmap='gray')
        axes[ax_idx].set_title('LCE Angles', fontsize=14, fontweight='bold')
        
        # Ground truth (if available)
        if ground_truth_landmarks is not None:
            # Right hip: [1] center, [0] lateral ace
            draw_lce_angle(axes[ax_idx],
                         ground_truth_landmarks[1], ground_truth_landmarks[0],
                         color='lime', label='Ground Truth')
            # Left hip: [5] center, [4] lateral ace
            draw_lce_angle(axes[ax_idx],
                         ground_truth_landmarks[5], ground_truth_landmarks[4],
                         color='lime')
        
        # Predictions
        # Right hip: [1] center, [0] lateral ace
        draw_lce_angle(axes[ax_idx],
                     predicted_landmarks[1], predicted_landmarks[0],
                     color='red', label='Prediction')
        # Left hip: [5] center, [4] lateral ace
        draw_lce_angle(axes[ax_idx],
                     predicted_landmarks[5], predicted_landmarks[4],
                     color='red')
        
        # Add angle text
        text_str = f"Right LCE: {pred_right_lce:.1f}°\nLeft LCE: {pred_left_lce:.1f}°"
        if ground_truth_landmarks is not None:
            text_str += f"\n\nGT Right: {gt_right_lce:.1f}°\nGT Left: {gt_left_lce:.1f}°"
        axes[ax_idx].text(0.02, 0.98, text_str, transform=axes[ax_idx].transAxes,
                         fontsize=12, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[ax_idx].legend(loc='upper right')
        axes[ax_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig





# -----------------------------------------------------------------------------------------------------------------#
#                                           ICC (INTRACLASS CORRELATION) FUNCTIONS                                #
# -----------------------------------------------------------------------------------------------------------------#

def _compute_ms_components(data: np.ndarray):
    """
    Compute mean squares needed for ICC from a data matrix shape (n_subjects, k_raters).
    Returns MSR, MSC, MSE
    """
    n, k = data.shape
    grand_mean = np.mean(data)
    mean_by_subject = np.mean(data, axis=1)
    mean_by_rater = np.mean(data, axis=0)

    # Sum squares
    ss_between_subjects = k * np.sum((mean_by_subject - grand_mean) ** 2)
    ss_between_raters = n * np.sum((mean_by_rater - grand_mean) ** 2)
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_residual = ss_total - ss_between_subjects - ss_between_raters

    MSR = ss_between_subjects / (n - 1) if n > 1 else 0.0
    MSC = ss_between_raters / (k - 1) if k > 1 else 0.0
    MSE = ss_residual / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0.0
    return MSR, MSC, MSE


def icc_2_1_from_matrix(data: np.ndarray):
    """
    ICC(2,1) - Two-way random effects, absolute agreement, single measurement.
    Recommended for model vs ground-truth comparison when you want to measure
    absolute agreement and generalize to other raters.
    
    Args:
        data: shape (n_subjects, k_raters) - typically (n_samples, 2) for pred vs GT
    
    Returns:
        ICC(2,1) value or np.nan if computation fails
    """
    n, k = data.shape
    MSR, MSC, MSE = _compute_ms_components(data)
    denom = MSR + (k - 1) * MSE + (k / n) * (MSC - MSE) if n > 0 else 0.0
    if denom == 0:
        return np.nan
    return (MSR - MSE) / denom


def icc_3_1_from_matrix(data: np.ndarray):
    """
    ICC(3,1) - Two-way mixed effects, consistency (single measurement).
    Use when raters are fixed (e.g., specific model vs specific annotator)
    and you care about consistency rather than absolute agreement.
    
    Args:
        data: shape (n_subjects, k_raters) - typically (n_samples, 2) for pred vs GT
    
    Returns:
        ICC(3,1) value or np.nan if computation fails
    """
    MSR, MSC, MSE = _compute_ms_components(data)
    denom = MSR + (data.shape[1] - 1) * MSE
    if denom == 0:
        return np.nan
    return (MSR - MSE) / denom


# -----------------------------------------------------------------------------------------------------------------#
#                                           DIAGNOSIS FUNCTIONS                                                    #
# -----------------------------------------------------------------------------------------------------------------#

def diagnose_cam_impingement(alpha_angle: float, 
                            threshold: float = 65.0) -> bool:
    """
    Diagnose cam impingement based on alpha angle.
    
    Args:
        alpha_angle: Calculated alpha angle in degrees
        threshold: Diagnostic threshold (default 65°)
                  Clinical thresholds: > 70° for men, > 61° for women
        
    Returns:
        bool: True if cam impingement is diagnosed
    """
    return alpha_angle > threshold


def diagnose_pincer_impingement(lce_angle: float,
                               threshold: float = 40.0) -> bool:
    """
    Diagnose pincer impingement based on LCE angle.
    
    Args:
        lce_angle: Calculated LCE angle in degrees
        threshold: Diagnostic threshold (default 40° as per literature)
        
    Returns:
        bool: True if pincer impingement is diagnosed
    """
    return lce_angle > threshold


def compute_angle_statistics(predicted_angles: np.ndarray,
                            ground_truth_angles: np.ndarray) -> Dict[str, float]:
    """
    Compute statistics for angle predictions, including proper ICC estimates.
    
    Returns both ICC(2,1) and ICC(3,1); by default we report ICC(2,1) as the primary 'icc'
    since it measures absolute agreement and generalizes to other raters.
    
    Args:
        predicted_angles: Array of predicted angles
        ground_truth_angles: Array of ground truth angles
        
    Returns:
        Dict containing mean difference, std, median, ICC(2,1), ICC(3,1), and Pearson correlation
    """
    from scipy.stats import pearsonr
    
    predicted_angles = np.asarray(predicted_angles).astype(float)
    ground_truth_angles = np.asarray(ground_truth_angles).astype(float)
    
    # Ensure same shape
    if predicted_angles.shape != ground_truth_angles.shape:
        raise ValueError("predicted_angles and ground_truth_angles must have the same shape")
    
    differences = np.abs(predicted_angles - ground_truth_angles)
    
    # Pearson correlation
    correlation = 0.0
    if len(predicted_angles) > 1:
        try:
            correlation, _ = pearsonr(predicted_angles, ground_truth_angles)
        except Exception:
            correlation = np.nan
    
    # Prepare data matrix for ICC: shape (n_subjects, k_raters)
    # Each row = one subject; columns = raters (ground truth, prediction)
    data_matrix = np.vstack([ground_truth_angles, predicted_angles]).T  # shape (n, 2)
    
    icc_2_1 = icc_2_1_from_matrix(data_matrix)
    icc_3_1 = icc_3_1_from_matrix(data_matrix)
    
    # Choose primary ICC: default to ICC(2,1) (absolute agreement). Fallback to Pearson if ICC is nan.
    primary_icc = icc_2_1 if not (icc_2_1 is None or np.isnan(icc_2_1)) else correlation
    
    return {
        'mean_difference': float(np.mean(differences)),
        'std_difference': float(np.std(differences)),
        'median_difference': float(np.median(differences)),
        'icc': float(primary_icc) if not np.isnan(primary_icc) else None,
        'icc_2_1': float(icc_2_1) if not np.isnan(icc_2_1) else None,
        'icc_3_1': float(icc_3_1) if not np.isnan(icc_3_1) else None,
        'correlation': float(correlation) if not np.isnan(correlation) else None
    }

# -----------------------------------------------------------------------------------------------------------------#
#                                           DIAGNOSTIC REPORT GENERATION                                           #
# -----------------------------------------------------------------------------------------------------------------#

def generate_diagnostic_report(predicted_landmarks: np.ndarray,
                              ground_truth_landmarks: np.ndarray = None,
                              cam_threshold: float = 65.0,
                              pincer_threshold: float = 40.0) -> Dict:
    """
    Generate a comprehensive diagnostic report for both hips.
    
    Args:
        predicted_landmarks: Predicted landmarks array (8, 2)
        ground_truth_landmarks: Optional ground truth landmarks (8, 2)
        cam_threshold: Threshold for cam impingement diagnosis
        pincer_threshold: Threshold for pincer impingement diagnosis
        
    Returns:
        Dictionary containing diagnostic information
    """
    # Calculate angles
    pred_right_alpha, pred_left_alpha = calculate_alpha_angles_batch(
        predicted_landmarks[np.newaxis, :])
    pred_right_lce, pred_left_lce = calculate_lce_angles_batch(
        predicted_landmarks[np.newaxis, :])
    
    pred_right_alpha, pred_left_alpha = pred_right_alpha[0], pred_left_alpha[0]
    pred_right_lce, pred_left_lce = pred_right_lce[0], pred_left_lce[0]
    
    # Diagnoses
    right_cam_diagnosis = diagnose_cam_impingement(pred_right_alpha, cam_threshold)
    left_cam_diagnosis = diagnose_cam_impingement(pred_left_alpha, cam_threshold)
    right_pincer_diagnosis = diagnose_pincer_impingement(pred_right_lce, pincer_threshold)
    left_pincer_diagnosis = diagnose_pincer_impingement(pred_left_lce, pincer_threshold)
    
    report = {
        'predicted': {
            'right_hip': {
                'alpha_angle': pred_right_alpha,
                'lce_angle': pred_right_lce,
                'cam_impingement': right_cam_diagnosis,
                'pincer_impingement': right_pincer_diagnosis
            },
            'left_hip': {
                'alpha_angle': pred_left_alpha,
                'lce_angle': pred_left_lce,
                'cam_impingement': left_cam_diagnosis,
                'pincer_impingement': left_pincer_diagnosis
            }
        }
    }
    
    # Add ground truth if available
    if ground_truth_landmarks is not None:
        gt_right_alpha, gt_left_alpha = calculate_alpha_angles_batch(
            ground_truth_landmarks[np.newaxis, :])
        gt_right_lce, gt_left_lce = calculate_lce_angles_batch(
            ground_truth_landmarks[np.newaxis, :])
        
        gt_right_alpha, gt_left_alpha = gt_right_alpha[0], gt_left_alpha[0]
        gt_right_lce, gt_left_lce = gt_right_lce[0], gt_left_lce[0]
        
        report['ground_truth'] = {
            'right_hip': {
                'alpha_angle': gt_right_alpha,
                'lce_angle': gt_right_lce,
                'cam_impingement': diagnose_cam_impingement(gt_right_alpha, cam_threshold),
                'pincer_impingement': diagnose_pincer_impingement(gt_right_lce, pincer_threshold)
            },
            'left_hip': {
                'alpha_angle': gt_left_alpha,
                'lce_angle': gt_left_lce,
                'cam_impingement': diagnose_cam_impingement(gt_left_alpha, cam_threshold),
                'pincer_impingement': diagnose_pincer_impingement(gt_left_lce, pincer_threshold)
            }
        }
        
        # Add differences
        report['differences'] = {
            'right_hip': {
                'alpha_angle_diff': abs(pred_right_alpha - gt_right_alpha),
                'lce_angle_diff': abs(pred_right_lce - gt_right_lce)
            },
            'left_hip': {
                'alpha_angle_diff': abs(pred_left_alpha - gt_left_alpha),
                'lce_angle_diff': abs(pred_left_lce - gt_left_lce)
            }
        }
    
    return report


def plot_bland_altman(gt_values, pred_values, title='Bland-Altman Plot', 
                      xlabel='Mean', ylabel='Difference', save_path=None,
                      clinical_threshold=None):
    """
    Generate a Bland-Altman plot to assess agreement between predicted and ground truth values.
    
    The Bland-Altman plot shows:
    - X-axis: Mean of ground truth and predicted values [(GT + Pred) / 2]
    - Y-axis: Difference between predicted and ground truth [Pred - GT]
    - Horizontal lines showing:
        * Mean difference (bias)
        * Limits of agreement (mean ± 1.96 * SD)
    
    Args:
        gt_values: Ground truth values (numpy array)
        pred_values: Predicted values (numpy array)
        title: Title for the plot
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        save_path: Path to save the plot (optional)
        clinical_threshold: Optional clinical threshold to mark on plot (e.g., 65° for CAM)
    
    Returns:
        Dictionary with statistics: mean_diff, std_diff, lower_loa, upper_loa, 
                                    outliers_pct, proportional_bias_p
    """
    from scipy import stats
    
    gt_values = np.array(gt_values)
    pred_values = np.array(pred_values)
    
    # Calculate mean and difference
    mean_values = (gt_values + pred_values) / 2
    diff_values = pred_values - gt_values
    
    # Calculate statistics
    mean_diff = np.mean(diff_values)
    std_diff = np.std(diff_values, ddof=1)
    
    # Limits of agreement (mean ± 1.96 * SD)
    lower_loa = mean_diff - 1.96 * std_diff
    upper_loa = mean_diff + 1.96 * std_diff
    
    # Calculate percentage of outliers (outside LoA)
    outliers = np.sum((diff_values < lower_loa) | (diff_values > upper_loa))
    outliers_pct = 100.0 * outliers / len(diff_values)
    
    # Test for proportional bias (regression of difference vs mean)
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(mean_values, diff_values)
        proportional_bias_p = p_value
        has_proportional_bias = p_value < 0.05
    except:
        proportional_bias_p = None
        has_proportional_bias = False
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(mean_values, diff_values, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
    
    # Mean difference line
    ax.axhline(mean_diff, color='blue', linestyle='--', linewidth=2, 
               label=f'Bias: {mean_diff:.2f}')
    
    # Limits of agreement
    ax.axhline(upper_loa, color='red', linestyle='--', linewidth=2, 
               label=f'Upper LoA: {upper_loa:.2f}')
    ax.axhline(lower_loa, color='red', linestyle='--', linewidth=2, 
               label=f'Lower LoA: {lower_loa:.2f}')
    
    # Zero line (perfect agreement)
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3, label='Perfect agreement')
    
    # Add regression line if proportional bias detected
    if has_proportional_bias:
        regression_line = slope * mean_values + intercept
        ax.plot(mean_values, regression_line, 'g--', linewidth=1.5, alpha=0.7,
                label=f'Trend (p={proportional_bias_p:.3f})')
    
    # Add clinical threshold if provided (as vertical line on x-axis)
    if clinical_threshold is not None:
        ax.axvline(clinical_threshold, color='orange', linestyle=':', linewidth=2,
                   label=f'Clinical threshold: {clinical_threshold}')
    
    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    title_full = f"{title}\n(Outliers: {outliers_pct:.1f}%)"
    ax.set_title(title_full, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'lower_loa': lower_loa,
        'upper_loa': upper_loa,
        'outliers_count': int(outliers),
        'outliers_pct': outliers_pct,
        'proportional_bias_p': proportional_bias_p,
        'n_samples': len(diff_values)
    }


def plot_combined_bland_altman(gt_alpha, pred_alpha, gt_lce, pred_lce,
                                 cam_threshold=65.0, pincer_threshold=40.0,
                                 save_path=None):
    """
    Generate a combined horizontal Bland-Altman plot with Alpha and LCE angles side by side.
    Both plots are squared for better visual consistency.
    
    Args:
        gt_alpha: Ground truth alpha angles (numpy array)
        pred_alpha: Predicted alpha angles (numpy array)
        gt_lce: Ground truth LCE angles (numpy array)
        pred_lce: Predicted LCE angles (numpy array)
        cam_threshold: Clinical threshold for CAM impingement (default: 65°)
        pincer_threshold: Clinical threshold for pincer impingement (default: 40°)
        save_path: Path to save the plot (optional)
    
    Returns:
        Tuple of (alpha_stats, lce_stats) dictionaries with statistics
    """
    from scipy import stats
    
    # Convert to numpy arrays
    gt_alpha = np.array(gt_alpha)
    pred_alpha = np.array(pred_alpha)
    gt_lce = np.array(gt_lce)
    pred_lce = np.array(pred_lce)
    
    # Calculate statistics for Alpha angle
    mean_alpha = (gt_alpha + pred_alpha) / 2
    diff_alpha = pred_alpha - gt_alpha
    mean_diff_alpha = np.mean(diff_alpha)
    std_diff_alpha = np.std(diff_alpha, ddof=1)
    lower_loa_alpha = mean_diff_alpha - 1.96 * std_diff_alpha
    upper_loa_alpha = mean_diff_alpha + 1.96 * std_diff_alpha
    outliers_alpha = np.sum((diff_alpha < lower_loa_alpha) | (diff_alpha > upper_loa_alpha))
    outliers_pct_alpha = 100.0 * outliers_alpha / len(diff_alpha)
    
    # Calculate statistics for LCE angle
    mean_lce = (gt_lce + pred_lce) / 2
    diff_lce = pred_lce - gt_lce
    mean_diff_lce = np.mean(diff_lce)
    std_diff_lce = np.std(diff_lce, ddof=1)
    lower_loa_lce = mean_diff_lce - 1.96 * std_diff_lce
    upper_loa_lce = mean_diff_lce + 1.96 * std_diff_lce
    outliers_lce = np.sum((diff_lce < lower_loa_lce) | (diff_lce > upper_loa_lce))
    outliers_pct_lce = 100.0 * outliers_lce / len(diff_lce)
    
    # Test for proportional bias for Alpha
    try:
        slope_alpha, intercept_alpha, r_value_alpha, p_value_alpha, std_err_alpha = stats.linregress(mean_alpha, diff_alpha)
        proportional_bias_p_alpha = p_value_alpha
        has_proportional_bias_alpha = p_value_alpha < 0.05
    except:
        proportional_bias_p_alpha = None
        has_proportional_bias_alpha = False
    
    # Test for proportional bias for LCE
    try:
        slope_lce, intercept_lce, r_value_lce, p_value_lce, std_err_lce = stats.linregress(mean_lce, diff_lce)
        proportional_bias_p_lce = p_value_lce
        has_proportional_bias_lce = p_value_lce < 0.05
    except:
        proportional_bias_p_lce = None
        has_proportional_bias_lce = False
    
    # Create figure with 1 row, 2 columns - squared subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # ===== Alpha Angle Plot (Left) =====
    ax_alpha = axes[0]
    
    # Scatter plot with larger markers
    ax_alpha.scatter(mean_alpha, diff_alpha, alpha=0.7, s=80, edgecolors='k', linewidths=1.2)
    
    # Mean difference line
    ax_alpha.axhline(mean_diff_alpha, color='blue', linestyle='--', linewidth=3, 
                     label=f'Bias: {mean_diff_alpha:.2f}°')
    
    # Limits of agreement
    ax_alpha.axhline(upper_loa_alpha, color='red', linestyle='--', linewidth=3, 
                     label=f'Upper LoA: {upper_loa_alpha:.2f}°')
    ax_alpha.axhline(lower_loa_alpha, color='red', linestyle='--', linewidth=3, 
                     label=f'Lower LoA: {lower_loa_alpha:.2f}°')
    
    # Zero line (perfect agreement)
    ax_alpha.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.4, label='Perfect agreement')
    
    # Add regression line if proportional bias detected
    if has_proportional_bias_alpha:
        regression_line_alpha = slope_alpha * mean_alpha + intercept_alpha
        ax_alpha.plot(mean_alpha, regression_line_alpha, 'g--', linewidth=2.5, alpha=0.7,
                      label=f'Trend (p={proportional_bias_p_alpha:.3f})')
    
    # Add clinical threshold (vertical line)
    ax_alpha.axvline(cam_threshold, color='orange', linestyle=':', linewidth=3,
                     label=f'Clinical threshold: {cam_threshold}°')
    
    # Labels and title with larger fonts
    ax_alpha.set_xlabel('Mean Angle (Predicted + Ground Truth)/2 (°)', fontsize=22, fontweight='bold')
    ax_alpha.set_ylabel('Difference (Predicted - Ground Truth) (°)', fontsize=22, fontweight='bold')
    title_alpha = f"Alpha Angle (Outliers: {outliers_pct_alpha:.1f}%)"
    ax_alpha.set_title(title_alpha, fontsize=24, fontweight='bold')
    ax_alpha.legend(loc='best', fontsize=16, framealpha=0.9)
    ax_alpha.grid(True, alpha=0.3, linewidth=1)
    ax_alpha.tick_params(axis='both', which='major', labelsize=18)
    
    # ===== LCE Angle Plot (Right) =====
    ax_lce = axes[1]
    
    # Scatter plot with larger markers
    ax_lce.scatter(mean_lce, diff_lce, alpha=0.7, s=80, edgecolors='k', linewidths=1.2)
    
    # Mean difference line
    ax_lce.axhline(mean_diff_lce, color='blue', linestyle='--', linewidth=3, 
                   label=f'Bias: {mean_diff_lce:.2f}°')
    
    # Limits of agreement
    ax_lce.axhline(upper_loa_lce, color='red', linestyle='--', linewidth=3, 
                   label=f'Upper LoA: {upper_loa_lce:.2f}°')
    ax_lce.axhline(lower_loa_lce, color='red', linestyle='--', linewidth=3, 
                   label=f'Lower LoA: {lower_loa_lce:.2f}°')
    
    # Zero line (perfect agreement)
    ax_lce.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.4, label='Perfect agreement')
    
    # Add regression line if proportional bias detected
    if has_proportional_bias_lce:
        regression_line_lce = slope_lce * mean_lce + intercept_lce
        ax_lce.plot(mean_lce, regression_line_lce, 'g--', linewidth=2.5, alpha=0.7,
                    label=f'Trend (p={proportional_bias_p_lce:.3f})')
    
    # Add clinical threshold (vertical line)
    ax_lce.axvline(pincer_threshold, color='orange', linestyle=':', linewidth=3,
                   label=f'Clinical threshold: {pincer_threshold}°')
    
    # Labels and title with larger fonts
    ax_lce.set_xlabel('Mean Angle (Predicted + Ground Truth)/2 (°)', fontsize=22, fontweight='bold')
    #ax_lce.set_ylabel('Difference (Predicted - Ground Truth) (°)', fontsize=18, fontweight='bold')
    title_lce = f"LCE Angle (Outliers: {outliers_pct_lce:.1f}%)"
    ax_lce.set_title(title_lce, fontsize=24, fontweight='bold')
    ax_lce.legend(loc='best', fontsize=16, framealpha=0.9)
    ax_lce.grid(True, alpha=0.3, linewidth=1)
    ax_lce.tick_params(axis='both', which='major', labelsize=18)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        # save pdf
        fig.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    
    # Return statistics for both plots
    alpha_stats = {
        'mean_diff': mean_diff_alpha,
        'std_diff': std_diff_alpha,
        'lower_loa': lower_loa_alpha,
        'upper_loa': upper_loa_alpha,
        'outliers_count': int(outliers_alpha),
        'outliers_pct': outliers_pct_alpha,
        'proportional_bias_p': proportional_bias_p_alpha,
        'n_samples': len(diff_alpha)
    }
    
    lce_stats = {
        'mean_diff': mean_diff_lce,
        'std_diff': std_diff_lce,
        'lower_loa': lower_loa_lce,
        'upper_loa': upper_loa_lce,
        'outliers_count': int(outliers_lce),
        'outliers_pct': outliers_pct_lce,
        'proportional_bias_p': proportional_bias_p_lce,
        'n_samples': len(diff_lce)
    }
    
    return alpha_stats, lce_stats


def print_diagnostic_report(report: Dict):
    """
    Print a formatted diagnostic report.
    
    Args:
        report: Dictionary from generate_diagnostic_report
    """
    print("\n" + "="*70)
    print("HIP DIAGNOSTIC REPORT")
    print("="*70)
    
    for hip in ['right_hip', 'left_hip']:
        hip_name = hip.replace('_', ' ').title()
        print(f"\n{hip_name}:")
        print("-" * 70)
        
        pred = report['predicted'][hip]
        print(f"  Alpha Angle: {pred['alpha_angle']:.2f}° ", end="")
        print(f"[{'CAM IMPINGEMENT' if pred['cam_impingement'] else 'Normal'}]")
        
        print(f"  LCE Angle:   {pred['lce_angle']:.2f}° ", end="")
        print(f"[{'PINCER IMPINGEMENT' if pred['pincer_impingement'] else 'Normal'}]")
        
        if 'ground_truth' in report:
            gt = report['ground_truth'][hip]
            diff = report['differences'][hip]
            print(f"\n  Ground Truth Alpha: {gt['alpha_angle']:.2f}° (diff: {diff['alpha_angle_diff']:.2f}°)")
            print(f"  Ground Truth LCE:   {gt['lce_angle']:.2f}° (diff: {diff['lce_angle_diff']:.2f}°)")
    
    print("\n" + "="*70 + "\n")