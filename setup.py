"""Setup utilities for model configuration, training, and logging.

This module provides utility functions for:
- Model architecture setup (U-Net, U-Net++, DPT)
- Loss function configuration
- Optimizer and scheduler setup
- Logging and system information
- Random seed setting for reproducibility
"""

import os
import sys
import time
import random
import logging
import argparse
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
import segmentation_models_pytorch as smp


class Unet(nn.Module):
    def __init__(self, encoder_name, encoder_weights, decoder_channels, in_channels, no_of_landmarks):
        super(Unet, self).__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            classes=no_of_landmarks,
        )
        self.num_landmarks = no_of_landmarks
    
    def forward(self, x):
        return self.unet(x)


def two_d_softmax(x):
        exp_y = torch.exp(x)
        return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)

class CustomNLLLoss(nn.Module):
    """
    Custom Negative Log-Likelihood loss for landmark detection with heatmaps.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output.
            'none': no reduction will be applied
            'mean': the sum of the output will be divided by the number of elements
            'perkeypoint': the loss will be calculated per keypoint and returned
    """
    def __init__(self, reduction="mean"):
        super(CustomNLLLoss, self).__init__()
        assert reduction in {"mean", "none", "perkeypoint"}
        self.reduction = reduction
    
    @staticmethod
    def two_d_softmax(x):
        """Apply 2D softmax to heatmaps"""
        exp_y = torch.exp(x)
        return exp_y / torch.sum(exp_y, dim=(2, 3), keepdim=True)
    
    def forward(self, output, target):
        """
        Forward pass for the NLL loss.
        
        Args:
            output (torch.Tensor): Predicted heatmaps
            target (torch.Tensor): Target heatmaps
        
        Returns:
            torch.Tensor: Calculated loss
        """
        # Compute NLL with numerical stability
        nll = -target * torch.log(output.float() + 1e-10)
        
        if self.reduction == "none":
            return nll
        elif self.reduction == "perkeypoint":
            return torch.sum(nll, dim=(2, 3))
        elif self.reduction == "mean":
            return torch.mean(torch.sum(nll, dim=(2, 3)))

def set_random_seed(seed=42):
    """
    Set the random seed for reproducibility.
    
    Args:
    seed (int): The seed value to set for random number generators
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    random.seed(seed)
    np.random.seed(seed)

    
def setup_model(model_type, encoder_name, encoder_weights, decoder_channels, in_channels, num_landmarks, device):
    if model_type == "smp_Unet":
        model = Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if encoder_weights == "imagenet" else None,
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            no_of_landmarks=num_landmarks
        ).to(device)
    elif model_type == "smp_UnetPlusPlus":
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if encoder_weights == "imagenet" else None,
            in_channels=in_channels,
            decoder_channels=decoder_channels,
            classes=num_landmarks
        ).to(device)
    elif model_type == "smp_DPT":
        model = smp.DPT(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights if encoder_weights == "imagenet" else None,
            in_channels=in_channels,
            classes=num_landmarks
        ).to(device)
    else:
        raise ValueError(f"Model {model_type} not found... Choose between: smp_Unet")
    
    return model
            

def setup_loss_function(loss_function_name, **kwargs):
    """
    Setup loss function with optional parameters.
    
    Args:
        loss_function_name (str): Name of the loss function
        **kwargs: Additional parameters for loss functions
            For UncertaintyGuided:
                - uncertainty_weight: Weight for variance penalty (default: 0.1)
                - concentration_weight: Weight for concentration term (default: 0.01)
    """
    if loss_function_name == "CrossEntropyLoss" or loss_function_name == "CE":
        return nn.CrossEntropyLoss()
    elif loss_function_name == "NegLogLikelihood" or loss_function_name == "NLL":
        return CustomNLLLoss()
    else:
        raise ValueError(f"Loss function {loss_function_name} not found... Choose between: CrossEntropyLoss, NegLogLikelihood")

def setup_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name == "Adam":
        return torch.optim.Adam(params=model_parameters, lr=lr)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(params=model_parameters, lr=lr, weight_decay=1e-5)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not found... Choose between: Adam, AdamW")

def setup_scheduler(scheduler_name, optimizer, patience):
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "reduceonplateau":
        return ReduceLROnPlateau(optimizer, patience=patience, factor=0.5, verbose=True)
    elif scheduler_name == "multisteplr":
        return MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=0.1, verbose=False)
    elif scheduler_name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
    elif scheduler_name == "exponentiallr":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, last_epoch=-1)
    else:
        raise ValueError(f"Scheduler {scheduler_name} not found... Choose between: ReduceLROnPlateau")

def setup_logging(base_dir):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f"{base_dir}/{time_str}_experiment.log"

    logging.basicConfig(filename=log_file, format='%(message)s')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger, log_file

    
def log_system_info(logger):
    logger.info("\n----------------------------------------- SYSTEM INFO -----------------------------------------")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Pytorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        try:
            logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
        except RuntimeError:
            logger.info("CUDNN version: unavailable (version mismatch)")
    logger.info("------------------------------------------------------------------------------------------------")

def generate_path(path):
    os.makedirs(path, exist_ok=True)
    return path
