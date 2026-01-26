

"""Main training and evaluation script for hip landmark detection.

This script trains and evaluates deep learning models for anatomical landmark
detection in hip X-ray and MRI images for Femoroacetabular Impingement (FAI) diagnosis.
"""

import os
import sys
import time
import argparse
import json
import logging

import torch

from aligned_datasets import get_aligned_dataset, create_dataloaders
from deep_learning import train_and_validate, evaluate
from setup import (
    set_random_seed, setup_model, setup_loss_function, setup_optimizer,
    setup_scheduler, setup_logging, log_system_info, generate_path
)

def main(config, logger, base_dir):
    """Main training and evaluation function.
    
    Args:
        config: Configuration dictionary containing model, dataset, and training parameters
        logger: Logger instance for output
        base_dir: Base directory for experiment outputs
    """

    device = torch.device(f"cuda:{config['gpu']}" if torch.cuda.is_available() and config["gpu"] != -1 else "cpu")
    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"]["path"]
    partition_path = config["dataset"]["partition_path"]
    image_size = tuple(config["dataset"]["image_size"])
    image_channels = config["dataset"]["image_channels"]
    sigma = config["dataset"]["sigma"]
    training_samples = config["dataset"]["training_samples"]
    if dataset_name in ["aligned_hip_xray", "aligned_hip_mri"]:
        modality = "xray" if dataset_name == "aligned_hip_xray" else "mri"
        train_dataset = get_aligned_dataset(modality, dataset_path, "train", image_size, image_channels, sigma, partition_path)
        val_dataset = get_aligned_dataset(modality, dataset_path, "val", image_size, image_channels, sigma, partition_path)
        test_dataset = get_aligned_dataset(modality, dataset_path, "test", image_size, image_channels, sigma, partition_path)
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}. Only 'aligned_hip_xray' and 'aligned_hip_mri' are supported.")

    num_landmarks = train_dataset.num_landmarks
    
    logger.info("\n----------------------------------------- DATASET -----------------------------------------")
    logger.info(f"Dataset: {dataset_name} |  Path: {dataset_path}")
    logger.info(f"Number of Landmarks: {num_landmarks}")
    logger.info(f"Image shape: ({image_channels}, {image_size[0]}, {image_size[1]})")
    logger.info(f"Sigma: {sigma}")
    logger.info(f"Training samples: {len(train_dataset)} | Validation samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")
    logger.info("------------------------------------------------------------------------------------------------")
    
    batch_size = config["dataloader"]["batch_size"]
    pin_memory = config["dataloader"]["pin_memory"]
    num_workers = config["dataloader"]["num_workers"]
   
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, batch_size, pin_memory, num_workers, training_samples
    )
    
    logger.info("\n----------------------------------------- DATALOADERS -----------------------------------------")
    logger.info(f"Batch size: {batch_size} x {config['dataloader']['grad_accumulation']} | Num workers: {num_workers} | Pin memory: {pin_memory}")
    logger.info(f"Training batches: {len(train_dataloader)} | Validation batches: {len(val_dataloader)} | Test batches: {len(test_dataloader)}")
    logger.info("------------------------------------------------------------------------------------------------")

    model_type = config["model"]["name"]
    model_encoder = config["model"]["encoder"]
    model_encoder_weights = config["model"]["encoder_weights"]
    model_decoder_channels = config["model"]["decoder_channels"]

    model = setup_model(model_type, model_encoder, model_encoder_weights, model_decoder_channels, image_channels, num_landmarks, device)

    # Create model identifier for path and logging
    arch_name = model_type.replace("smp_", "")
    encoder_name = model_encoder.replace("tu-", "").replace("_base_tf_512", "")
    model_display = f"{arch_name}_{encoder_name}"

    logger.info("\n----------------------------------------- MODEL -----------------------------------------")
    logger.info(f"Model: {model_display} | Encoder: {model_encoder} | Weights: {model_encoder_weights}")
    logger.info(f"Decoder channels: {model_decoder_channels}")
    logger.info("------------------------------------------------------------------------------------------------")

    loss_fn = setup_loss_function(config["training"]["loss_function"])
    optimizer = setup_optimizer(config["training"]["optimizer"], model.parameters(), config["training"]["learning_rate"])
    patience = config["training"]["early_stopping_patience"]
    scheduler = setup_scheduler(config["training"]["scheduler"], optimizer, (patience//2)-1)

    # Determine experiment category: main_results for UnetPlusPlus+resnet18, ablation_study otherwise
    is_main_model = (arch_name == "UnetPlusPlus" and encoder_name == "resnet18")
    if is_main_model:
        save_model_path = generate_path(f"{base_dir}/main_results/{modality}/seed_{config['seed']}")
    else:
        save_model_path = generate_path(f"{base_dir}/ablation_study/{model_display}/seed_{config['seed']}")

    logger.info("\n----------------------------------------- TRAINING SETTINGS -----------------------------------------")
    logger.info(f"Optimizer: {config['training']['optimizer']} | Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Scheduler: {config['training']['scheduler']} | Patience: {patience}")
    logger.info(f"Loss function: {config['training']['loss_function']}")
    logger.info(f"Save model path: {save_model_path}")
    logger.info("------------------------------------------------------------------------------------------------")


    best_model = f"{save_model_path}/best_checkpoint.pt"
    best_scaled_model = f"{save_model_path}/best_scaled_checkpoint.pt"
    
    start_time = time.time()
    if config["training"]["apply"] and not os.path.exists(best_model):

        import collections.abc
        def to_serializable(val):
            """Convert various Python types to JSON-serializable format."""
            if isinstance(val, (str, int, float, bool, type(None))):
                return val
            elif isinstance(val, (list, tuple)):
                return [to_serializable(v) for v in val]
            elif isinstance(val, dict):
                return {k: to_serializable(v) for k, v in val.items() if isinstance(k, str)}
            else:
                return str(val)

        exp_config = config.copy()
        try:
            if hasattr(train_dataset, 'transforms'):
                transforms = train_dataset.transforms
                if hasattr(transforms, 'transforms'):
                    aug_list = []
                    for t in transforms.transforms:
                        t_name = t.__class__.__name__
                        t_params = {}
                        for k, v in t.__dict__.items():
                            if k in ('always_apply', 'p'): continue
                            try:
                                t_params[k] = to_serializable(v)
                            except Exception:
                                t_params[k] = str(v)
                        aug_list.append({'name': t_name, 'params': t_params})
                    exp_config['dataset_augmentations'] = aug_list
        except Exception as e:
            exp_config['dataset_augmentations'] = f"Could not extract: {str(e)}"

        with open(f"{save_model_path}/exp_config.json", 'w') as f:
            json.dump(exp_config, f, indent=4)

        logger.info("\n----------------------------------------- START TRAINING -----------------------------------------")
        # Train and validate the model
        loss_results = train_and_validate(
            model, device, train_dataloader, val_dataloader, optimizer, scheduler, loss_fn,
            config["training"]["epochs"], save_model_path, patience=patience,
            gradient_accumulation_steps=config["dataloader"]["grad_accumulation"],
            continue_training=config["training"]["resume"], logger=logger
        )
        
        logger.info(f"\nTraining and validation completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
        logger.info("------------------------------------------------------------------------------------------------")

    if os.path.exists(best_model):
        logger.info("\n----------------------------------------- TESTING -----------------------------------------")
        logger.info(f"Loading best model from {best_model}")
        model.load_state_dict(torch.load(best_model, map_location=device, weights_only=False)["model_state_dict"])

        results = evaluate(model, device, test_dataloader, loss_fn, num_landmarks, output_path=f"{save_model_path}/test_results", logger=logger, use_tta=config["testing"]["tta"])
        logger.info("------------------------------------------------------------------------------------------------")

    end_time = time.time()
    logger.info(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))}")
    


def load_and_update_config(config_path, args=None):
    """Load configuration from JSON file and update with command-line arguments.
    
    Args:
        config_path: Path to the JSON config file
        args: Parsed command-line arguments (optional)
        
    Returns:
        dict: Configuration dictionary with CLI overrides applied
    """
    with open(config_path) as f:
        config = json.load(f)
    
    # Update configuration with command-line arguments if provided
    if args is not None:
        for key, value in vars(args).items():
            if key != 'config' and value is not None:
                if isinstance(value, str) and '.' in value and value.replace('.', '', 1).isdigit():
                    value = float(value)
                elif isinstance(value, str) and value.isdigit():
                    value = int(value)
                
                if '.' in key:
                    parts = key.split('.')
                    curr = config
                    for part in parts[:-1]:
                        if part not in curr:
                            curr[part] = {}
                        curr = curr[part]
                    curr[parts[-1]] = value
                else:
                    config[key] = value
    
    return config

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", type=int, default=0, help="GPU device index to use. Set to -1 to use CPU.")
    parser.add_argument("-c", "--config", type=str, default="config_aligned.json", help="Path to the JSON config file.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    config = load_and_update_config(args.config, args)
    set_random_seed(config["seed"])

    base_dir = generate_path(config["experiments_path"])
    log_dir = generate_path(os.path.join(base_dir, "logs"))

    logger, log_file = setup_logging(log_dir)
    log_system_info(logger)

    try:
        main(config, logger, base_dir)
    except (KeyboardInterrupt, SystemExit, Exception) as e:
        logger.exception("An error occurred during execution:")
        if log_file and os.path.exists(log_file):
            os.remove(log_file)
        sys.exit(1)
        