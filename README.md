# Automated Landmark Detection for Assessing Hip Conditions: A Cross-Modality Validation of MRI versus X-Ray

**ISBI 2026 Paper Implementation**

## Abstract

Femoroacetabular impingement (FAI) is a pathomechanical hip disorder affecting 20–25% of the population, characterized by abnormal contact between the femoral head–neck junction and acetabular rim during motion. Early identification is critical, as untreated FAI accelerates degenerative cartilage loss and predisposes to premature osteoarthritis. This project presents the **first cross-modality validation** of automated hip landmark detection between x-ray and MRI for FAI assessment. Using standard heatmap regression architectures on a matched-cohort dataset (89 patients with paired MRI/x-ray), we demonstrate that MRI achieves equivalent localization and diagnostic accuracy for cam-type impingement. Our method demonstrates clinical feasibility for FAI assessment in coronal views of 3D MRI volumes, opening the possibility for volumetric analysis through placing further landmarks and supporting integration of automated FAI assessment into routine MRI workflows.

## Table of Contents

- [Abstract](#abstract)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Quick Start](#quick-start)
- [Running Experiments](#running-experiments)
- [Results](#results)
- [Citation](#citation)

## Requirements

### Software Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 11GB+ VRAM)
- 16GB+ RAM

### Python Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

## Dataset

### Data Acquisition

The dataset consists of a **paired matched-cohort of 89 patients** who underwent both anteroposterior (AP) pelvic x-ray and coronal hip MRI for FAI evaluation, collected at Oxford University as part of the FAIT trial. This pathological cohort includes subjects with clinical and imaging evidence of FAI but without significant osteoarthritis or hip dysplasia.

**MRI Specifications:**
- **Modality**: T1-weighted coronal acquisitions
- **Volume**: 20 slices with 3.3 mm spacing
- **Target slice**: Middle slice (index 10) - mid-acetabular coronal plane where all four landmarks are concurrently visible with maximal clarity
- **Preprocessing**: Resized/padded to 512×512, min–max normalized to [0, 1]

**X-ray Specifications:**
- **Modality**: Anteroposterior (AP) pelvic x-rays
- **Preprocessing**: Standardized to 512×512, normalized to [0, 1]

### Landmarks

Four anatomical keypoints are annotated for each hip to enable FAI angle computation:

1. **FHC (Femoral Head Centre)**: Center of the femoral head (required for both angles)
2. **NA (Neck-Axis point)**: Point along the femoral neck centreline (for α-angle)
3. **LAE (Lateral Acetabular Edge)**: Lateral edge of the acetabulum (for LCE angle)
4. **LCP (Lateral Cam Point)**: Point where the femoral head deviates from sphericity (for α-angle)

### Clinical Angles

- **α-angle (alpha angle)**: Quantifies femoral head asphericity (cam morphology). Computed as the angle between the femoral neck axis (FHC→NA) and the cam deformity vector (FHC→LCP). Pathological threshold: **α > 65°**
- **LCE angle (Lateral Centre-Edge angle)**: Measures acetabular coverage (pincer morphology). Calculated as the angle between the vertical axis and the acetabular coverage vector (FHC→LAE). Pathological threshold: **LCE > 40°**

### Dataset Organization

Your dataset should be organized as follows:

```
XRAY-MRI-RIGHT-HIP/
├── imgs/
│   └── pngs/
│       └── PATIENT_ID/
│           └── VISIT_ID/
│               ├── MRI/
│               │   └── *.png          # T1-weighted MRI slice 10
│               └── XRAY/
│                   └── *_aligned.png  # AP pelvic x-ray
├── annotations/
│   ├── MRI/
│   │   └── PATIENT-VISIT-FILENAME.txt  # Landmark coordinates (FHC, NA, LAE, LCP)
│   └── XRAY/
│       └── PATIENT-VISIT-FILENAME.txt
└── pixel_sizes/
    ├── MRI/
    │   └── PATIENT-VISIT-FILENAME.txt  # Pixel spacing information
    └── XRAY/
        └── PATIENT-VISIT-FILENAME.txt
```

**Annotation format** (each line in `.txt` files):
```
landmark_name x_coordinate y_coordinate
```

**Pixel size format**:
```
pixel_size x_spacing y_spacing
```

### Data Splits

Patient-level splits ensure no data leakage. From 89 patients, the split yields 105:17:40 images (57:8:24 patients):
- **Training**: 65% of patients (105 images)
- **Validation**: 10% of patients (17 images)
- **Testing**: 25% of patients (40 images)

Splits are balanced across α-angle distributions using Kolmogorov–Smirnov testing, defined in `partitions/xray_mri_hip_partition.json`.

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/hip-landmark-detection.git
cd hip-landmark-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Dataset

Organize your paired MRI/x-ray images and annotations according to the [Dataset](#dataset) structure above.

### 4. Configure Experiment

Edit `config_aligned.json` to set your dataset path:

```json
{
    "dataset": {
        "name": "aligned_hip_mri",
        "path": "/path/to/your/XRAY-MRI-RIGHT-HIP",
        "partition_path": "./partitions/xray_mri_hip_partition.json",
        "image_size": [512, 512],
        "sigma": 5
    }
}
```

### 5. Train Model

```bash
python main.py --config config_aligned.json --gpu 0 --seed 42
```

## Running Experiments

### Method: Heatmap Regression

Following established practice in medical landmark detection, we adopt a **heatmap regression formulation**. For each landmark k ∈ {1, ..., 4}, we generate a ground-truth Gaussian heatmap centered at the annotated location with σ=5 (empirically providing the best trade-off between localization precision and training stability).

**Training Details:**
- **Architecture**: UNet++ with ResNet18 encoder (ImageNet-initialized)
- **Loss**: Negative Log-Likelihood (NLL) over four landmarks
- **Optimizer**: AdamW (learning rate 1×10⁻⁴, weight decay 1×10⁻⁵)
- **Scheduler**: ExponentialLR (γ = 0.95)
- **Batch size**: 4
- **Early stopping**: Applied to prevent overfitting
- **Test-Time Augmentation (TTA)**: Averaging heatmap predictions across augmented views

**Data Augmentation (during training):**
- Affine transforms: scale 0.95–1.05, translation ±5%, rotation ±10°, shear ±5°
- Intensity jitter: brightness/contrast ±15–20%, gamma 0.85–1.15

### Train on MRI Images

```bash
python main.py --config config_aligned.json --gpu 0 --seed 42
```

### Train on X-ray Images

Modify `config_aligned.json`:
```json
"dataset": {
    "name": "aligned_hip_xray",
    ...
}
```

Then run:
```bash
python main.py --config config_aligned.json --gpu 0 --seed 42
```

### Evaluate Only (No Training)

Modify `config_aligned.json`:
```json
"training": {
    "apply": false,
    ...
}
```

Then run:
```bash
python main.py --config config_aligned.json
```

### Test Different Architectures

The paper evaluates multiple architecture–encoder combinations:

**UNet++ with ResNet18 (Best Performance):**
```json
"model": {
    "name": "smp_UnetPlusPlus",
    "encoder": "resnet18",
    "encoder_weights": "imagenet"
}
```

**UNet++ with VGG16:**
```json
"model": {
    "name": "smp_UnetPlusPlus",
    "encoder": "vgg16",
    "encoder_weights": "imagenet"
}
```

**UNet with MiT-B1:**
```json
"model": {
    "name": "smp_Unet",
    "encoder": "mit_b1",
    "encoder_weights": "imagenet"
}
```

**DPT with ResNet18:**
```json
"model": {
    "name": "smp_DPT",
    "encoder": "resnet18",
    "encoder_weights": "imagenet"
}
```

**DPT with MaxViT-Base:**
```json
"model": {
    "name": "smp_DPT",
    "encoder": "tu-maxvit_base_tf_512",
    "encoder_weights": "imagenet"
}
```


### Configuration Parameters

Key parameters in `config_aligned.json`:

- **gpu**: GPU device ID (0, 1, 2...) or -1 for CPU
- **seed**: Random seed for reproducibility (17, 42, 90 used in paper)
- **model.name**: Architecture (`smp_Unet`, `smp_UnetPlusPlus`, `smp_DPT`)
- **model.encoder**: Backbone encoder (e.g., `resnet18`, `vgg16`, `mit_b1`, `tu-maxvit_base_tf_512`)
- **training.learning_rate**: Initial learning rate (1×10⁻⁴ in paper)
- **training.weight_decay**: AdamW weight decay (1×10⁻⁵ in paper)
- **training.epochs**: Maximum epochs (200 with early stopping)
- **dataset.image_size**: Input size [512, 512]
- **dataset.sigma**: Gaussian heatmap sigma (5 pixels)
- **dataloader.batch_size**: Batch size (4 in paper)
- **testing.tta**: Enable test-time augmentation (true/false)

### Results Directory Structure

Results are automatically saved to:
```
experiments/main_results/SEED/DATASET/MODEL_ENCODER/
├── exp_config.json                      # Full configuration
├── best_checkpoint.pt                   # Best model weights
├── train_val_loss.png                   # Training curves
└── test_results/
    ├── evaluation_results_detailed.json # All metrics
    ├── predictions_grid_*.png           # Visual predictions with landmarks
    ├── angles_grid_*.png                # α-angle and LCE angle visualizations
    ├── bland_altman_*.png               # Bland-Altman agreement plots
    └── cam_impingement_errors.txt       # Diagnostic analysis (TP/FP/TN/FN)
```

### Evaluation Metrics

The code reports metrics at three hierarchical levels:

**1. Landmark Localization:**
- Mean Radial Error (MRE): Average Euclidean distance between predicted and ground-truth landmarks (mm)
- Per-landmark MRE for each keypoint (FHC, NA, LAE, LCP)
- Success Detection Rate at radius r (SDR@r): Percentage of landmarks localized within r mm

**2. Clinical Angle Assessment:**
- Mean Absolute Error (MAE) between predicted and ground-truth angles
- Intraclass Correlation Coefficient ICC(2,1): Two-way random effects model quantifying absolute agreement (<0.40 poor, 0.40-0.59 fair, 0.60-0.74 good, >0.75 excellent)
- Median absolute difference as robust summary
- Bland–Altman analysis: Mean bias, 95% limits of agreement, proportional bias testing

**3. Diagnostic Performance (cam-type impingement, α > 65°):**
- Accuracy, Sensitivity, Specificity
- Positive Predictive Value (PPV), Negative Predictive Value (NPV)
- Confusion matrix (TP/FP/TN/FN)

## Citation

If you use this code or method in your research, please cite:

```bibtex
@inproceedings{divia2026automated,
  title={Automated Landmark Detection for Assessing Hip Conditions: A Cross-Modality Validation of MRI versus X-Ray},
  author={Di Via, Roberto and Pastore, Vito Paolo and Odone, Francesca and Glyn-Jones, Siˆon and Voiculescu, Irina},
  booktitle={IEEE International Symposium on Biomedical Imaging (ISBI)},
  year={2026}
}
```

## Acknowledgments

This research was conducted retrospectively using human subject data from the FAIT study, which received ethical approval. Data were collected at Oxford University as part of a multi-centre randomised controlled trial comparing surgical and non-surgical management of femoroacetabular impingement.

**Institutional Affiliations:**
- MaLGa Center, DIBRIS, University of Genoa, Italy
- Nuffield Department of Orthopaedics, Rheumatology and Musculoskeletal Sciences, Oxford, UK
- Oxford University Department of Computer Science, UK

## License

This project is licensed under the MIT License.

## Contact

For correspondence regarding this implementation, please contact: irina@cs.ox.ac.uk