# Homework 2: Transfer Learning and Fine-Tuning
**Course:** Computer Vision

## Objective
This repository contains a deep learning pipeline designed to evaluate and compare different Transfer Learning strategies using a pre-trained Convolutional Neural Network (**MobileNetV3-Small**) on the **Oxford-IIIT Pet Dataset** (classification across 37 animal breeds).

## File Structure
- `data/`: Local directory where the Oxford-IIIT Pet dataset is automatically downloaded.
- `dataset_setup.py`: Handles data loading, preprocessing, stratified data splitting (80% train / 20% validation), and universal seeding for reproducibility.
- `trainer.py`: Contains helper functions for model training, validation loops, evaluation metrics calculation, confusion matrix plotting, and prediction visualization.
- `1_approach_finetuning.py`: Implements and runs **Approach 1** (Full Fine-Tuning of the entire network).
- `2_approach_headtraining.py`: Implements and runs **Approach 2** (Classifier Head-Training with frozen feature extractor).
- `3_approach_exfeature.py`: Implements and runs **Approach 3** (Feature Extraction combined with linear and RBF Support Vector Machines).
- `checkpoints/`: Contains the saved neural model weights (`.pth`). SVM `.pkl` models are excluded from the repository due to file size and are generated locally on run.
- `plots/`: Automatically generated directory containing high-resolution confusion matrices and prediction grids (`.png`).
- `outputs.txt`: Contains the complete execution logs and per-class test classification reports.

## Environment & Dependencies
To ensure maximum compatibility and proper execution across platforms (Apple Silicon MPS, CUDA, and CPU), the project requires the following environment:
- **Python:** 3.15
- **PyTorch / TorchVision:** `torch: 2.15.0`, `torchvision: 0.29.0`
- **Scikit-Learn:** `scikit-learn: 1.9.1`
- **Data Manipulation & Plotting:** `numpy`, `matplotlib`, `seaborn`, `joblib`

## Dataset Instructions
The **Oxford-IIIT Pet Dataset** is used for this project. No manual download is required. The `dataset_setup.py` script automatically downloads the dataset via `torchvision.datasets` into the `data/` directory. The official `trainval` partition is split into an 80% training and 20% validation set using stratified sampling to maintain class balance.

## Hyperparameters & Checkpoints
Fixed hyperparameter configurations were used for all approaches (no systematic tuning). For Approaches 1 and 2, the validation set was used only to save the checkpoint from the best epoch; Approach 3 did not use the validation set.

*   **Approach 1 (Full Fine-Tuning):**
    *   Optimizer: Adam
    *   Learning Rate: 1e-4
    *   Epochs: 5
    *   Checkpoint: `checkpoints/checkpoint_approach_1.pth`
*   **Approach 2 (Classifier Head-Training):**
    *   Optimizer: Adam
    *   Learning Rate: 1e-3
    *   Epochs: 5
    *   Checkpoint: `checkpoints/checkpoint_approach_2.pth`
*   **Approach 3 (Feature Extraction with SVM):**
    *   Classifiers: SVM (Kernel: Linear) and SVM (Kernel: RBF) with default parameters
    *   Checkpoints (generated locally, not included in repository): `checkpoints/checkpoint_approach_3_linear.pkl` and `checkpoints/checkpoint_approach_3_rbf.pkl`

## Reproducibility & Experimental Settings
A universal seeding function (`set_seed(42)`) is implemented in `dataset_setup.py` to fix randomness across Python's `random` module, NumPy, and PyTorch (CPU, CUDA, and MPS), alongside `cudnn.deterministic` and `cudnn.benchmark` flags for CUDA architectures. Note that experiments were executed on Apple Silicon (MPS), which does not guarantee strict bit-for-bit determinism across runs. The test set was strictly held out and not used for model selection or hyperparameter tuning.

## How to Run the Pipeline
To run the experiments, regenerate the reported metrics, and save the plots and checkpoints, ensure your environment is active and execute the respective Python scripts in your terminal:

1. Run the Full Fine-Tuning approach:
   ```bash
   python 1_approach_finetuning.py
   ```

2. Run the Classifier Head-Training approach:
   ```bash
   python 2_approach_headtraining.py
   ```

3. Run the Feature Extraction and SVM classification approach:
   ```bash
   python 3_approach_exfeature.py
   ```