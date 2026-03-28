# DeepLense GSoC 2026 - Common Test I: Multi-Class Classification

This repository contains the solution for **Common Test I** as part of the Google Summer of Code (GSoC) 2026 application process for the DeepLense project.

## 🎯 Task Objective
The goal of this project is to build a deep learning model using PyTorch to classify simulated strong gravitational lensing images into three distinct categories:
1. `no`: Strong lensing images with no substructure
2. `sphere`: Strong lensing images with subhalo substructure
3. `vort`: Strong lensing images with vortex substructure

## 📁 Repository Structure
* `main_project.ipynb`: The primary Jupyter Notebook containing the complete end-to-end pipeline (data preprocessing, splitting, model definition, training, and evaluation).
* `lens_classifier_final.pth`: The saved PyTorch model weights for the best-performing training epoch.
* `README.md`: Project documentation and strategy overview.

## 🛠️ Methodology & Strategy

### 1. Data Pipeline & 90:10 Split
The dataset consists of `.npy` files containing min-max normalized grayscale images. The notebook implements a custom PyTorch `Dataset` class to handle these files directly. To meet the evaluation requirements, a custom splitting function was developed to merge the provided data and strictly enforce a stratified **90:10 Train-Validation split**, ensuring class balance is preserved. 

Conservative data augmentation (random horizontal and vertical flips) was applied to the training set to improve generalization without distorting the scientific structures of the lensing images.

### 2. Model Architecture
A **Custom Convolutional Neural Network (CNN)** was selected as the baseline model. 
* **Why:** CNNs are highly effective for extracting hierarchical spatial features from grid-like topological data. Given the relatively small spatial dimensions and single-channel nature of the images, a custom architecture provides an efficient and clean baseline.
* **Structure:** The model progressively increases channel depth (32 → 64 → 128 → 256) through four convolutional blocks featuring Batch Normalization, ReLU activations, and Max Pooling. A Dropout layer (p=0.4) is included in the final fully connected classifier to prevent overfitting.

### 3. Training & Optimization
* **Loss Function:** `CrossEntropyLoss` is utilized as this is a mutually exclusive multi-class classification problem.
* **Optimizer:** Adam optimizer with an initial learning rate of `0.001`, paired with a `ReduceLROnPlateau` scheduler to dynamically lower the learning rate if validation loss plateaus.

### 4. Evaluation Metrics
The model's performance is comprehensively evaluated on the 10% validation split. The notebook outputs:
* A detailed **Confusion Matrix** and **Classification Report**.
* **ROC Curves** generated using a One-vs-Rest strategy.
* Calculated **Macro and Weighted AUC Scores**, fully satisfying the task's required evaluation metrics.

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have the required dependencies installed:
   ```bash
   pip install torch torchvision scikit-learn matplotlib seaborn tqdm