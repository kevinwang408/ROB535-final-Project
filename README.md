# ROB 535 Final Project: DETR Implementation

## 1. Project Overview
This repository contains the implementation of the DETR (DEtection TRansformer) model for the ROB 535 final project. The codebase is based on MMDetection but has been streamlined to focus specifically on reproducing the experimental results for this project.

## 2. Environment Setup

To ensure reproducibility, please follow these steps to set up the environment.

### Prerequisites
*   Python 3.8+
*   PyTorch 1.13+ (compatible with your CUDA version)
*   CUDA (if running on GPU)

### Installation Steps

1.  **Clone the repository (if not already local):**
    ```bash
    git clone https://github.com/kevinwang408/ROB535-final-Project.git
    cd ROB535-final-Project
    ```

2.  **Install dependencies:**
    This project relies on specific versions of packages. Please install them using the provided requirements file:
    ```bash
    conda install --file requirements.txt
    ```

3.  **Install MMDetection in editable mode:**
    This allows the code in `mmdet/` to be imported directly.
    ```bash
    pip install -v -e .
    ```

## 3. Project Structure

*   `configs/`: Contains the configuration files (hyperparameters, dataset paths) for the DETR model.
*   `mmdet/`: The core source code for the detection framework.
*   `tools/`: Helper scripts for training (`train.py`) and testing (`test.py`).
*   `work_dirs/`: (Auto-generated) Checkpoints and logs will be saved here during training.

## 4. How to reproduce the result

Since the trained weight files (`.pth`) exceed GitHub's file size limit, they are hosted externally along with the training logs. Please follow the steps below to reproduce our results.

### Step 1: Download Weights, Logs, and Data
Download the `work_dirs` zips file which contian the pre-trained weights (`epoch_50.pth`) and training logs from the following link:

* **['work_dirs' Download Link]**: https://drive.google.com/file/d/1ey_vJHzSuUOhf9fAtgamIHwhH5ppdcPb/view?usp=sharing

Download the data used to train DETR model.

* **['data' Download Link]**: https://drive.google.com/file/d/1-Ei2spXLw5hhwuEL40x817hE4pn0_kAP/view?usp=sharing

### Step 2: Organize Directory Structure
After downloading, unzip or place the files into the root directory of this project. Your directory structure should look exactly like this to ensure the scripts can find the weights:

```text
Project_Root/
├── configs/
├── Data/
├── tools/
├── ...
└── work_dirs/
    ├── deformable_detr_kitti/
    │   ├── epoch_50.pth        <-- Pre-trained weights for KITTI
    │   └── 202412xxx.log       <-- Training logs
    │
    └── deformable_detr_bdd_ninja/
        ├── epoch_50.pth        <-- Pre-trained weights for BDD100K
        └── 202412xxx.log       <-- Training logs
```

### Step 3: Run Evaluation

Use the `tools/test.py` script to evaluate the model on the test dataset using the downloaded weights.

**1. Reproduce KITTI Results:**

```bash
python tools/test.py kitti_config.py work_dirs/deformable_detr_kitti/epoch_50.pth --eval bbox
```

**2. Reproduce BDD Results:**

```bash
python tools/test.py bdd_ninja_config.py work_dirs/deformable_detr_bdd_ninja/epoch_50.pth --eval bbox
```