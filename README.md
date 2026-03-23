✨ STA-Diff ✨

Stage-Adaptive Attention Diffusion Model for Unsupervised Brain MRI Anomaly Detection

This repository provides the official PyTorch implementation of our work:

"Stage-Adaptive Attention in Diffusion Models for Unsupervised Brain MRI Anomaly Detection"

🎨 Approach

🔍 Key Idea

STA-Diff introduces a Stage-Adaptive Attention (STA) mechanism into the diffusion UNet to better handle anomaly detection in brain MRI.

Unlike conventional diffusion models that treat all timesteps equally, our method:

🧠 Adapts attention across diffusion stages
🎯 Focuses on anatomical consistency at early stages
🔥 Enhances anomaly sensitivity at later stages
🧩 Preserves healthy regions while highlighting abnormal structures

This leads to:

More stable reconstruction
Better anomaly localization
Reduced false positives in normal tissues
🚀 Getting Started
🛠️ Environment Setup

We use Python 3.11 and PyTorch.

pip3 install -r requirements.txt

📁 Dataset
We evaluate on brain MRI datasets (BraTS21).

Data Preparation

If you are using the raw BraTS21 dataset, please preprocess it first with:

python data/preprocess.py

Alternatively, you may directly use our preprocessed dataset, available here: https://drive.google.com/file/d/1U5W1vJalDECpqknyBXfX6-dQ6FWgz5pW/view?usp=drive_link

The preprocessed data is stored in .npy format for efficient training and evaluation.

Expected Structure

data/

    train/
    ...
    test/
        t1/
        t2/
        flair/
        t1ce/
        seg/
    val/
    ...

    
🏋️ Training

Train STA-Diff with:

torchrun train_STA_Diff.py \
  --dataset brat \
  --data-dir ./data \
  --model-size UNet_M \
  --object-category all \
  --image-size 288 \
  --center-size 256 \
  --center-crop True \
  --use-sta True
  
🔧 Key Options
Argument	Description
--use-sta	Enable Stage-Adaptive Attention
--model-size	UNet scale (S / M / L)
--image-size	Input resolution
--center-crop	Use center cropping

🧪 Testing
Standard Evaluation
python evaluation_DeCo_Diff.py \
  --dataset brat \
  --data-dir ./data \
  --model-size UNet_M \
  --object-category all \
  --anomaly-class all \
  --image-size 288 \
  --center-size 256 \
  --center-crop true \
  --model-path path/checkpoints/last.pt 


📊 Results
| Modality  | DSC  | Sensitivity | Precision | SSIM  |
| --------- | ---- | ----------- | --------- | ----- |
| **FLAIR** | 86.1 | 86.7        | 93.3      | 0.960 |
| **T2**    | 81.8 | 78.7        | 89.1      | 0.955 |
| **T1CE**  | 61.4 | 67.2        | 64.5      | 0.921 |
| **T1**    | 55.5 | 57.4        | 41.1      | 0.919 |


💡 Observations
Better preservation of healthy tissue structure
More accurate localization of large lesions
Robust across multi-modal MRI inputs
📸 Qualitative Results

✨ Highlights
Clear anomaly boundaries
Reduced noise in normal regions
Stable reconstruction across modalities
📚 Citation

If you find this work useful, please cite:


🧠 Acknowledgements

This project builds upon:

Diffusion models
Latent Diffusion Models (LDM)
Medical anomaly detection frameworks
🔒 Notes
This repository is intended for research purposes only
For clinical use, further validation is required
