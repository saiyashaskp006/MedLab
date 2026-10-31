# 👁️ RetinaXAI: Trustworthy & Explainable AI for Diabetic Retinopathy

A clinical-grade, end-to-end medical diagnostic and decision support tool for **Diabetic Retinopathy (DR)** from retinal fundus imaging. Built with **PyTorch (ResNet50)**, **Ben Graham's illumination-equalized preprocessing**, **Grad-CAM visual attribution**, and an interactive **Streamlit** clinical dashboard.

---
## 🚀 Live Demo

[Visit RetinaXAI Application](https://saiyashaskp006-medlab-app-x5aoe1.streamlit.app/)


## 📋 Table of Contents
1. [Clinical Problem Background](#clinical-problem-background)
2. [Diagnostic & Preprocessing Pipeline](#diagnostic--preprocessing-pipeline)
3. [Project Directory Structure](#project-directory-structure)
4. [Quickstart: Running Locally](#quickstart-running-locally)
5. [Model Training on Google Colab & Kaggle](#model-training-on-google-colab--kaggle)
6. [Explainable AI (XAI) & Grad-CAM](#explainable-ai-xai--grad-cam)
7. [Online Deployment via Streamlit Community Cloud](#online-deployment-via-streamlit-community-cloud)
8. [Clinical Decision Support & Triage Matrix](#clinical-decision-support--triage-matrix)

---

## 1. Clinical Problem Background

Diabetic Retinopathy (DR) is a microvascular complication of diabetes and the leading cause of preventable blindness in working-age adults worldwide. Early detection via routine retinal screening reduces the risk of severe vision loss by over 95%.

The **International Clinical Diabetic Retinopathy (ICDR)** disease severity scale:

| Grade | Disease Stage | Key Pathological Retinal Hallmark | Recommended Clinical Timeline |
|:---:|:---|:---|:---|
| **0** | **No Diabetic Retinopathy** | No visible microvascular abnormalities | Routine annual screening |
| **1** | **Mild NPDR** | Microaneurysms only (isolated focal capillary dilatations) | Follow-up in 6–12 months |
| **2** | **Moderate NPDR** | Microaneurysms, dot & blot hemorrhages, hard lipid exudates | Comprehensive ophthalmologist review (2–4 months) |
| **3** | **Severe NPDR** | International 4-2-1 rule (>20 hemorrhages in 4 quadrants, venous beading in 2+, IRMA in 1+) | Urgent retina specialist referral (2–4 weeks) |
| **4** | **Proliferative DR (PDR)** | Neovascularization (NVD/NVE), preretinal or vitreous hemorrhage | Emergency intervention (<48h: PRP laser or anti-VEGF) |

---

## 2. Diagnostic & Preprocessing Pipeline

```
Raw Fundus Image
       │
       ▼
[Ben Graham Retinal Preprocessing]
  ├─ 1. Threshold & crop non-informative black margin
  ├─ 2. Resize to 224 × 224 px
  ├─ 3. Illumination compensation: 4*img - 4*GaussianBlur(10) + 128
  └─ 4. Circular aperture mask
       │
       ▼
[Deep Convolutional Neural Network (ResNet50)]
  ├─ ImageNet transfer learning backbone
  ├─ Dropout (0.3) + Linear(2048, 512) + BatchNorm + ReLU + Linear(512, 5)
  └─ Softmax probability distribution across ICDR Grades 0–4
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[Clinical Decision Support]     [Explainable AI (XAI)]
  ├─ Severity classification      ├─ Grad-CAM gradient feature attribution
  ├─ Triage action plan           ├─ Pathology lesion localization heatmap
  └─ Standardized PDF/Text report └─ Patch occlusion sensitivity analysis
```

---

## 3. Project Directory Structure

```
MedLab/
├── app.py                     # Streamlit clinical web application
├── requirements.txt           # Dependencies for local and cloud deployment
├── README.md                  # Comprehensive system documentation
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Ben Graham illumination & circular crop algorithms
│   ├── model.py               # RetinalResNet50 architecture & ICDR knowledge base
│   ├── xai.py                 # Grad-CAM engine & patch sensitivity analyzer
│   ├── train.py               # Training loop with AMP, QWK, and checkpointing
│   ├── create_samples.py      # Generates synthetic fundus samples for Grades 0–4
│   └── create_baseline_model.py # Baseline calibration checkpoint generator
├── notebooks/
│   └── diabetic_retinopathy_training.ipynb  # 1-Click Google Colab / Kaggle notebook
├── samples/                   # Representative fundus images (Grades 0 to 4)
├── models/
│   └── dr_resnet50_baseline.pth # Calibrated baseline model checkpoint
└── tests/
    └── test_pipeline.py       # Automated test suite
```

---

## 4. Quickstart: Running Locally

### Step 1: Install Dependencies
Ensure you are using Python 3.10+ (tested on Python 3.10 to 3.14):
```bash
pip install -r requirements.txt
```

### Step 2: Run the Diagnostic Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 3: Run the Verification Suite
```bash
python tests/test_pipeline.py
```

---

## 5. Model Training on Google Colab & Kaggle

Large-scale training on the **APTOS 2019 Blindness Detection** (3,662 images) or **EyePACS** (35,126 images) dataset requires a GPU:

1. Open [`notebooks/diabetic_retinopathy_training.ipynb`](notebooks/diabetic_retinopathy_training.ipynb) in **Google Colab** or upload to **Kaggle Notebooks**.
2. Select runtime: **GPU** (T4, P100, or A100).
3. If on Kaggle: Click **Add Data** -> Search `aptos2019-blindness-detection`.
4. Run all cells:
   - Stratified 80/20 train/validation split.
   - Mixed precision training with PyTorch `torch.amp.autocast()`.
   - Primary clinical evaluation metric: **Quadratic Weighted Kappa (QWK)**.
5. Download the saved `dr_resnet50_best.pth` and place it in the `models/` directory for the Streamlit web app.

---

## 6. Explainable AI (XAI) & Grad-CAM

Deep neural networks in healthcare must never operate as opaque "black boxes." 

### How Grad-CAM Works:
1. **Target Layer**: Captures feature activations from `resnet50.layer4[-1]` (the last residual block before global average pooling).
2. **Gradients**: Computes $\frac{\partial Y^c}{\partial A^k}$ with respect to predicted ICDR class score $Y^c$.
3. **Global Average Pooling**: Computes channel weights $\alpha_k^c$:
   $$\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$
4. **ReLU Activation**: Retains features with positive attribution:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$
5. **Lesion Overlay**: Normalizes activations $[0, 1]$ and overlays color heatmaps (`cv2.COLORMAP_JET` / `INFERNO`) onto the fundus view with adjustable opacity ($\alpha$) and activation cutoff threshold.

---

## 7. Online Deployment via Streamlit Community Cloud

You can deploy this web application publicly for free in less than 3 minutes:

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Deploy RetinaXAI clinical diagnostic tool"
   git branch -M main
   git remote add origin https://github.com/<YOUR_USERNAME>/<REPO_NAME>.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **New app** -> Select your repository, branch (`main`), and main file path (`app.py`).
4. Click **Deploy!** Your clinical diagnostic tool is now live with a public URL!

---

## 8. Clinical Decision Support & Triage Matrix

The web application automatically formats a structured clinical report containing:
- **Patient Identifier & Exam Date**
- **Predicted ICDR Severity Grade & Multiclass Probability Breakdown**
- **Grad-CAM Visual Lesion Attribution Assessment**
- **Actionable Triage Protocol & Referral Urgency**
- **1-Click Text/Markdown Report Download**

---

## ⚖️ Clinical Disclaimer
This software is designed for clinical research, physician assistance, and medical education. It does not replace a comprehensive ophthalmic exam, fluorescein angiography (FA), or optical coherence tomography (OCT) performed by a licensed eye care specialist.
