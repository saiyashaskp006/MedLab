"""
Diabetic Retinopathy Clinical Decision Support & Explainable AI (XAI) System.
Web application powered by PyTorch, ResNet50, Ben Graham Preprocessing, and Grad-CAM.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch
import streamlit as st

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.preprocessing import preprocess_for_inference, ben_graham_preprocessing, crop_retina_border
from src.model import build_model, predict_fundus, DR_CLASSES
from src.xai import GradCAM, overlay_heatmap_on_image, compute_patch_attribution

# Streamlit Page Configuration
st.set_page_config(
    page_title="RetinaXAI - Diabetic Retinopathy Diagnostic System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clinical styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.2rem;
    }
    .metric-box {
        border-radius: 8px;
        padding: 16px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }
    .alert-card {
        padding: 14px 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
        font-size: 0.95rem;
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #718096;
        border-top: 1px solid #e2e8f0;
        padding-top: 12px;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_model():
    """Loads and caches the deep learning model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Check if custom checkpoint exists
    checkpoint_candidates = [
        "models/dr_resnet50_best.pth",
        "models/dr_resnet50_baseline.pth"
    ]
    ckpt = None
    for cand in checkpoint_candidates:
        if os.path.exists(cand):
            ckpt = cand
            break
            
    model = build_model(
        model_name="resnet50",
        num_classes=5,
        pretrained=True,
        checkpoint_path=ckpt,
        device=device
    )
    cam = GradCAM(model)
    return model, cam, device


# App Header
st.markdown('<div class="main-header">👁️ RetinaXAI: Diabetic Retinopathy Diagnostic & XAI System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Clinical Decision Support Tool with Ben Graham Preprocessing and Grad-CAM Explainable AI</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/ophthalmology.png", width=64)
    st.title("Control Panel")
    
    st.subheader("1. Retinal Image Source")
    input_source = st.radio(
        "Select Image Input Method:",
        ["Clinical Test Cases (Built-in)", "Upload Patient Fundus Image"],
        index=0
    )
    
    selected_image = None
    sample_key = None
    
    if input_source == "Clinical Test Cases (Built-in)":
        sample_options = {
            "Grade 0: Normal Fundus (Healthy Retina)": "samples/grade_0_normal.png",
            "Grade 1: Mild NPDR (Microaneurysms)": "samples/grade_1_mild.png",
            "Grade 2: Moderate NPDR (Hemorrhages & Exudates)": "samples/grade_2_moderate.png",
            "Grade 3: Severe NPDR (Extensive Bleeds & Cotton Wool)": "samples/grade_3_severe.png",
            "Grade 4: Proliferative DR (Neovascularization)": "samples/grade_4_proliferative.png"
        }
        sample_choice = st.selectbox("Select Patient Case:", list(sample_options.keys()), index=2)
        sample_path = sample_options[sample_choice]
        if os.path.exists(sample_path):
            selected_image = Image.open(sample_path).convert("RGB")
        else:
            st.error(f"Sample file {sample_path} not found.")
    else:
        uploaded_file = st.file_uploader(
            "Upload Retinal Fundus Photograph (JPG / PNG):",
            type=["png", "jpg", "jpeg"]
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
            
    st.markdown("---")
    st.subheader("2. Explainable AI (XAI) Parameters")
    alpha = st.slider("Grad-CAM Heatmap Opacity (α):", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    threshold = st.slider("Lesion Activation Threshold:", min_value=0.0, max_value=0.8, value=0.25, step=0.05,
                          help="Filters low activations to emphasize focal lesions.")
    colormap_choice = st.selectbox("Color Mapping:", ["Jet", "Inferno", "Viridis"], index=0)
    cm_dict = {"Jet": cv2.COLORMAP_JET, "Inferno": cv2.COLORMAP_INFERNO, "Viridis": cv2.COLORMAP_VIRIDIS}
    selected_colormap = cm_dict[colormap_choice]
    
    st.markdown("---")
    st.markdown("""
    **System Status:**
    - Model: **ResNet50 Backbone**
    - Resolution: **224 × 224 px**
    - Preprocessing: **Ben Graham Method**
    - Explainability: **Grad-CAM + Patch Occlusion**
    """)


# Ensure model is ready
with st.spinner("Initializing Deep Learning Engine & PyTorch Grad-CAM Hooks..."):
    model, grad_cam, device = load_cached_model()

if selected_image is None:
    st.info("👈 Please select a clinical sample test case or upload a patient fundus photograph from the sidebar to begin.")
else:
    # 1. Image Preprocessing
    with st.spinner("Executing Ben Graham illumination correction and circular masking..."):
        preprocessed_np, tensor_batch = preprocess_for_inference(selected_image, target_size=224)
        
    # 2. Diagnostic Prediction
    with st.spinner("Running deep neural network inference..."):
        results = predict_fundus(model, tensor_batch, device=device)
        pred_class = results["pred_class"]
        
    # 3. Grad-CAM Generation
    with st.spinner("Synthesizing Grad-CAM gradient feature attribution..."):
        heatmap, target_cls, score = grad_cam.generate_heatmap(tensor_batch.to(device), target_class=pred_class)
        overlay_preprocessed = overlay_heatmap_on_image(
            preprocessed_np, heatmap, alpha=alpha, colormap=selected_colormap, threshold=threshold
        )
        # Also overlay on cropped raw image for clinical comparison
        raw_np = np.array(selected_image)
        cropped_raw = cv2.resize(crop_retina_border(raw_np), (224, 224))
        overlay_raw = overlay_heatmap_on_image(
            cropped_raw, heatmap, alpha=alpha, colormap=selected_colormap, threshold=threshold
        )

    # Main Layout Tabs
    tab_diag, tab_xai, tab_prep, tab_guide, tab_report = st.tabs([
        "📊 Diagnostic Findings & Triage",
        "🔍 Explainable AI (Grad-CAM)",
        "🧪 Preprocessing Inspection",
        "📖 Clinical ICDR Guidelines",
        "📄 Clinical Report Export"
    ])
    
    # -------------------------------------------------------------
    # TAB 1: Diagnostic Findings & Triage
    # -------------------------------------------------------------
    with tab_diag:
        # Top KPI cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Predicted Severity", results["short_name"])
        with col2:
            st.metric("ICDR Severity Grade", f"Grade {pred_class} / 4")
        with col3:
            st.metric("Diagnostic Confidence", f"{results['confidence'] * 100:.1f}%")
        with col4:
            st.metric("Clinical Urgency", results["triage"])

        st.markdown("<br>", unsafe_allow_html=True)
        
        c_left, c_right = st.columns([1.1, 1.3])
        
        with c_left:
            st.markdown("#### Patient Retinal View & Target Lesions")
            st.image(overlay_preprocessed, caption=f"Grad-CAM Heatmap Overlay ({results['short_name']})", use_container_width=True)
            
        with c_right:
            st.markdown("#### Multi-Class Probability Distribution")
            prob_df = pd.DataFrame({
                "ICDR Stage": list(results["probabilities"].keys()),
                "Probability": list(results["probabilities"].values())
            })
            st.bar_chart(prob_df.set_index("ICDR Stage"), color="#1f77b4")
            
            # Clinical Action Callout
            severity_colors = {0: "#e6fffa", 1: "#e6f7ff", 2: "#fffbe6", 3: "#fff7e6", 4: "#fff1f0"}
            border_colors = {0: "#38b2ac", 1: "#1890ff", 2: "#faad14", 3: "#fa8c16", 4: "#f5222d"}
            
            st.markdown(f"""
            <div style="background-color: {severity_colors.get(pred_class, '#f8fafc')};
                        border-left: 5px solid {border_colors.get(pred_class, '#4a5568')};
                        padding: 16px; border-radius: 4px; margin-top: 15px;">
                <h4 style="margin: 0 0 8px 0; color: #1a202c;">Clinical Triage Plan: {results['triage']}</h4>
                <p style="margin: 0 0 6px 0;"><strong>Follow-up Timeline:</strong> {results['urgency']}</p>
                <p style="margin: 0 0 6px 0;"><strong>Key Pathological Hallmark:</strong> {results['findings']}</p>
                <p style="margin: 0;"><strong>Action Recommendation:</strong> {results['recommendation']}</p>
            </div>
            """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 2: Explainable AI (Grad-CAM & Pathology Map)
    # -------------------------------------------------------------
    with tab_xai:
        st.markdown("### Model Explainability & Visual Feature Attribution")
        st.markdown("""
        To establish clinician trust, **Grad-CAM** computes the gradients of the predicted diagnostic score with respect 
        to the final convolutional feature maps (`layer4` of ResNet50). 
        The warmer zones (red/yellow) indicate focal retinal regions that contributed most to the diagnosis.
        """)
        
        x_col1, x_col2, x_col3 = st.columns(3)
        with x_col1:
            st.markdown("**1. Preprocessed Fundus Image**")
            st.image(preprocessed_np, caption="Ben Graham Illumination Equalized", use_container_width=True)
            
        with x_col2:
            st.markdown("**2. Raw Grad-CAM Activation Heatmap**")
            heatmap_norm = np.uint8(255 * heatmap)
            st.image(cv2.applyColorMap(heatmap_norm, selected_colormap), caption="Gradient Activation Intensity", use_container_width=True)
            
        with x_col3:
            st.markdown("**3. Clinical Heatmap Overlay**")
            st.image(overlay_preprocessed, caption=f"Lesion Overlay (α = {alpha}, cutoff = {threshold})", use_container_width=True)

        st.markdown("---")
        st.markdown("#### Clinical Pathology Interpretation Guide")
        
        diag_interpretations = {
            0: "The heatmap shows diffuse, low-amplitude background activation without focal clustering. No localized microvascular pathology is driving classification.",
            1: "Localized focus is concentrated near small capillary dilatations (microaneurysms), typically in the macula or temporal vascular arcades.",
            2: "Attribution hotspots correspond to retinal hemorrhages (dot/blot bleeds) and lipid hard exudates indicative of vascular breakdown.",
            3: "Broad, extensive multi-quadrant activations reflecting diffuse intraretinal microvascular abnormalities (IRMA) and severe blot hemorrhages complying with the 4-2-1 rule.",
            4: "High-intensity activation highlights neovascularization (abnormal fragile fronds on the optic disc or arcade) and preretinal vitreous hemorrhage."
        }
        st.info(f"**Diagnostic Attribution for {results['class_name']}:**\n\n{diag_interpretations.get(pred_class, '')}")

        # Optional Patch Sensitivity Analysis
        with st.expander("🧪 Advanced: Run Patch Occlusion Sensitivity Analysis (LIME-style superpixel perturbation)"):
            if st.button("Compute Patch Occlusion Sensitivity Map"):
                with st.spinner("Evaluating patch occlusions across retinal grid..."):
                    patch_map = compute_patch_attribution(model, tensor_batch, target_class=pred_class, patch_size=28, device=device)
                    patch_overlay = overlay_heatmap_on_image(preprocessed_np, patch_map, alpha=0.55, colormap=cv2.COLORMAP_MAGMA)
                    st.image(patch_overlay, caption="Patch Occlusion Sensitivity (Regions causing maximum probability drop upon removal)", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: Retinal Preprocessing Inspection
    # -------------------------------------------------------------
    with tab_prep:
        st.markdown("### Medical Preprocessing: Ben Graham's Retinal Normalization")
        st.markdown("""
        Variations in pupil dilation, retinal pigmentation, and camera optical lighting create significant domain shift in clinical practice. 
        Ben Graham's award-winning algorithm standardizes fundus images through:
        1. **Retinal Border Isolation**: Automatic segmentation and removal of non-informative black borders.
        2. **Local Lighting Equalization**: Gaussian blur subtraction (`4 * img - 4 * GaussianBlur + 128`) to normalize exposure across the macula and periphery.
        3. **Circular Aperture Masking**: Masks perimeter noise to prevent edge artifacts.
        """)
        
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("#### Raw Fundus Photograph")
            st.image(selected_image, caption=f"Original Resolution: {selected_image.size[0]} × {selected_image.size[1]}", use_container_width=True)
            
        with p_col2:
            st.markdown("#### Ben Graham Preprocessed Retinal View")
            st.image(preprocessed_np, caption="Normalized 224 × 224 px, Illumination Subtracted", use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: Clinical ICDR Guidelines
    # -------------------------------------------------------------
    with tab_guide:
        st.markdown("### International Clinical Diabetic Retinopathy (ICDR) Scale")
        st.markdown("""
        The ICDR severity grading scale is the worldwide clinical standard adopted by the American Academy of Ophthalmology (AAO) and WHO:
        """)
        
        guide_data = []
        for grade_id, info in DR_CLASSES.items():
            guide_data.append({
                "Grade": grade_id,
                "Classification": info["name"],
                "Key Fundus Examination Findings": info["findings"],
                "Triage & Action Timeline": info["urgency"]
            })
        st.table(pd.DataFrame(guide_data))

    # -------------------------------------------------------------
    # TAB 5: Clinical Report Export
    # -------------------------------------------------------------
    with tab_report:
        st.markdown("### Standardized Clinical Decision Support Report")
        
        patient_id = st.text_input("Patient / Record ID:", value="PT-2026-09821")
        exam_date = datetime.date.today().strftime("%Y-%m-%d")
        
        report_text = f"""================================================================================
DIABETIC RETINOPATHY CLINICAL DECISION SUPPORT REPORT
================================================================================
Date: {exam_date}
Patient Identifier: {patient_id}
Imaging Modality: Color Fundus Photography
Deep Learning Architecture: ResNet50 (Transfer Learning) with Grad-CAM

--------------------------------------------------------------------------------
DIAGNOSTIC FINDINGS
--------------------------------------------------------------------------------
Predicted ICDR Severity: Grade {pred_class} - {results['class_name']}
Model Confidence: {results['confidence'] * 100:.2f}%
Probability Breakdown:
  - Grade 0 (No DR):         {results['probabilities']['No DR'] * 100:.1f}%
  - Grade 1 (Mild NPDR):     {results['probabilities']['Mild NPDR'] * 100:.1f}%
  - Grade 2 (Moderate NPDR): {results['probabilities']['Moderate NPDR'] * 100:.1f}%
  - Grade 3 (Severe NPDR):   {results['probabilities']['Severe NPDR'] * 100:.1f}%
  - Grade 4 (Proliferative): {results['probabilities']['Proliferative DR'] * 100:.1f}%

--------------------------------------------------------------------------------
EXPLAINABLE AI (XAI) ATTRIBUTION
--------------------------------------------------------------------------------
Methodology: Gradient-weighted Class Activation Mapping (Grad-CAM)
Target Layer: ResNet50 layer4 Bottleneck
Clinical Correlation: {diag_interpretations.get(pred_class, '')}

--------------------------------------------------------------------------------
CLINICAL TRIAGE & RECOMMENDATION
--------------------------------------------------------------------------------
Triage Status: {results['triage']}
Urgency: {results['urgency']}
Pathological Criteria: {results['findings']}
Recommended Action: {results['recommendation']}

================================================================================
DISCLAIMER: This diagnostic summary was generated by an AI clinical decision support system.
It must be reviewed and verified by a licensed ophthalmologist or optometrist before initiating
any clinical intervention.
================================================================================
"""
        st.text_area("Report Preview:", report_text, height=360)
        st.download_button(
            label="📥 Download Clinical Report (.txt)",
            data=report_text,
            file_name=f"DR_Report_{patient_id}_{exam_date}.txt",
            mime="text/plain"
        )

# Clinical Disclaimer Footer
st.markdown("""
<div class="disclaimer">
    <strong>Clinical Decision Support Disclaimer:</strong> This application is intended for clinical research, education, and triage decision support. 
    It is not a replacement for a formal ophthalmological examination, fluorescein angiography, or optical coherence tomography (OCT).
</div>
""", unsafe_allow_html=True)
