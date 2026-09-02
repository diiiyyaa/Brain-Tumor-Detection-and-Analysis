# 🧠 NeuroSight AI — Brain Tumor Detection & Analysis

### Detect → Segment → Measure → Explain → Report

NeuroSight AI is an AI-powered brain MRI analysis platform designed to assist in the detection, segmentation, quantitative analysis, and visualization of brain tumors from MRI scans.

The system combines **Deep Learning, Computer Vision, Explainable AI, and Medical Image Analysis** into a unified interactive dashboard.

> ⚠️ **Medical Disclaimer:** NeuroSight AI is a research and educational prototype. It is not a medical device and should not be used as a substitute for diagnosis or treatment by qualified medical professionals.

---

## 🚀 Live Demo

🌐 **Try the deployed application:**

👉 https://brain-tumor-detection-and-analysis-jppgkbd2lxqq4oda79wnzm.streamlit.app/

📂 **GitHub Repository:**

👉 [**Brain Tumor Detection & Analysis →**](https://github.com/diiiyyaa/Brain-Tumor-Detection-and-Analysis)

---

## ✨ Key Features

### 🔍 1. Brain Tumor Classification

A Convolutional Neural Network (CNN) analyzes the uploaded MRI scan and predicts whether a tumor is present.

**Output includes:**

- 🧠 Tumor / No Tumor classification
- 📊 Prediction confidence
- 🔬 AI-based image analysis

---

### 🎯 2. Tumor Segmentation

A **U-Net segmentation model** identifies the tumor region at the pixel level.

The system produces:

- Original MRI
- Predicted tumor mask
- Tumor overlay
- Segmented tumor visualization

This allows users to understand **where the tumor is located**, rather than only receiving a classification result.

---

### 📏 3. Quantitative Tumor Analysis

After segmentation, NeuroSight AI extracts quantitative information from the predicted tumor region.

The dashboard can provide measurements such as:

- Tumor area
- Tumor percentage
- Bounding box
- Centroid
- Width
- Height
- Shape-related characteristics

These measurements help convert the segmentation output into interpretable numerical information.

---

### 🔥 4. Explainable AI — Grad-CAM

NeuroSight AI uses **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize the regions of the MRI that influenced the CNN's prediction.

This improves interpretability by showing:

> **"Where did the AI focus when making its prediction?"**

The dashboard provides a visual heatmap of the model's important regions.

---

### 📊 5. Interactive Streamlit Dashboard

The complete AI pipeline is integrated into an interactive Streamlit interface.

The dashboard provides:

- MRI upload
- AI prediction
- Segmentation visualization
- Quantitative measurements
- Grad-CAM explanation
- Model information
- Clinical-style report generation
- Emergency triage demonstration

---

### 📄 6. Clinical PDF Report

NeuroSight AI can generate a structured PDF report containing the analysis results.

The report can include:

- Patient/scan information
- Classification result
- Confidence score
- Tumor measurements
- Segmentation findings
- Explainability information
- Digital sign-off section

The report is designed to demonstrate how AI results could be transformed into a structured clinical-style output.

---

### 🚨 7. Emergency Triage Alert Demo

The dashboard includes an emergency triage workflow for demonstration purposes.

Users can trigger a simulated emergency notification based on the analysis.

> This feature is a **demo workflow only** and does not send real medical alerts or replace hospital communication systems.

---

### 🖼️ 8. External MRI Dataset Support

NeuroSight AI supports MRI images from external datasets, allowing the system to demonstrate its analysis pipeline beyond a single training dataset.

The project includes integration with the **BRISC2025 dataset structure** for classification and segmentation experimentation.

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     MRI Upload      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Preprocessing │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ CNN Classifier  │         │   U-Net Model   │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 ▼                           ▼
        ┌─────────────────┐         ┌─────────────────┐
        │ Tumor Detection │         │ Tumor Mask      │
        └────────┬────────┘         └────────┬────────┘
                 │                           │
                 │                           ▼
                 │                  ┌─────────────────┐
                 │                  │ Tumor Analysis  │
                 │                  └────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Grad-CAM        │
                    │ Explainability Map  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Streamlit Dashboard │
                    └──────────┬──────────┘
                               │
                  ┌────────────┼────────────┐
                  ▼            ▼            ▼
              Analytics      PDF       Triage Demo
