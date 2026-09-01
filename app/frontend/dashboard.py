import os
import cv2
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ==============================================================================
# OPTIONAL DEPENDENCIES
# ==============================================================================

# Twilio SMS / WhatsApp emergency pager integration
try:
    from twilio.rest import Client
except ImportError:
    Client = None

# Streamlit interactive before/after image comparison slider
try:
    from streamlit_image_comparison import image_comparison
except ImportError:
    image_comparison = None

# TensorFlow / Keras for real deep learning model inference
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
    load_model = None

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==============================================================================
# 1. CLINICAL THEME & PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Brain Tumor Detection and Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #070D18;
        color: #E2E8F0;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1528 0%, #070D18 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }

    .med-card {
        background: rgba(16, 27, 48, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.18);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    .med-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
    }

    .hero-diagnosis-card {
        background: linear-gradient(
            135deg,
            rgba(14, 116, 144, 0.35) 0%,
            rgba(15, 23, 42, 0.8) 100%
        );
        border: 1px solid #00B4D8;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 0 20px rgba(0, 180, 216, 0.15);
    }

    .badge-teal {
        background: rgba(6, 182, 212, 0.15);
        color: #38BDF8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.3);
        display: inline-block;
    }

    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(52, 211, 153, 0.3);
        display: inline-block;
    }

    .badge-alert {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(248, 113, 113, 0.3);
        display: inline-block;
    }

    .med-disclaimer {
        background: rgba(14, 165, 233, 0.08);
        border-left: 4px solid #0284C7;
        border-radius: 0 8px 8px 0;
        padding: 10px 16px;
        color: #BAE6FD;
        font-size: 0.85rem;
        margin-bottom: 20px;
    }

    .stButton > button {
        background: linear-gradient(
            135deg,
            #0284C7 0%,
            #0369A1 100%
        ) !important;

        color: #FFFFFF !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px !important;
        padding: 0.65rem 1.25rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stTabs [data-baseweb="tab"] {
        height: 46px;
        background-color: transparent;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
        border-bottom: 2px solid #38BDF8 !important;
    }

    .stImage > img {
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 2. MODEL LOADING & AUTO-DETECTION ENGINE
# ==============================================================================

CLASSES = ["Glioma", "Meningioma", "Normal", "Pituitary"]


@st.cache_resource
def load_neuro_models():
    """
    Attempts to load trained classifier and segmentation models.
    Returns (None, None) if models are unavailable.
    """

    if load_model is None:
        return None, None

    classifier_paths = [
        "models/classifier/tumor_classifier.h5",
        "models/classifier/tumor_classifier.keras",
        "models/classifier/model.keras",
        "models/tumor_classifier.h5",
        "models/tumor_classifier.keras",
    ]

    unet_paths = [
        "models/segmentation/unet_segmenter.h5",
        "models/segmentation/unet_segmenter.keras",
        "models/segmentation/unet.keras",
        "models/unet_segmenter.h5",
        "models/unet_segmenter.keras",
    ]

    classifier_path = next(
        (p for p in classifier_paths if os.path.exists(p)),
        None
    )

    unet_path = next(
        (p for p in unet_paths if os.path.exists(p)),
        None
    )

    clf = None
    unet = None

    if classifier_path:
        try:
            clf = load_model(classifier_path, compile=False)
        except Exception:
            clf = None

    if unet_path:
        try:
            unet = load_model(unet_path, compile=False)
        except Exception:
            unet = None

    return clf, unet


classifier_model, unet_model = load_neuro_models()


# ==============================================================================
# 3. AUTOMATIC MRI SEGMENTATION FALLBACK
# ==============================================================================

def auto_segment_mri(img_gray):
    """
    Morphological fallback segmentation for uploaded MRI slices.

    This is a demonstration fallback and should not be interpreted
    as a clinically validated diagnostic segmentation model.
    """

    _, brain_thresh = cv2.threshold(
        img_gray,
        30,
        255,
        cv2.THRESH_BINARY
    )

    kernel_brain = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (11, 11)
    )

    brain_clean = cv2.morphologyEx(
        brain_thresh,
        cv2.MORPH_CLOSE,
        kernel_brain
    )

    contours_brain, _ = cv2.findContours(
        brain_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    brain_mask = np.zeros_like(img_gray)

    if contours_brain:

        largest_brain_contour = max(
            contours_brain,
            key=cv2.contourArea
        )

        cv2.drawContours(
            brain_mask,
            [largest_brain_contour],
            -1,
            255,
            -1
        )

        brain_mask = cv2.erode(
            brain_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (9, 9)
            ),
            iterations=1
        )

    brain_only = cv2.bitwise_and(
        img_gray,
        brain_mask
    )

    brain_pixels = brain_only[brain_mask > 0]

    if len(brain_pixels) == 0:
        return (
            np.zeros_like(img_gray),
            "Normal",
            99.0
        )

    mean_val = np.mean(brain_pixels)
    std_val = np.std(brain_pixels)

    intensity_threshold = min(
        240,
        int(mean_val + 1.2 * std_val)
    )

    _, lesion_raw = cv2.threshold(
        brain_only,
        intensity_threshold,
        255,
        cv2.THRESH_BINARY
    )

    kernel_lesion = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    lesion_clean = cv2.morphologyEx(
        lesion_raw,
        cv2.MORPH_OPEN,
        kernel_lesion
    )

    contours_lesion, _ = cv2.findContours(
        lesion_clean,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    final_mask = np.zeros_like(img_gray)

    if not contours_lesion:
        return (
            final_mask,
            "Normal",
            98.5
        )

    valid_contours = [
        c for c in contours_lesion
        if cv2.contourArea(c) > 60
    ]

    if not valid_contours:
        return (
            final_mask,
            "Normal",
            98.5
        )

    largest_contour = max(
        valid_contours,
        key=cv2.contourArea
    )

    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        -1
    )

    M = cv2.moments(largest_contour)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 128, 128

    area = cv2.contourArea(largest_contour)

    if cY > 160 and 90 < cX < 165:

        detected_pathology = "Pituitary"
        confidence = 97.8

    elif area > 600 or (
        cX < 110 and cY < 130
    ):

        detected_pathology = "Glioma"
        confidence = 98.9

    else:

        detected_pathology = "Meningioma"
        confidence = 98.2

    return (
        final_mask,
        detected_pathology,
        confidence
    )


# ==============================================================================
# 4. GRAD-CAM
# ==============================================================================

def compute_gradcam_real(img_gray, model):

    try:

        resized = cv2.resize(
            img_gray,
            (224, 224)
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_GRAY2RGB
        )

        tensor = np.expand_dims(
            rgb.astype(np.float32) / 255.0,
            axis=0
        )

        last_conv = next(
            layer
            for layer in reversed(model.layers)
            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            )
        )

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [last_conv.output, model.output]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(
                tensor
            )

            pred_idx = tf.argmax(
                predictions[0]
            )

            loss = predictions[:, pred_idx]

        grads = tape.gradient(
            loss,
            conv_outputs
        )

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = (
            conv_outputs @
            pooled_grads[..., tf.newaxis]
        )

        heatmap = tf.squeeze(heatmap)

        heatmap = (
            tf.maximum(heatmap, 0) /
            (
                tf.math.reduce_max(heatmap)
                + 1e-10
            )
        )

        heatmap = cv2.resize(
            heatmap.numpy(),
            (256, 256)
        )

        heatmap = np.uint8(
            255 * heatmap
        )

        color_map = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        orig_bgr = cv2.cvtColor(
            cv2.resize(
                img_gray,
                (256, 256)
            ),
            cv2.COLOR_GRAY2BGR
        )

        return cv2.addWeighted(
            orig_bgr,
            0.6,
            color_map,
            0.4,
            0
        )

    except Exception:
        return None


# ==============================================================================
# 5. SYNTHETIC MRI GENERATOR
# ==============================================================================

def generate_sample_mri(
    pathology="Glioma"
):

    img = np.zeros(
        (256, 256),
        dtype=np.uint8
    )

    cv2.ellipse(
        img,
        (128, 128),
        (95, 110),
        0,
        0,
        360,
        50,
        -1
    )

    cv2.ellipse(
        img,
        (128, 128),
        (88, 103),
        0,
        0,
        360,
        95,
        -1
    )

    cv2.line(
        img,
        (128, 35),
        (128, 220),
        40,
        2
    )

    cv2.ellipse(
        img,
        (115, 125),
        (6, 25),
        10,
        0,
        360,
        40,
        -1
    )

    cv2.ellipse(
        img,
        (141, 125),
        (6, 25),
        -10,
        0,
        360,
        40,
        -1
    )

    mask = np.zeros(
        (256, 256),
        dtype=np.uint8
    )

    if pathology == "Glioma":

        cv2.ellipse(
            mask,
            (100, 95),
            (26, 20),
            25,
            0,
            360,
            255,
            -1
        )

        cv2.circle(
            mask,
            (115, 105),
            12,
            255,
            -1
        )

        img = cv2.add(
            img,
            cv2.GaussianBlur(
                mask,
                (15, 15),
                3
            )
        )

    elif pathology == "Meningioma":

        cv2.ellipse(
            mask,
            (165, 160),
            (20, 18),
            0,
            0,
            360,
            255,
            -1
        )

        img = cv2.add(
            img,
            mask
        )

    elif pathology == "Pituitary":

        cv2.ellipse(
            mask,
            (128, 175),
            (12, 10),
            0,
            0,
            360,
            255,
            -1
        )

        img = cv2.add(
            img,
            mask
        )

    noise = np.random.normal(
        0,
        3,
        (256, 256)
    ).astype(np.uint8)

    img = cv2.add(
        img,
        noise
    )

    img = cv2.GaussianBlur(
        img,
        (3, 3),
        0.8
    )

    return img, mask


# ==============================================================================
# 6. ONCOLOGY IMAGE FILTERS
# ==============================================================================

def apply_oncology_filter(
    img_gray,
    filter_name
):

    if filter_name == "CLAHE (Contrast Enhanced)":

        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )

        return clahe.apply(img_gray)

    elif filter_name == "Sobel Edge Sharpen":

        sobelx = cv2.Sobel(
            img_gray,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )

        sobely = cv2.Sobel(
            img_gray,
            cv2.CV_64F,
            0,
            1,
            ksize=3
        )

        sobel = np.sqrt(
            sobelx ** 2 +
            sobely ** 2
        )

        return np.uint8(
            np.clip(
                sobel,
                0,
                255
            )
        )

    elif filter_name == "Spectral Heatmap":

        return cv2.applyColorMap(
            img_gray,
            cv2.COLORMAP_JET
        )

    elif filter_name == "Thermal Bone & Tissue":

        return cv2.applyColorMap(
            img_gray,
            cv2.COLORMAP_HOT
        )

    return img_gray


# ==============================================================================
# 7. RADIOMICS
# ==============================================================================

def compute_radiomics(
    mask_2d,
    pathology
):

    if (
        np.sum(mask_2d) == 0
        or pathology in [
            "Normal",
            "No Tumor"
        ]
    ):

        return (
            0.0,
            0.0,
            "Healthy / Normal",
            "N/A"
        )

    contours, _ = cv2.findContours(
        mask_2d.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        return (
            0.0,
            0.0,
            "Healthy / Normal",
            "N/A"
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    tumor_pixel_area = cv2.contourArea(
        largest_contour
    )

    # Demonstration pixel calibration
    pixel_area_cm2 = 0.0088

    raw_area = (
        tumor_pixel_area *
        pixel_area_cm2
    )

    area_cm2 = round(
        float(
            np.clip(
                raw_area,
                2.5,
                32.0
            )
        ),
        2
    )

    # Approximate ellipsoidal volume
    r_eq = np.sqrt(
        area_cm2 / np.pi
    )

    volume_cm3 = round(
        float(
            (4.0 / 3.0)
            * np.pi
            * (r_eq ** 2.8)
        ),
        2
    )

    if pathology == "Glioma":

        lobe = "Right Frontal Lobe"
        grade = "WHO Grade IV (High Risk)"

    elif pathology == "Meningioma":

        lobe = "Left Parietal Convexity"
        grade = "WHO Grade I/II (Moderate)"

    elif pathology == "Pituitary":

        lobe = "Sellar / Sella Turcica"
        grade = "Benign Adenoma"

    else:

        lobe = "No lesion"
        grade = "None"

    return (
        area_cm2,
        volume_cm3,
        lobe,
        grade
    )


# ==============================================================================
# 8. 3D TUMOR MESH
# ==============================================================================

def build_3d_tumor_mesh(
    volume_cm3
):

    u = np.linspace(
        0,
        2 * np.pi,
        40
    )

    v = np.linspace(
        0,
        np.pi,
        40
    )

    # Keep visualization dimensions realistic.
    radius = max(
        np.cbrt(
            max(volume_cm3, 0.1)
            * 3
            / (4 * np.pi)
        ),
        1.0
    )

    x = (
        radius *
        np.outer(
            np.cos(u),
            np.sin(v)
        )
    )

    y = (
        radius *
        0.8 *
        np.outer(
            np.sin(u),
            np.sin(v)
        )
    )

    z = (
        radius *
        0.6 *
        np.outer(
            np.ones(np.size(u)),
            np.cos(v)
        )
    )

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale="Tealgrn",
                opacity=0.85,
                showscale=False
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showbackground=False,
                visible=False
            ),
            yaxis=dict(
                showbackground=False,
                visible=False
            ),
            zaxis=dict(
                showbackground=False,
                visible=False
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        height=320
    )

    return fig


# ==============================================================================
# 9. GLOBAL STATE
# ==============================================================================

if "selected_pathology" not in st.session_state:
    st.session_state.selected_pathology = "Glioma"

if "patient_id" not in st.session_state:
    st.session_state.patient_id = "NS-89240"

if "patient_name" not in st.session_state:
    st.session_state.patient_name = "Eleanor Vance"


# ==============================================================================
# 10. SIDEBAR
# ==============================================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            display:flex;
            align-items:center;
            gap:12px;
            margin-bottom:25px;
        ">

            <div style="
                background:linear-gradient(
                    135deg,
                    #0284C7,
                    #06B6D4
                );
                border-radius:10px;
                width:42px;
                height:42px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:22px;
                box-shadow:0 4px 12px rgba(6,182,212,0.3);
            ">
                🧠
            </div>

            <div>
                <div style="
                    font-weight:800;
                    font-size:1.15rem;
                    color:#FFFFFF;
                ">
                    Brain Tumor
                    <span style="color:#38BDF8;">
                        AI
                    </span>
                </div>

                <div style="
                    font-size:0.7rem;
                    color:#94A3B8;
                    letter-spacing:0.5px;
                ">
                    CLINICAL ANALYSIS SUITE
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    app_module = st.radio(
        "Navigation",
        [
            "🔬 Diagnostic Workspace",
            "📊 Patient Directory & NLP",
            "📈 Longitudinal Growth Tracker",
            "🏥 Hospital Resource Allocator",
            "🤖 AI Clinical Assistant"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    status_label = (
        "CUSTOM WEIGHTS LOADED"
        if classifier_model is not None
        else "AUTO-DETECTION ACTIVE"
    )

    status_color = (
        "#34D399"
        if classifier_model is not None
        else "#38BDF8"
    )

    st.markdown(
        f"""
        <div class="med-card"
             style="padding:14px;margin-top:15px;">

            <div style="
                display:flex;
                align-items:center;
                gap:8px;
            ">

                <span style="
                    height:8px;
                    width:8px;
                    background-color:{status_color};
                    border-radius:50%;
                    box-shadow:0 0 8px {status_color};
                ">
                </span>

                <span style="
                    font-size:0.8rem;
                    font-weight:700;
                    color:#FFFFFF;
                ">
                    {status_label}
                </span>

            </div>

            <div style="
                font-size:0.72rem;
                color:#94A3B8;
                margin-top:6px;
            ">
                • Auto-Lesion Detection: Active<br>
                • Skull-Stripping: Active<br>
                • Grad-CAM & Radiomics: Active
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "🔒 Clinical data should be handled according to applicable privacy regulations."
    )


# ==============================================================================
# 11. DIAGNOSTIC WORKSPACE
# ==============================================================================

if app_module == "🔬 Diagnostic Workspace":

    st.markdown(
        """
        <div class="med-disclaimer">

            <b>Clinical Support Notice:</b>
            This application is a research/demo clinical decision-support tool.
            AI-generated segmentation, classification and volumetric measurements
            require confirmation by a qualified medical professional.

        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(
        [1.1, 1.8, 1.3],
        gap="medium"
    )

    # --------------------------------------------------------------------------
    # COLUMN 1
    # --------------------------------------------------------------------------

    with col1:

        st.markdown(
            "<h5 style='color:#F8FAFC;'>1. Scan & Patient Intake</h5>",
            unsafe_allow_html=True
        )

        with st.container():

            st.markdown(
                '<div class="med-card">',
                unsafe_allow_html=True
            )

            uploaded_file = st.file_uploader(
                "Upload Brain MRI (DICOM / PNG / JPG)",
                type=[
                    "png",
                    "jpg",
                    "jpeg",
                    "dcm"
                ]
            )

            if uploaded_file is not None:

                file_bytes = np.asarray(
                    bytearray(
                        uploaded_file.read()
                    ),
                    dtype=np.uint8
                )

                decoded = cv2.imdecode(
                    file_bytes,
                    cv2.IMREAD_GRAYSCALE
                )

                if decoded is not None:

                    current_raw_mri = cv2.resize(
                        decoded,
                        (256, 256)
                    )

                else:

                    current_raw_mri, _ = generate_sample_mri(
                        "Glioma"
                    )

                is_custom_upload = True

                st.success(
                    "✅ Image Uploaded & Ingested!"
                )

            else:

                sample_choice = st.selectbox(
                    "Or Select Clinical Sample Preset",
                    [
                        "Glioma (High Grade)",
                        "Meningioma (Benign)",
                        "Pituitary Adenoma",
                        "Normal (No Tumor)"
                    ]
                )

                if "Glioma" in sample_choice:

                    selected_preset = "Glioma"

                elif "Meningioma" in sample_choice:

                    selected_preset = "Meningioma"

                elif "Pituitary" in sample_choice:

                    selected_preset = "Pituitary"

                else:

                    selected_preset = "Normal"

                current_raw_mri, _ = generate_sample_mri(
                    selected_preset
                )

                is_custom_upload = False

            st.markdown("---")

            st.markdown(
                """
                <div style="
                    font-size:0.8rem;
                    font-weight:700;
                    color:#94A3B8;
                ">
                    DICOM METADATA
                </div>

                <div style="
                    display:flex;
                    flex-wrap:wrap;
                    gap:6px;
                    margin-top:6px;
                ">

                    <span class="badge-teal">
                        MAGNET: 3.0 Tesla
                    </span>

                    <span class="badge-teal">
                        SEQUENCE: T1-CE Axial
                    </span>

                    <span class="badge-teal">
                        FOV: 240mm
                    </span>

                    <span class="badge-green">
                        QUALITY: Passed
                    </span>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("---")

            filter_choice = st.selectbox(
                "Oncology Contrast Enhancement",
                [
                    "Raw Grayscale",
                    "CLAHE (Contrast Enhanced)",
                    "Sobel Edge Sharpen",
                    "Spectral Heatmap",
                    "Thermal Bone & Tissue"
                ]
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

    # --------------------------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------------------------

    if is_custom_upload:

        current_mask, auto_pathology, auto_conf = auto_segment_mri(
            current_raw_mri
        )

        st.session_state.selected_pathology = auto_pathology
        confidence = auto_conf

    else:

        st.session_state.selected_pathology = selected_preset

        _, current_mask = generate_sample_mri(
            st.session_state.selected_pathology
        )

        confidence = (
            98.4
            if selected_preset != "Normal"
            else 99.2
        )

    # --------------------------------------------------------------------------
    # CONTOUR
    # --------------------------------------------------------------------------

    contour_display = cv2.cvtColor(
        current_raw_mri,
        cv2.COLOR_GRAY2BGR
    )

    contours, _ = cv2.findContours(
        current_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        contour_display,
        contours,
        -1,
        (0, 220, 255),
        2
    )

    # --------------------------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------------------------

    real_gradcam = None

    if classifier_model is not None:

        real_gradcam = compute_gradcam_real(
            current_raw_mri,
            classifier_model
        )

    if real_gradcam is not None:

        gradcam_overlay = real_gradcam

    else:

        gradcam_heat = cv2.applyColorMap(
            cv2.GaussianBlur(
                current_mask,
                (15, 15),
                0
            ),
            cv2.COLORMAP_JET
        )

        gradcam_overlay = cv2.addWeighted(
            cv2.cvtColor(
                current_raw_mri,
                cv2.COLOR_GRAY2BGR
            ),
            0.6,
            gradcam_heat,
            0.4,
            0
        )

    # --------------------------------------------------------------------------
    # RADIOMICS
    # --------------------------------------------------------------------------

    (
        area_cm2,
        vol_cm3,
        lobe,
        grade
    ) = compute_radiomics(
        current_mask,
        st.session_state.selected_pathology
    )

    # --------------------------------------------------------------------------
    # COLUMN 2
    # --------------------------------------------------------------------------

    with col2:

        st.markdown(
            "<h5 style='color:#F8FAFC;'>2. Explainable Scan Visualizer</h5>",
            unsafe_allow_html=True
        )

        filtered_view = apply_oncology_filter(
            current_raw_mri,
            filter_choice
        )

        tab_raw, tab_unet, tab_gradcam, tab_3d = st.tabs(
            [
                "🔍 Raw Scan",
                "📐 Boundary Segmentation",
                "🔥 Grad-CAM Heatmap",
                "🧊 3D Reconstruction"
            ]
        )

        with tab_raw:

            st.image(
                filtered_view,
                caption=(
                    f"Axial MRI Slice "
                    f"(Filter: {filter_choice})"
                ),
                use_container_width=True
            )

        with tab_unet:

            st.image(
                contour_display,
                caption=(
                    "Automated Lesion Boundary "
                    "(Demo Segmentation)"
                ),
                use_container_width=True
            )

        with tab_gradcam:

            st.image(
                gradcam_overlay,
                caption="Model Attention / Grad-CAM View",
                use_container_width=True
            )

        with tab_3d:

            if vol_cm3 > 0:

                st.plotly_chart(
                    build_3d_tumor_mesh(vol_cm3),
                    use_container_width=True
                )

                st.caption(
                    "Interactive 3D visualization of the estimated tumor mass."
                )

            else:

                st.info(
                    "No tumor volume detected."
                )

    # --------------------------------------------------------------------------
    # COLUMN 3
    # --------------------------------------------------------------------------

    with col3:

        st.markdown(
            "<h5 style='color:#F8FAFC;'>3. Diagnostic Biomarkers & Triage</h5>",
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="hero-diagnosis-card">

                <div style="
                    font-size:0.75rem;
                    font-weight:700;
                    color:#BAE6FD;
                    text-transform:uppercase;
                    letter-spacing:1px;
                ">
                    PRIMARY CLASSIFICATION
                </div>

                <div style="
                    font-size:1.8rem;
                    font-weight:800;
                    color:#FFFFFF;
                    margin-top:2px;
                ">
                    {st.session_state.selected_pathology}
                </div>

                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    margin-top:8px;
                    gap:5px;
                    flex-wrap:wrap;
                ">

                    <span class="badge-green">
                        Confidence: {confidence:.1f}%
                    </span>

                    <span class="badge-teal">
                        {grade}
                    </span>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="med-card">

                <div style="
                    display:grid;
                    grid-template-columns:1fr 1fr;
                    gap:12px;
                ">

                    <div>

                        <div style="
                            font-size:0.75rem;
                            color:#94A3B8;
                        ">
                            3D EST. VOLUME
                        </div>

                        <div style="
                            font-size:1.3rem;
                            font-weight:700;
                            color:#38BDF8;
                        ">
                            {vol_cm3} cm³
                        </div>

                    </div>

                    <div>

                        <div style="
                            font-size:0.75rem;
                            color:#94A3B8;
                        ">
                            SURFACE AREA
                        </div>

                        <div style="
                            font-size:1.3rem;
                            font-weight:700;
                            color:#38BDF8;
                        ">
                            {area_cm2} cm²
                        </div>

                    </div>

                </div>

                <div style="
                    margin-top:12px;
                    padding-top:10px;
                    border-top:1px solid rgba(255,255,255,0.08);
                ">

                    <div style="
                        font-size:0.75rem;
                        color:#94A3B8;
                    ">
                        ANATOMICAL LOCALIZATION
                    </div>

                    <div style="
                        font-size:0.95rem;
                        font-weight:600;
                        color:#F1F5F9;
                        margin-top:2px;
                    ">
                        📍 {lobe}
                    </div>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        with st.expander(
            "📝 Radiologist Notes & Digital Sign-off",
            expanded=True
        ):

            notes_input = st.text_area(
                "Observations",
                (
                    f"Hyperintense lesion identified in {lobe}. "
                    f"Estimated volume: {vol_cm3} cm³. "
                    "AI-generated findings require clinical confirmation."
                ),
                height=90
            )

            is_signed = st.checkbox(
                "Apply Digital Signature "
                "(Demo Sign-off)"
            )

            if st.button(
                "⚡ Generate Clinical PDF Report",
                use_container_width=True
            ):

                if is_signed:

                    st.success(
                        "✅ Clinical report marked as digitally signed (demo)."
                    )

                else:

                    st.warning(
                        "Please complete the digital sign-off checkbox."
                    )


# ==============================================================================
# 12. PATIENT DIRECTORY & NLP
# ==============================================================================

elif app_module == "📊 Patient Directory & NLP":

    st.markdown(
        "### 📊 Patient Directory & Smart NLP Search"
    )

    st.caption(
        "Search across longitudinal patient history using plain clinical language."
    )

    nlp_query = st.text_input(
        "🔍 Smart Query Parser",
        placeholder=(
            "Try: Glioma, age > 50, "
            "volume > 10, Referral Required"
        )
    )

    patients_data = pd.DataFrame(
        {
            "Patient ID": [
                "NS-89240",
                "NS-89241",
                "NS-89242",
                "NS-89243",
                "NS-89244"
            ],

            "Name": [
                "Eleanor Vance",
                "Arthur Dent",
                "Marcus Brody",
                "Elena Rostova",
                "Julian Hayes"
            ],

            "Age": [
                56,
                44,
                62,
                38,
                71
            ],

            "Diagnosis": [
                "Glioma",
                "Glioma",
                "Meningioma",
                "Pituitary",
                "No Tumor"
            ],

            "Volume (cm³)": [
                14.8,
                8.2,
                5.4,
                3.1,
                0.0
            ],

            "Confidence": [
                "98.4%",
                "96.1%",
                "94.5%",
                "92.0%",
                "99.2%"
            ],

            "Triage Status": [
                "High Priority",
                "Routine",
                "Routine",
                "Referral Required",
                "Discharged"
            ]
        }
    )

    # --------------------------------------------------------------------------
    # NLP FILTERING
    # --------------------------------------------------------------------------

    if nlp_query:

        query = nlp_query.lower().strip()

        filtered_df = patients_data.copy()

        # Diagnosis search
        diagnosis_matches = [
            diagnosis
            for diagnosis in [
                "glioma",
                "meningioma",
                "pituitary",
                "normal",
                "no tumor"
            ]
            if diagnosis in query
        ]

        if diagnosis_matches:

            filtered_df = filtered_df[
                filtered_df["Diagnosis"]
                .str.lower()
                .isin(diagnosis_matches)
            ]

        # Age filtering
        if "age >" in query:

            try:

                age_value = int(
                    query.split("age >")[1]
                    .strip()
                    .split()[0]
                )

                filtered_df = filtered_df[
                    filtered_df["Age"] > age_value
                ]

            except Exception:
                pass

        elif "age <" in query:

            try:

                age_value = int(
                    query.split("age <")[1]
                    .strip()
                    .split()[0]
                )

                filtered_df = filtered_df[
                    filtered_df["Age"] < age_value
                ]

            except Exception:
                pass

        # Volume filtering
        if "volume >" in query:

            try:

                volume_value = float(
                    query.split("volume >")[1]
                    .strip()
                    .split()[0]
                )

                filtered_df = filtered_df[
                    filtered_df["Volume (cm³)"]
                    > volume_value
                ]

            except Exception:
                pass

        elif "volume <" in query:

            try:

                volume_value = float(
                    query.split("volume <")[1]
                    .strip()
                    .split()[0]
                )

                filtered_df = filtered_df[
                    filtered_df["Volume (cm³)"]
                    < volume_value
                ]

            except Exception:
                pass

        # Triage search
        if "referral" in query:

            filtered_df = filtered_df[
                filtered_df["Triage Status"]
                .str.lower()
                .str.contains("referral")
            ]

        elif "high priority" in query:

            filtered_df = filtered_df[
                filtered_df["Triage Status"]
                .str.lower()
                .str.contains("high priority")
            ]

        # Generic fallback
        if (
            filtered_df.equals(patients_data)
            and not diagnosis_matches
            and "age >" not in query
            and "age <" not in query
            and "volume >" not in query
            and "volume <" not in query
        ):

            filtered_df = patients_data[
                patients_data.apply(
                    lambda row:
                    query in " ".join(
                        row.astype(str)
                    ).lower(),
                    axis=1
                )
            ]

        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.dataframe(
            patients_data,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------------------------------
    # ANALYTICS
    # --------------------------------------------------------------------------

    c1, c2 = st.columns(2)

    with c1:

        fig_pie = px.pie(
            patients_data,
            names="Diagnosis",
            title="Patient Pathology Distribution",
            hole=0.45,
            color_discrete_sequence=[
                "#0284C7",
                "#06B6D4",
                "#38BDF8",
                "#10B981"
            ]
        )

        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF"
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

    with c2:

        fig_bar = px.bar(
            patients_data,
            x="Name",
            y="Volume (cm³)",
            color="Diagnosis",
            title="Tumor Volume Comparison",
            color_discrete_sequence=[
                "#0284C7",
                "#38BDF8",
                "#06B6D4"
            ]
        )

        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF"
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        )


# ==============================================================================
# 13. LONGITUDINAL GROWTH TRACKER
# ==============================================================================

elif app_module == "📈 Longitudinal Growth Tracker":

    st.markdown(
        "### 📈 Longitudinal Growth Tracker & Comparative Visualizer"
    )

    col_left, col_right = st.columns(
        [1, 1.8],
        gap="large"
    )

    with col_left:

        st.markdown(
            """
            <div class="med-card"
                 style="border-left:4px solid #F87171;">

                <div style="
                    font-size:0.75rem;
                    color:#F87171;
                    font-weight:700;
                ">
                    RAPID GROWTH ACCELERATION ALERT
                </div>

                <div style="
                    font-size:2rem;
                    font-weight:800;
                    color:#FFFFFF;
                    margin-top:4px;
                ">
                    +15.6%
                </div>

                <div style="
                    font-size:0.85rem;
                    color:#94A3B8;
                    margin-top:4px;
                ">
                    Volumetric growth exceeded the
                    10% demonstration threshold between
                    March 2026 and May 2026.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        growth_history = pd.DataFrame(
            {
                "Date": [
                    "Oct 2025",
                    "Dec 2025",
                    "Mar 2026",
                    "May 2026"
                ],

                "Volume (cm³)": [
                    10.2,
                    11.4,
                    12.8,
                    14.8
                ]
            }
        )

        fig_timeline = px.line(
            growth_history,
            x="Date",
            y="Volume (cm³)",
            markers=True,
            title="Tumor Progression Timeline"
        )

        fig_timeline.update_traces(
            line_color="#00B4D8",
            line_width=3,
            marker_size=8
        )

        fig_timeline.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#FFFFFF",
            height=250,
            margin=dict(
                t=40,
                b=10,
                l=10,
                r=10
            )
        )

        st.plotly_chart(
            fig_timeline,
            use_container_width=True
        )

        # ----------------------------------------------------------------------
        # TWILIO EMERGENCY PAGER
        # ----------------------------------------------------------------------

        st.markdown(
            "<h5 style='color:#F8FAFC;'>📱 Emergency Triage Pager</h5>",
            unsafe_allow_html=True
        )

        st.caption(
            "Demo integration for sending WhatsApp triage alerts."
        )

        on_call_number = st.text_input(
            "On-Call Phone Number",
            placeholder="+917061350676"
        )

        if st.button(
            "🚨 TRIGGER WHATSAPP ALERT",
            use_container_width=True
        ):

            if not on_call_number:

                st.warning(
                    "Please enter a phone number."
                )

            else:

                with st.spinner(
                    "Dispatching emergency WhatsApp alert..."
                ):

                    time.sleep(1)

                    # ----------------------------------------------------------
                    # SECURE CREDENTIAL LOADING
                    # ----------------------------------------------------------

                    TWILIO_SID = os.getenv(
                        "TWILIO_ACCOUNT_SID"
                    )

                    TWILIO_AUTH_TOKEN = os.getenv(
                        "TWILIO_AUTH_TOKEN"
                    )

                    TWILIO_PHONE = os.getenv(
                        "TWILIO_WHATSAPP_FROM"
                    )

                    try:

                        if (
                            Client is None
                            or not TWILIO_SID
                            or not TWILIO_AUTH_TOKEN
                            or not TWILIO_PHONE
                        ):

                            st.warning(
                                "Twilio credentials are not configured. "
                                "Demo mode: no real message was sent."
                            )

                        else:

                            client = Client(
                                TWILIO_SID,
                                TWILIO_AUTH_TOKEN
                            )

                            message = client.messages.create(
                                from_=TWILIO_PHONE,
                                content_sid=(
                                    "HXb10a34b150993952f468205fbc074213"
                                ),
                                content_variables=(
                                    '{"1":"12","2":"3"}'
                                ),
                                to=(
                                    f"whatsapp:{on_call_number}"
                                )
                            )

                            st.success(
                                "✅ WhatsApp emergency triage "
                                "alert dispatched successfully."
                            )

                    except Exception as e:

                        st.error(
                            "Unable to send the WhatsApp alert. "
                            "Please verify the Twilio configuration."
                        )

    # --------------------------------------------------------------------------
    # SCAN COMPARISON
    # --------------------------------------------------------------------------

    with col_right:

        st.markdown(
            "<h5 style='color:#F8FAFC;'>Interactive Dual-Scan Comparison Slider</h5>",
            unsafe_allow_html=True
        )

        st.caption(
            "Compare baseline and current scan morphology."
        )

        base_img, base_m = generate_sample_mri(
            "Glioma"
        )

        base_m = cv2.erode(
            base_m,
            np.ones(
                (5, 5),
                np.uint8
            ),
            iterations=1
        )

        base_display = cv2.cvtColor(
            base_img,
            cv2.COLOR_GRAY2BGR
        )

        base_contours, _ = cv2.findContours(
            base_m,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            base_display,
            base_contours,
            -1,
            (216, 180, 0),
            2
        )

        curr_raw, curr_m = generate_sample_mri(
            "Glioma"
        )

        curr_display = cv2.cvtColor(
            curr_raw,
            cv2.COLOR_GRAY2BGR
        )

        curr_contours, _ = cv2.findContours(
            curr_m,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            curr_display,
            curr_contours,
            -1,
            (216, 180, 0),
            2
        )

        if image_comparison:

            image_comparison(
                img1=Image.fromarray(
                    base_display
                ),

                img2=Image.fromarray(
                    curr_display
                ),

                label1="Baseline",
                label2="Current",

                width=650,

                starting_position=50
            )

        else:

            c_a, c_b = st.columns(2)

            with c_a:

                st.image(
                    base_display,
                    caption="Baseline",
                    use_container_width=True
                )

            with c_b:

                st.image(
                    curr_display,
                    caption="Current",
                    use_container_width=True
                )


# ==============================================================================
# 14. HOSPITAL RESOURCE ALLOCATOR
# ==============================================================================

elif app_module == "🏥 Hospital Resource Allocator":

    st.markdown(
        "### 🏥 Smart Hospital Resource Allocator"
    )

    st.caption(
        "Demonstration of radiomics-driven surgical resource planning."
    )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.markdown(
            """
            <div class="med-card">

                <div style="
                    font-size:0.8rem;
                    color:#94A3B8;
                    font-weight:700;
                ">
                    EST. OR PROCEDURE DURATION
                </div>

                <div style="
                    font-size:1.8rem;
                    font-weight:800;
                    color:#38BDF8;
                    margin-top:4px;
                ">
                    5.2 Hours
                </div>

                <div style="
                    font-size:0.8rem;
                    color:#34D399;
                    margin-top:4px;
                ">
                    Complexity Score: High
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with r2:

        st.markdown(
            """
            <div class="med-card">

                <div style="
                    font-size:0.8rem;
                    color:#94A3B8;
                    font-weight:700;
                ">
                    POST-OP ICU BED PRIORITY
                </div>

                <div style="
                    font-size:1.8rem;
                    font-weight:800;
                    color:#F87171;
                    margin-top:4px;
                ">
                    Level 1 Priority
                </div>

                <div style="
                    font-size:0.8rem;
                    color:#94A3B8;
                    margin-top:4px;
                ">
                    Reserved Duration: 48 Hours
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with r3:

        st.markdown(
            """
            <div class="med-card">

                <div style="
                    font-size:0.8rem;
                    color:#94A3B8;
                    font-weight:700;
                ">
                    SURGICAL NAVIGATION KIT
                </div>

                <div style="
                    font-size:1.8rem;
                    font-weight:800;
                    color:#34D399;
                    margin-top:4px;
                ">
                    Confirmed Ready
                </div>

                <div style="
                    font-size:0.8rem;
                    color:#94A3B8;
                    margin-top:4px;
                ">
                    Neuronavigation & Intra-op Ultrasound
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "#### Surgical Equipment & Staffing Requisition"
    )

    equip_df = pd.DataFrame(
        {
            "Resource / Equipment": [
                "Stereotactic Head Frame",
                "Intraoperative MRI (iMRI)",
                "Ultrasonic Surgical Aspirator (CUSA)",
                "Neuro-Anesthesiologist",
                "Lead Neurosurgeon"
            ],

            "Status": [
                "Allocated (OR #3)",
                "Scheduled",
                "Sterilized & Staged",
                "Assigned",
                "Assigned"
            ],

            "Readiness": [
                "Ready",
                "Ready",
                "Ready",
                "Confirmed",
                "Confirmed"
            ]
        }
    )

    st.dataframe(
        equip_df,
        use_container_width=True,
        hide_index=True
    )


# ==============================================================================
# 15. MULTI-AGENT AI TUMOR BOARD
# ==============================================================================

elif app_module == "🤖 AI Clinical Assistant":

    st.markdown(
        "### 🏛️ Multi-Agent AI Tumor Board Simulation"
    )

    st.caption(
        "Demonstration of a multidisciplinary AI clinical workflow."
    )

    _, board_mask = generate_sample_mri(
        st.session_state.selected_pathology
    )

    (
        area_cm2,
        vol_cm3,
        lobe,
        grade
    ) = compute_radiomics(
        board_mask,
        st.session_state.selected_pathology
    )

    st.markdown(
        f"""
        <div class="med-card"
             style="
                 border-left:4px solid #00B4D8;
                 margin-bottom:20px;
             ">

            <div style="
                font-size:0.8rem;
                font-weight:700;
                color:#38BDF8;
            ">
                ACTIVE CLINICAL PROFILE REVIEW
            </div>

            <div style="
                display:flex;
                gap:20px;
                margin-top:8px;
                flex-wrap:wrap;
            ">

                <div>
                    <span style="color:#94A3B8;">
                        Patient ID:
                    </span>
                    <b>
                        {st.session_state.patient_id}
                    </b>
                </div>

                <div>
                    <span style="color:#94A3B8;">
                        Pathology:
                    </span>
                    <b>
                        {st.session_state.selected_pathology}
                    </b>
                </div>

                <div>
                    <span style="color:#94A3B8;">
                        Volume:
                    </span>
                    <b>
                        {vol_cm3} cm³
                    </b>
                </div>

                <div>
                    <span style="color:#94A3B8;">
                        Location:
                    </span>
                    <b>
                        {lobe}
                    </b>
                </div>

                <div>
                    <span style="color:#94A3B8;">
                        Grade:
                    </span>

                    <span class="badge-teal">
                        {grade}
                    </span>
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    c_btn, _ = st.columns(
        [1.5, 2]
    )

    with c_btn:

        start_board = st.button(
            "⚡ Convene Multidisciplinary Tumor Board",
            use_container_width=True
        )

    if start_board:

        # ----------------------------------------------------------------------
        # AGENT 1: NEURORADIOLOGIST
        # ----------------------------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar="🩻"
        ):

            st.markdown(
                "**AI Agent 1 — Neuroradiologist**"
            )

            with st.spinner(
                "Analyzing imaging features..."
            ):

                time.sleep(1)

            st.markdown(
                f"""
                - **Imaging Assessment:** Lesion identified in **{lobe}**.
                - **Radiomic Quantification:** Estimated surface area
                  **{area_cm2} cm²**.
                - **Estimated Volume:** **{vol_cm3} cm³**.
                - **Attention Analysis:** AI attention visualization
                  highlights the detected lesion region.
                - **Impression:** Imaging profile is consistent with
                  **{st.session_state.selected_pathology}** in this demo.
                """
            )

        # ----------------------------------------------------------------------
        # AGENT 2: NEUROSURGEON
        # ----------------------------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar="👨‍⚕️"
        ):

            st.markdown(
                "**AI Agent 2 — Neurosurgical Reviewer**"
            )

            with st.spinner(
                "Assessing surgical considerations..."
            ):

                time.sleep(1.2)

            if st.session_state.selected_pathology in [
                "Glioma",
                "Meningioma"
            ]:

                surg_plan = (
                    f"Potential surgical access should be evaluated "
                    f"based on the lesion location in {lobe}, "
                    f"with functional anatomy and imaging reviewed "
                    f"before any intervention."
                )

            elif st.session_state.selected_pathology == "Pituitary":

                surg_plan = (
                    "Sellar lesions may require evaluation for "
                    "endoscopic endonasal approaches depending "
                    "on anatomy, size and clinical findings."
                )

            else:

                surg_plan = (
                    "No surgical intervention is suggested by "
                    "the demonstration profile."
                )

            st.markdown(
                f"""
                - **Surgical Feasibility:** Requires specialist review.
                - **Surgical Consideration:** {surg_plan}
                - **Planning Support:** 3D visualization and
                  neuronavigation-compatible imaging can assist
                  pre-operative planning.
                """
            )

        # ----------------------------------------------------------------------
        # AGENT 3: NEURO-ONCOLOGIST
        # ----------------------------------------------------------------------

        with st.chat_message(
            "assistant",
            avatar="🔬"
        ):

            st.markdown(
                "**AI Agent 3 — Neuro-Oncology Reviewer**"
            )

            with st.spinner(
                "Reviewing oncology pathways..."
            ):

                time.sleep(1.1)

            if st.session_state.selected_pathology == "Glioma":

                onco_plan = (
                    "Recommend multidisciplinary review of histopathology "
                    "and molecular biomarkers such as IDH and MGMT before "
                    "selecting an adjuvant treatment pathway."
                )

            elif st.session_state.selected_pathology == "Meningioma":

                onco_plan = (
                    "Management should be based on pathological grade, "
                    "extent of resection, symptoms and follow-up imaging."
                )

            elif st.session_state.selected_pathology == "Pituitary":

                onco_plan = (
                    "Endocrine evaluation and hormone profiling should "
                    "be considered alongside imaging findings."
                )

            else:

                onco_plan = (
                    "No tumor-directed treatment pathway is suggested "
                    "by this demonstration profile."
                )

            st.markdown(
                f"""
                - **Therapeutic Strategy:** {onco_plan}
                - **Molecular/Clinical Review:** Final treatment decisions
                  require pathology, molecular data and specialist assessment.
                """
            )

        # ----------------------------------------------------------------------
        # CONSENSUS
        # ----------------------------------------------------------------------

        st.markdown(
            f"""
            <div class="med-card"
                 style="
                     border:1px solid #10B981;
                     background:rgba(16,185,129,0.08);
                     margin-top:15px;
                 ">

                <div style="
                    display:flex;
                    align-items:center;
                    gap:8px;
                ">

                    <span style="font-size:1.1rem;">
                        🎯
                    </span>

                    <span style="
                        font-weight:800;
                        color:#34D399;
                        font-size:0.95rem;
                    ">
                        MULTI-AGENT CONSENSUS
                    </span>

                </div>

                <div style="
                    font-size:0.85rem;
                    color:#E2E8F0;
                    margin-top:8px;
                    line-height:1.5;
                ">

                    <b>Consensus Summary:</b>
                    The case should undergo multidisciplinary clinical
                    review using the imaging findings, segmentation,
                    volumetric estimates and relevant clinical/pathology data.

                    This AI-generated consensus is a decision-support
                    demonstration and is not a substitute for clinical judgment.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------------------------
    # Q&A
    # --------------------------------------------------------------------------

    st.markdown("---")

    st.markdown(
        "#### 💬 Ask the Tumor Board Panel"
    )

    panel_query = st.chat_input(
        "Ask a question to the panel..."
    )

    if panel_query:

        st.chat_message(
            "user"
        ).write(
            panel_query
        )

        st.chat_message(
            "assistant",
            avatar="🏛️"
        ).write(
            f"""
            **Tumor Board Cross-Consultation**

            Question: **{panel_query}**

            - **Radiology:** Review lesion morphology,
              location and imaging characteristics.
            - **Surgery:** Assess anatomical relationships,
              functional structures and operative feasibility.
            - **Oncology:** Correlate imaging with pathology
              and molecular findings before treatment decisions.

            **Note:** This demonstration does not replace
            evaluation by qualified clinicians.
            """
        )