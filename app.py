import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import cv2

# ── Must be first Streamlit call ──────────────────────────────
st.set_page_config(
    page_title="🦋 Butterfly Classifier",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #1a1d2e; }
    .metric-card {
        background: linear-gradient(135deg, #1e2140, #2a2d4a);
        border: 1px solid #3d4270;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: #7eb8f7; }
    .metric-label { font-size: 0.85rem; color: #9aa3c2; margin-top: 4px; }
    .model-card {
        background: linear-gradient(135deg, #1a2436, #1e2d45);
        border: 1px solid #2e4a6e;
        border-radius: 14px;
        padding: 22px;
        margin: 8px 0;
    }
    .best-card {
        background: linear-gradient(135deg, #1a3028, #1e4038);
        border: 2px solid #2ecc71;
        border-radius: 14px;
        padding: 22px;
        margin: 8px 0;
    }
    .result-box {
        background: linear-gradient(135deg, #1a2436, #1e2d45);
        border: 1px solid #3d6ea8;
        border-radius: 14px;
        padding: 24px;
        text-align: center;
    }
    .species-name { font-size: 1.8rem; font-weight: 800; color: #7eb8f7; letter-spacing: 1px; }
    .info-box {
        background: #1e2140;
        border-left: 4px solid #7eb8f7;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #c8d8f0;
        border-bottom: 2px solid #2e4a6e;
        padding-bottom: 8px;
        margin: 24px 0 16px 0;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)


# ── Load Model & Class Map ────────────────────────────────────
@st.cache_resource(show_spinner="Loading model... please wait ⏳")
def load_resources():
    from tensorflow.keras.models import load_model
    model_path = os.path.join('saved_models', 'resnet50v2_butterfly_best.keras')
    json_path  = os.path.join('saved_models', 'class_indices.json')

    if not os.path.exists(model_path):
        return None, None, f"Model file not found at: {model_path}"
    if not os.path.exists(json_path):
        return None, None, f"class_indices.json not found at: {json_path}"

    model = load_model(model_path)
    with open(json_path) as f:
        idx_to_class = json.load(f)
    return model, idx_to_class, None

model, idx_to_class, load_error = load_resources()


# ── Preprocess image ──────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)


# ── Sidebar Navigation ────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦋 Butterfly Classifier")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔍 Predict Species", "📊 Model Metrics", "🦋 About Butterflies"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.8rem; color:#9aa3c2; line-height:1.7'>
    <b>Best Model:</b> ResNet50V2<br>
    <b>Test Accuracy:</b> 94.00%<br>
    <b>Species:</b> 100 classes<br>
    <b>Framework:</b> TensorFlow 2.19
    </div>
    """, unsafe_allow_html=True)

    if load_error:
        st.error(f"⚠️ {load_error}")
    else:
        st.success("✅ Model loaded")


# ════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ════════════════════════════════════════════════════════════
if page == "🔍 Predict Species":
    st.markdown("# 🔍 Identify a Butterfly or Moth")
    st.markdown("Upload any butterfly or moth image — the model will classify it into one of **100 species**.")
    st.markdown("---")

    if load_error:
        st.error(f"Cannot run predictions: {load_error}")
    else:
        uploaded = st.file_uploader(
            "Drop your image here",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )

        if uploaded:
            image = Image.open(uploaded)
            col1, col2 = st.columns([1, 1.2], gap="large")

            with col1:
                st.markdown('<div class="section-header">📷 Your Image</div>', unsafe_allow_html=True)
                st.image(image, use_column_width=True)

            with col2:
                st.markdown('<div class="section-header">🧠 Prediction</div>', unsafe_allow_html=True)

                with st.spinner("Analysing image..."):
                    arr   = preprocess(image)
                    preds = model.predict(arr, verbose=0)[0]

                top5_idx     = np.argsort(preds)[::-1][:5]
                top5_species = [idx_to_class[str(i)] for i in top5_idx]
                top5_confs   = [float(preds[i]) * 100 for i in top5_idx]

                best_name = top5_species[0]
                best_conf = top5_confs[0]

                st.markdown(f"""
                <div class="result-box">
                    <div style="font-size:0.9rem; color:#9aa3c2; margin-bottom:6px;">IDENTIFIED SPECIES</div>
                    <div class="species-name">{best_name}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                if best_conf >= 80:
                    st.success(f"✅ High Confidence: **{best_conf:.2f}%**")
                elif best_conf >= 50:
                    st.warning(f"⚠️ Moderate Confidence: **{best_conf:.2f}%**")
                else:
                    st.error(f"❌ Low Confidence: **{best_conf:.2f}%** — try a clearer image")

            # Top 5 chart
            st.markdown("---")
            st.markdown('<div class="section-header">📊 Top 5 Predictions</div>', unsafe_allow_html=True)
            col_chart, col_table = st.columns([1.5, 1], gap="large")

            with col_chart:
                colors = ['#2ecc71'] + ['#4a90d9'] * 4
                fig, ax = plt.subplots(figsize=(9, 4))
                fig.patch.set_facecolor('#1a1d2e')
                ax.set_facecolor('#1a1d2e')
                bars = ax.barh(top5_species[::-1], top5_confs[::-1],
                               color=colors[::-1], alpha=0.9, height=0.6, edgecolor='none')
                ax.bar_label(bars, fmt='%.1f%%', padding=6,
                             fontsize=10, color='white', fontweight='bold')
                ax.set_xlabel('Confidence (%)', color='#9aa3c2', fontsize=10)
                ax.tick_params(colors='#c8d8f0', labelsize=9)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#3d4270')
                ax.spines['left'].set_color('#3d4270')
                ax.set_xlim(0, max(top5_confs) * 1.25)
                ax.grid(axis='x', color='#2a2d4a', linewidth=0.7)
                green = mpatches.Patch(color='#2ecc71', label='Top Prediction')
                blue  = mpatches.Patch(color='#4a90d9', label='Alternatives')
                ax.legend(handles=[green, blue], fontsize=8,
                          facecolor='#1a1d2e', labelcolor='white', edgecolor='#3d4270')
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            with col_table:
                df = pd.DataFrame({
                    'Rank'      : [f"#{i+1}" for i in range(5)],
                    'Species'   : top5_species,
                    'Confidence': [f"{c:.2f}%" for c in top5_confs]
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

        else:
            st.markdown("""
            <div class="info-box">
            📌 <b>How to use:</b><br>
            1. Click <b>Browse files</b> or drag and drop an image above<br>
            2. The model will automatically identify the species<br>
            3. You will see the top 5 predictions with confidence scores
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
            🦋 <b>Supported species include:</b> Monarch, Blue Morpho, Atlas Moth,
            Luna Moth, Swallowtail, Viceroy, Clearwing Moth, Adonis Blue, and 92 more!
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 2 — MODEL METRICS
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Metrics":
    st.markdown("# 📊 Model Performance Metrics")
    st.markdown("Results from training two CNN transfer learning models on 100 butterfly/moth species.")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">94.00%</div>
            <div class="metric-label">Best Test Accuracy<br>(ResNet50V2)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">100</div>
            <div class="metric-label">Species Classified</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">12,594</div>
            <div class="metric-label">Training Images</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class="metric-card">
            <div class="metric-value">2</div>
            <div class="metric-label">Models Trained</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🏆 Model Comparison</div>', unsafe_allow_html=True)
    col_x, col_r = st.columns(2, gap="large")

    with col_x:
        st.markdown("""<div class="model-card">
            <h3 style="color:#7eb8f7; margin:0 0 12px 0">Xception</h3>
            <table style="width:100%; font-size:0.88rem; color:#c8d8f0">
                <tr><td style="color:#9aa3c2">Val Accuracy</td><td><b>90.00%</b></td></tr>
                <tr><td style="color:#9aa3c2">Test Accuracy</td><td><b>90.40%</b></td></tr>
                <tr><td style="color:#9aa3c2">Best Epoch</td><td>14 / 15</td></tr>
                <tr><td style="color:#9aa3c2">LR Reduced At</td><td>Epoch 7</td></tr>
                <tr><td style="color:#9aa3c2">Trainable Params</td><td>2,200,676</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with col_r:
        st.markdown("""<div class="best-card">
            <h3 style="color:#2ecc71; margin:0 0 4px 0">ResNet50V2 &nbsp;🏆 Best</h3>
            <table style="width:100%; font-size:0.88rem; color:#c8d8f0">
                <tr><td style="color:#9aa3c2">Val Accuracy</td><td><b>91.80%</b></td></tr>
                <tr><td style="color:#9aa3c2">Test Accuracy</td><td><b style="color:#2ecc71">94.00%</b></td></tr>
                <tr><td style="color:#9aa3c2">Best Epoch</td><td>13 / 15</td></tr>
                <tr><td style="color:#9aa3c2">LR Reduced At</td><td>Epoch 11</td></tr>
                <tr><td style="color:#9aa3c2">Trainable Params</td><td>2,200,676</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📈 Accuracy Comparison Chart</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1a1d2e')
    ax.set_facecolor('#1a1d2e')
    x     = np.arange(2)
    width = 0.32
    b1 = ax.bar(x - width/2, [90.00, 91.80], width, label='Val Accuracy',
                color='#4a90d9', alpha=0.85, edgecolor='none')
    b2 = ax.bar(x + width/2, [90.40, 94.00], width, label='Test Accuracy',
                color='#2ecc71', alpha=0.85, edgecolor='none')
    ax.bar_label(b1, fmt='%.1f%%', padding=4, fontsize=10, color='white', fontweight='bold')
    ax.bar_label(b2, fmt='%.1f%%', padding=4, fontsize=10, color='white', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Xception', 'ResNet50V2'], color='#c8d8f0', fontsize=11)
    ax.set_ylim(85, 98)
    ax.set_ylabel('Accuracy (%)', color='#9aa3c2')
    ax.tick_params(colors='#9aa3c2')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('#3d4270')
    ax.spines['left'].set_color('#3d4270')
    ax.grid(axis='y', color='#2a2d4a', linewidth=0.7)
    ax.legend(facecolor='#1a1d2e', labelcolor='white', edgecolor='#3d4270', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">📋 Epoch-by-Epoch Training Log</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["  Xception  ", "  ResNet50V2  "])

    xception_log = pd.DataFrame({
        'Epoch'      : list(range(1, 16)),
        'Train Acc %': [29.68,67.64,73.06,76.86,79.99,80.68,82.57,
                        85.74,87.59,87.79,87.72,89.12,89.31,89.08,89.89],
        'Val Acc %'  : [73.60,80.60,83.20,85.00,84.00,85.40,83.40,
                        89.00,88.60,89.40,89.60,89.80,89.80,90.00,90.00],
        'Val Loss'   : [1.0177,0.7572,0.6256,0.5453,0.5540,0.5503,0.5572,
                        0.4220,0.4208,0.4105,0.4049,0.4105,0.3927,0.3884,0.3895],
        'Note'       : ['','','','','','','⬇ LR reduced',
                        '','','','','','','✅ Best','']
    })

    resnet_log = pd.DataFrame({
        'Epoch'      : list(range(1, 16)),
        'Train Acc %': [41.21,79.05,83.42,86.35,87.54,88.11,89.55,
                        90.56,90.92,90.22,91.43,93.75,95.43,95.88,96.08],
        'Val Acc %'  : [83.00,87.60,88.00,88.40,89.40,87.60,89.00,
                        91.00,90.80,89.60,90.80,91.80,91.60,91.80,91.80],
        'Val Loss'   : [0.6522,0.4644,0.4273,0.4222,0.3490,0.3757,0.3448,
                        0.3415,0.3474,0.3693,0.3653,0.3100,0.3038,0.3056,0.3153],
        'Note'       : ['','','','','','','','',
                        '','','⬇ LR reduced','','✅ Best','','']
    })

    with tab1:
        st.dataframe(xception_log, use_container_width=True, hide_index=True)
    with tab2:
        st.dataframe(resnet_log, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🏗️ Model Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Both models use <b>Transfer Learning</b> with frozen ImageNet weights:<br><br>
    <code>Input(224×224×3) → Frozen Base → GlobalAveragePooling2D → Dense(1024, ReLU) → Dropout(0.5) → Dense(100, Softmax)</code><br><br>
    <b>Training:</b> Adam(lr=0.001) | ReduceLROnPlateau(factor=0.2, patience=3) | EarlyStopping(patience=5, restore_best_weights=True)
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT BUTTERFLIES
# ════════════════════════════════════════════════════════════
elif page == "🦋 About Butterflies":
    st.markdown("# 🦋 About Butterflies & Moths")
    st.markdown("Interesting facts about the species this model can classify.")
    st.markdown("---")

    f1, f2, f3 = st.columns(3)
    facts = [
        ("🦋", "20,000+", "Known butterfly species worldwide"),
        ("🌡️", "Cold-blooded", "Cannot fly below 13°C — need warmth to activate"),
        ("👁️", "Ultraviolet", "Can see UV light, revealing hidden wing patterns"),
    ]
    for col, (icon, val, desc) in zip([f1, f2, f3], facts):
        with col:
            st.markdown(f"""<div class="metric-card">
                <div style="font-size:2rem">{icon}</div>
                <div class="metric-value" style="font-size:1.4rem">{val}</div>
                <div class="metric-label">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🦋 Notable Species in This Dataset</div>', unsafe_allow_html=True)

    species_info = {
        "Monarch"        : "Migrates up to 4,800 km. Bright orange with black borders — toxic to predators.",
        "Blue Morpho"    : "Iridescent blue wings caused by microscopic scales, not pigment. Found in Amazon.",
        "Atlas Moth"     : "One of the largest moths — wingspan up to 30 cm. Adults have no mouth and never eat.",
        "Luna Moth"      : "Pale green with elegant tails. Adults live only about 1 week — solely to reproduce.",
        "Viceroy"        : "Mimics the Monarch butterfly for protection despite being a different species.",
        "Mourning Cloak" : "One of the longest-living butterflies, surviving up to 11 months as an adult.",
        "Clearwing Moth" : "Transparent wings mimic bees or wasps as a clever predator defence.",
        "Swallowtail"    : "Named for tail-like extensions. Found on every continent except Antarctica.",
    }

    col_a, col_b = st.columns(2)
    for i, (name, desc) in enumerate(species_info.items()):
        col = col_a if i % 2 == 0 else col_b
        with col:
            st.markdown(f"""<div class="info-box">
                <b style="color:#7eb8f7">🦋 {name}</b><br>{desc}
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🔬 Butterfly vs Moth — Key Differences</div>', unsafe_allow_html=True)
    diff_df = pd.DataFrame({
        'Feature'    : ['Antennae', 'Activity time', 'Wings at rest', 'Pupa type', 'Body shape'],
        'Butterfly 🦋': ['Club-tipped', 'Daytime', 'Held upright', 'Smooth chrysalis', 'Slender'],
        'Moth 🦗'     : ['Feathery or tapered', 'Mostly night', 'Spread flat', 'Silk cocoon', 'Stout and fuzzy']
    })
    st.dataframe(diff_df, use_container_width=True, hide_index=True)