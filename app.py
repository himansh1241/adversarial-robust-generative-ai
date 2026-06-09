import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from torchvision import transforms
from PIL import Image

from model.gan               import Generator, Discriminator
from model.train             import train_gan, NOISE_DIM, DEVICE
from model.classifier        import PneumoniaClassifier
from model.train_classifier  import train_classifier, predict_single_image
from attacks.fgsm            import fgsm_attack
from attacks.pgd             import pgd_attack
from defense.defend          import gaussian_denoise, median_filter_defense, detect_adversarial

# ─────────────────────────────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# Global CSS — professional dark medical theme
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base & fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #60a5fa;
}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f1629;
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #1e2d4a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
    border: none;
    transition: all 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #1e40af !important;
    color: #ffffff !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #e2e8f0 !important;
    background: #1e2d4a !important;
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 24px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 500;
    font-size: 14px;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.4);
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #64748b; font-size: 12px; }
[data-testid="stMetricValue"] { color: #e2e8f0; font-size: 22px; font-weight: 600; }

/* ── Info / warning / success / error boxes ── */
.stAlert {
    border-radius: 8px;
    border-left-width: 3px;
    font-size: 13px;
}

/* ── Sliders ── */
[data-testid="stSlider"] label { color: #94a3b8; font-size: 13px; }

/* ── Selectbox ── */
[data-testid="stSelectbox"] label { color: #94a3b8; font-size: 13px; }

/* ── Radio ── */
.stRadio label { color: #94a3b8; font-size: 13px; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f1629;
    border: 1px dashed #1e2d4a;
    border-radius: 10px;
    padding: 12px;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #1e40af, #3b82f6);
    border-radius: 4px;
}

/* ── Custom cards ── */
.med-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.med-card-title {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 8px;
}
.med-card-value {
    font-size: 26px;
    font-weight: 600;
    color: #e2e8f0;
    font-family: 'JetBrains Mono', monospace;
}
.med-card-sub {
    font-size: 12px;
    color: #475569;
    margin-top: 4px;
}

/* ── Diagnosis pill ── */
.diag-normal {
    display: inline-block;
    background: #052e16;
    color: #4ade80;
    border: 1px solid #166534;
    border-radius: 20px;
    padding: 6px 18px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.diag-pneumonia {
    display: inline-block;
    background: #2d0a0a;
    color: #f87171;
    border: 1px solid #7f1d1d;
    border-radius: 20px;
    padding: 6px 18px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.05em;
}

/* ── Section divider ── */
.section-rule {
    border: none;
    border-top: 1px solid #1e2d4a;
    margin: 24px 0;
}

/* ── Step badge ── */
.step-badge {
    display: inline-block;
    background: #1e2d4a;
    color: #3b82f6;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* ── JSON output ── */
.stJson {
    background: #0a0e1a !important;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #1e2d4a;
}

/* ── Caption ── */
.stCaption { color: #475569 !important; font-size: 11px !important; }

/* ── Pyplot charts ── */
.stPlotlyChart, [data-testid="stPyplotGlobalUse"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1e2d4a;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Helper: styled matplotlib figure
# ─────────────────────────────────────────────────────────────────────
def dark_fig(figsize=(10, 3)):
    fig, ax_or_axes = plt.subplots(
        1, 1, figsize=figsize,
        facecolor="#0f1629"
    ) if len(figsize) == 2 else (None, None)
    return fig

def style_ax(ax):
    ax.set_facecolor("#0a0e1a")
    ax.tick_params(colors="#475569", labelsize=9)
    ax.xaxis.label.set_color("#64748b")
    ax.yaxis.label.set_color("#64748b")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d4a")

# ─────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 32px 0 24px 0;">
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:8px;">
        <div style="background:#1e2d4a; border:1px solid #1e40af;
                    border-radius:10px; padding:10px 14px; font-size:22px;">🛡️</div>
        <div>
            <div style="font-size:26px; font-weight:600; color:#e2e8f0;
                        letter-spacing:-0.02em;">MedShield AI</div>
            <div style="font-size:13px; color:#475569; margin-top:2px;">
                Adversarially Robust Generative AI · Chest X-Ray Analysis
            </div>
        </div>
    </div>
    <hr style="border:none; border-top:1px solid #1e2d4a; margin-top:20px;">
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px 0;">
        <div style="font-size:11px; font-weight:600; letter-spacing:0.1em;
                    text-transform:uppercase; color:#3b82f6; margin-bottom:16px;">
            Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**GAN Training**")
    epochs = st.slider("Epochs", 5, 50, 10, help="More epochs = better quality images")

    st.markdown("<hr style='border-color:#1e2d4a; margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("**Attack Settings**")
    epsilon   = st.slider("Attack strength (ε)", 0.01, 0.5, 0.1,
                          help="Higher = stronger attack, more visible noise")
    pgd_steps = st.slider("PGD iterations", 5, 50, 20,
                          help="More steps = stronger but slower PGD attack")

    st.markdown("<hr style='border-color:#1e2d4a; margin:16px 0'>", unsafe_allow_html=True)
    st.markdown("**Defense**")
    defense = st.selectbox("Defense method",
                           ["Gaussian denoising", "Median filter", "Both"])

    st.markdown("<hr style='border-color:#1e2d4a; margin:16px 0'>", unsafe_allow_html=True)

    # Pipeline status
    st.markdown("""
    <div style="font-size:11px; font-weight:600; letter-spacing:0.1em;
                text-transform:uppercase; color:#3b82f6; margin-bottom:12px;">
        Pipeline Status
    </div>
    """, unsafe_allow_html=True)

    gan_ready = "✅" if "generator" in st.session_state else "⬜"
    clf_ready = "✅" if "classifier" in st.session_state else "⬜"
    atk_ready = "✅" if "adv_img" in st.session_state else "⬜"
    def_ready = "✅" if "defended" in st.session_state else "⬜"

    st.markdown(f"""
    <div style="font-size:13px; line-height:2; color:#94a3b8;">
        {gan_ready} GAN trained<br>
        {clf_ready} Classifier trained<br>
        {atk_ready} Attack executed<br>
        {def_ready} Defense applied
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📊  Train GAN  ",
    "  🫁  Train Classifier  ",
    "  ⚔️  Adversarial Attacks  ",
    "  🛡️  Defense  ",
    "  📂  Upload & Diagnose  ",
])

# ══════════════════════════════════════════════════════════════════════
#  TAB 1 — TRAIN GAN
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="step-badge">Step 01</div>', unsafe_allow_html=True)
    st.markdown("### Generative Adversarial Network")
    st.markdown("""
    <div class="med-card">
        <div class="med-card-title">How it works</div>
        <div style="font-size:13px; color:#94a3b8; line-height:1.7;">
            The <b style="color:#e2e8f0;">Generator</b> learns to synthesize realistic chest X-rays from random noise.
            The <b style="color:#e2e8f0;">Discriminator</b> learns to tell real X-rays from generated ones.
            They compete until the Generator fools the Discriminator consistently.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀  Start GAN Training", key="train_gan"):
        progress_bar = st.progress(0)
        status       = st.empty()

        def update(val):
            progress_bar.progress(val)
            status.markdown(
                f'<div style="font-size:12px; color:#3b82f6;">'
                f'Training · {int(val*100)}% complete</div>',
                unsafe_allow_html=True
            )

        with st.spinner(""):
            gen, disc, g_losses, d_losses = train_gan(
                progress_callback=update,
                epochs_override=epochs
            )
            st.session_state["generator"]     = gen
            st.session_state["discriminator"] = disc

        status.empty()
        st.success("GAN training complete — weights saved to `generator.pth`")

        # Loss curve
        fig, ax = plt.subplots(figsize=(9, 3), facecolor="#0f1629")
        style_ax(ax)
        ax.plot(g_losses, color="#3b82f6",  linewidth=2,
                label="Generator",     marker="o", markersize=4)
        ax.plot(d_losses, color="#f59e0b", linewidth=2,
                label="Discriminator", marker="o", markersize=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Training loss curve", color="#94a3b8", fontsize=11)
        ax.legend(framealpha=0, labelcolor="#94a3b8", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Generated samples grid
        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px; color:#64748b; '
            'letter-spacing:0.05em; text-transform:uppercase; margin-bottom:12px;">'
            'Generated X-ray samples</div>',
            unsafe_allow_html=True
        )
        noise = torch.randn(16, NOISE_DIM).to(DEVICE)
        gen.eval()
        with torch.no_grad():
            samples = gen(noise).cpu()

        fig2, axes = plt.subplots(2, 8, figsize=(14, 4), facecolor="#0a0e1a")
        for i, ax in enumerate(axes.flatten()):
            img_np = samples[i].squeeze().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            ax.imshow(img_np, cmap="bone")
            ax.axis("off")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig2)
        plt.close()

# ══════════════════════════════════════════════════════════════════════
#  TAB 2 — TRAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="step-badge">Step 02</div>', unsafe_allow_html=True)
    st.markdown("### Pneumonia Classifier")
    st.markdown("""
    <div class="med-card">
        <div class="med-card-title">How it works</div>
        <div style="font-size:13px; color:#94a3b8; line-height:1.7;">
            A <b style="color:#e2e8f0;">CNN classifier</b> is trained on real chest X-rays to distinguish
            <span style="color:#4ade80;">NORMAL</span> lungs from
            <span style="color:#f87171;">PNEUMONIA</span> cases.
            This is a separate model from the GAN — it performs actual medical diagnosis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    clf_epochs = st.slider("Training epochs", 5, 30, 10, key="clf_ep")

    if st.button("🫁  Train Classifier", key="train_clf"):
        progress_bar = st.progress(0)
        status       = st.empty()

        def clf_update(val):
            progress_bar.progress(val)
            status.markdown(
                f'<div style="font-size:12px; color:#3b82f6;">'
                f'Training · {int(val*100)}% complete</div>',
                unsafe_allow_html=True
            )

        with st.spinner(""):
            clf, train_losses, val_accs = train_classifier(
                progress_callback=clf_update,
                epochs_override=clf_epochs
            )
            st.session_state["classifier"] = clf

        status.empty()
        st.success("Classifier trained — weights saved to `classifier.pth`")

        # Metrics row
        col1, col2, col3 = st.columns(3)
        col1.metric("Final val accuracy",  f"{val_accs[-1]:.1f}%")
        col2.metric("Final training loss", f"{train_losses[-1]:.4f}")
        col3.metric("Epochs trained",      clf_epochs)

        # Training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), facecolor="#0f1629")
        for ax in (ax1, ax2):
            style_ax(ax)

        ax1.plot(train_losses, color="#3b82f6", linewidth=2, marker="o", markersize=4)
        ax1.set_title("Training loss",        color="#94a3b8", fontsize=11)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")

        ax2.plot(val_accs, color="#4ade80", linewidth=2, marker="o", markersize=4)
        ax2.set_title("Validation accuracy (%)", color="#94a3b8", fontsize=11)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ══════════════════════════════════════════════════════════════════════
#  TAB 3 — ADVERSARIAL ATTACKS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="step-badge">Step 03</div>', unsafe_allow_html=True)
    st.markdown("### Adversarial Attacks")
    st.markdown("""
    <div class="med-card">
        <div class="med-card-title">What is an adversarial attack?</div>
        <div style="font-size:13px; color:#94a3b8; line-height:1.7;">
            Tiny invisible perturbations are added to an image that are imperceptible to the human eye
            but cause the model to make a completely wrong prediction.
            <b style="color:#f59e0b;">FGSM</b> takes one large step.
            <b style="color:#f59e0b;">PGD</b> takes many small steps — stronger but slower.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "discriminator" not in st.session_state:
        st.warning("Complete Step 01 — train the GAN first.")
    else:
        attack_type = st.radio(
            "Attack algorithm",
            ["FGSM", "PGD", "Both"],
            horizontal=True
        )

        if st.button("⚔️  Run Attack", key="run_attack"):
            disc = st.session_state["discriminator"]
            gen  = st.session_state["generator"]

            noise    = torch.randn(1, NOISE_DIM).to(DEVICE)
            fake_img = gen(noise).detach()
            label    = torch.zeros(1, 1).to(DEVICE)

            results = {}
            with st.spinner("Generating adversarial examples..."):
                if attack_type in ["FGSM", "Both"]:
                    results["FGSM"] = fgsm_attack(
                        disc, fake_img, label, epsilon=epsilon
                    )
                if attack_type in ["PGD", "Both"]:
                    results["PGD"] = pgd_attack(
                        disc, fake_img, label,
                        epsilon=epsilon, num_steps=pgd_steps
                    )

            # Results layout
            num_cols  = 1 + len(results)
            cols      = st.columns(num_cols)

            def show_xray(col, tensor, caption):
                arr = tensor.squeeze().detach().cpu().numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                col.image(arr, caption=caption, clamp=True, use_container_width=True)

            with cols[0]:
                show_xray(cols[0], fake_img, "Original (GAN)")
                disc.eval()
                with torch.no_grad():
                    sc = disc(fake_img.to(DEVICE)).item()
                st.markdown(
                    f'<div class="med-card" style="text-align:center;">'
                    f'<div class="med-card-title">Discriminator score</div>'
                    f'<div class="med-card-value">{sc:.3f}</div>'
                    f'<div class="med-card-sub">1.0 = real · 0.0 = fake</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            for i, (name, adv_img) in enumerate(results.items()):
                with cols[i + 1]:
                    show_xray(cols[i + 1], adv_img, f"{name} adversarial")
                    disc.eval()
                    with torch.no_grad():
                        adv_sc = disc(adv_img).item()
                    delta  = adv_sc - sc
                    color  = "#4ade80" if delta > 0 else "#f87171"
                    st.markdown(
                        f'<div class="med-card" style="text-align:center;">'
                        f'<div class="med-card-title">Score after {name}</div>'
                        f'<div class="med-card-value">{adv_sc:.3f}</div>'
                        f'<div class="med-card-sub" style="color:{color};">'
                        f'{"▲" if delta > 0 else "▼"} {abs(delta):.3f} shift</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            st.session_state["fake_img"] = fake_img
            st.session_state["adv_img"]  = list(results.values())[-1]
            st.session_state["label"]    = label

# ══════════════════════════════════════════════════════════════════════
#  TAB 4 — DEFENSE
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="step-badge">Step 04</div>', unsafe_allow_html=True)
    st.markdown("### Defense Framework")
    st.markdown("""
    <div class="med-card">
        <div class="med-card-title">Defense strategy</div>
        <div style="font-size:13px; color:#94a3b8; line-height:1.7;">
            Pre-processing defenses clean the image <i>before</i> the model sees it,
            removing adversarial noise while preserving diagnostic features.
            The detection module flags images that show suspicious confidence drops after smoothing.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "adv_img" not in st.session_state:
        st.warning("Complete Step 03 — run an attack first.")
    else:
        if st.button("🛡️  Apply Defense", key="apply_def"):
            adv_img = st.session_state["adv_img"]
            disc    = st.session_state["discriminator"]

            with st.spinner("Applying defense..."):
                if defense == "Gaussian denoising":
                    defended = gaussian_denoise(adv_img)
                elif defense == "Median filter":
                    defended = median_filter_defense(adv_img)
                else:
                    defended = median_filter_defense(gaussian_denoise(adv_img))

            disc.eval()
            with torch.no_grad():
                adv_sc = disc(adv_img).item()
                def_sc = disc(defended).item()

            # Side-by-side images
            col1, col2 = st.columns(2)

            def show_xray_plain(col, tensor, caption):
                arr = tensor.squeeze().detach().cpu().numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                col.image(arr, caption=caption, clamp=True, use_container_width=True)

            show_xray_plain(col1, adv_img,  "Before defense")
            show_xray_plain(col2, defended, "After defense")

            # Score comparison
            c1, c2, c3 = st.columns(3)
            c1.metric("Score before defense", f"{adv_sc:.4f}")
            c2.metric("Score after defense",  f"{def_sc:.4f}",
                      delta=f"{def_sc - adv_sc:+.4f}")
            c3.metric("Recovery",
                      f"{abs(def_sc - adv_sc):.4f}",
                      delta="score shift")

            st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

            # Detection result
            detection = detect_adversarial(disc, adv_img)
            is_adv    = detection["is_adversarial"]

            st.markdown(
                f'<div class="med-card">'
                f'<div class="med-card-title">Adversarial Detection Result</div>'
                f'<div style="font-size:22px; font-weight:600; margin:8px 0; '
                f'color:{"#f87171" if is_adv else "#4ade80"};">'
                f'{"⚠️  ADVERSARIAL DETECTED" if is_adv else "✅  IMAGE APPEARS CLEAN"}'
                f'</div>'
                f'<div style="font-size:12px; color:#475569; font-family:JetBrains Mono,monospace;">'
                f'confidence drop: {detection["confidence_drop"]:.4f} '
                f'· threshold: {detection["threshold"]}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.session_state["defended"] = defended

# ══════════════════════════════════════════════════════════════════════
#  TAB 5 — UPLOAD & DIAGNOSE
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="step-badge">Step 05</div>', unsafe_allow_html=True)
    st.markdown("### Upload & Diagnose")
    st.markdown("""
    <div class="med-card">
        <div class="med-card-title">How to use</div>
        <div style="font-size:13px; color:#94a3b8; line-height:1.7;">
            Upload a chest X-ray image. The pipeline will classify it as
            <span style="color:#4ade80;">NORMAL</span> or
            <span style="color:#f87171;">PNEUMONIA</span>,
            then attack it and show whether the attack changes the diagnosis —
            demonstrating why adversarial robustness matters in medical AI.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pre-flight checks
    missing = []
    if "discriminator" not in st.session_state:
        missing.append("GAN (Tab 1)")
    if "classifier" not in st.session_state:
        missing.append("Classifier (Tab 2)")
    if missing:
        st.warning(f"Complete these steps first: {', '.join(missing)}")

    uploaded = st.file_uploader(
        "Drop a chest X-ray here — JPG or PNG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        img        = Image.open(uploaded).convert("L").resize((64, 64))
        transform  = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_tensor = transform(img).unsqueeze(0)
        label      = torch.zeros(1, 1)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(uploaded, caption="Uploaded X-ray", use_container_width=True)
            st.caption(f"Tensor shape: {list(img_tensor.shape)}")

        with col2:
            if "classifier" in st.session_state and not missing:
                clf             = st.session_state["classifier"]
                pred, conf      = predict_single_image(clf, img_tensor)
                conf_display    = conf if pred == "PNEUMONIA" else 1 - conf
                pill_class      = "diag-pneumonia" if pred == "PNEUMONIA" else "diag-normal"

                st.markdown(
                    f'<div class="med-card">'
                    f'<div class="med-card-title">Initial diagnosis</div>'
                    f'<span class="{pill_class}">{pred}</span>'
                    f'<div style="font-size:13px; color:#64748b; margin-top:12px;">'
                    f'Confidence: <b style="color:#e2e8f0;">{conf_display:.1%}</b></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)

        if st.button("🚀  Run Full Pipeline", key="run_pipeline") and not missing:
            disc = st.session_state["discriminator"]
            clf  = st.session_state["classifier"]

            with st.spinner("Running adversarial pipeline..."):
                adv     = fgsm_attack(disc, img_tensor, label, epsilon=epsilon)
                def_img = gaussian_denoise(adv)

            # ── Visual comparison ──
            st.markdown(
                '<div style="font-size:11px; color:#475569; letter-spacing:0.08em; '
                'text-transform:uppercase; margin-bottom:12px;">Visual comparison</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            def show(col, tensor, title):
                arr = tensor.squeeze().detach().cpu().numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                col.image(arr, caption=title, clamp=True, use_container_width=True)

            show(c1, img_tensor, "Original")
            show(c2, adv,        f"After FGSM (ε={epsilon})")
            show(c3, def_img,    "After defense")

            # ── Diagnosis comparison ──
            st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:11px; color:#475569; letter-spacing:0.08em; '
                'text-transform:uppercase; margin-bottom:16px;">Diagnosis comparison</div>',
                unsafe_allow_html=True
            )

            pred_orig, conf_orig = predict_single_image(clf, img_tensor)
            pred_adv,  conf_adv  = predict_single_image(clf, adv)
            pred_def,  conf_def  = predict_single_image(clf, def_img)

            def diag_card(col, label_text, pred, conf, subtitle=""):
                conf_d     = conf if pred == "PNEUMONIA" else 1 - conf
                pill_class = "diag-pneumonia" if pred == "PNEUMONIA" else "diag-normal"
                col.markdown(
                    f'<div class="med-card" style="text-align:center;">'
                    f'<div class="med-card-title">{label_text}</div>'
                    f'<span class="{pill_class}">{pred}</span>'
                    f'<div style="font-size:12px; color:#64748b; margin-top:10px;">'
                    f'Confidence: <b style="color:#e2e8f0;">{conf_d:.1%}</b></div>'
                    f'{"<div style=font-size:11px;color:#475569;margin-top:4px;>" + subtitle + "</div>" if subtitle else ""}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            d1, d2, d3 = st.columns(3)
            diag_card(d1, "Original",        pred_orig, conf_orig)
            diag_card(d2, "After attack",    pred_adv,  conf_adv)
            diag_card(d3, "After defense",   pred_def,  conf_def)

            # ── Attack impact banner ──
            st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
            if pred_orig != pred_adv:
                st.markdown(
                    f'<div style="background:#2d0a0a; border:1px solid #7f1d1d; '
                    f'border-radius:10px; padding:16px 20px;">'
                    f'<div style="color:#f87171; font-weight:600; font-size:14px; margin-bottom:4px;">'
                    f'⚠️  Attack changed the diagnosis</div>'
                    f'<div style="color:#94a3b8; font-size:13px;">'
                    f'{pred_orig} → {pred_adv} — This is why adversarial robustness '
                    f'is critical in medical AI systems.</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background:#052e16; border:1px solid #166534; '
                    f'border-radius:10px; padding:16px 20px;">'
                    f'<div style="color:#4ade80; font-weight:600; font-size:14px; margin-bottom:4px;">'
                    f'✅  Model held its diagnosis</div>'
                    f'<div style="color:#94a3b8; font-size:13px;">'
                    f'Diagnosis remained {pred_orig} despite adversarial perturbation.</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # ── Detection result ──
            det = detect_adversarial(disc, adv)
            st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
            st.markdown(
                '<div style="font-size:11px; color:#475569; letter-spacing:0.08em; '
                'text-transform:uppercase; margin-bottom:12px;">Detection report</div>',
                unsafe_allow_html=True
            )
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Adversarial",     "YES" if det["is_adversarial"] else "NO")
            r2.metric("Original score",  det["original_score"])
            r3.metric("Smoothed score",  det["smoothed_score"])
            r4.metric("Confidence drop", det["confidence_drop"])
