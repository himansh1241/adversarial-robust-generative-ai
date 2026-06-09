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
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────
# Theme state  (dark = True by default)
# ─────────────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

dark = st.session_state["dark_mode"]

# ── Palette ──────────────────────────────────────────────────────────
if dark:
    BG       = "#0a0e1a"
    BG2      = "#0f1629"
    BORDER   = "#1e2d4a"
    TEXT1    = "#e2e8f0"
    TEXT2    = "#94a3b8"
    TEXT3    = "#475569"
    ACCENT   = "#1e40af"
    ACCENT2  = "#3b82f6"
    BTN_TXT  = "#ffffff"
    BTN_BG   = "linear-gradient(135deg,#1e40af,#1d4ed8)"
    BTN_HOV  = "linear-gradient(135deg,#1d4ed8,#2563eb)"
    BTN_SHA  = "rgba(30,64,175,0.45)"
    TAB_ACT  = "#1e40af"
    CHART_BG = "#0a0e1a"
    CHART_PL = "#0f1629"
    XRAY_CM  = "bone"
else:
    BG       = "#f0f4f8"
    BG2      = "#ffffff"
    BORDER   = "#cbd5e1"
    TEXT1    = "#0f172a"
    TEXT2    = "#334155"
    TEXT3    = "#64748b"
    ACCENT   = "#1e40af"
    ACCENT2  = "#2563eb"
    BTN_TXT  = "#ffffff"
    BTN_BG   = "linear-gradient(135deg,#1e40af,#2563eb)"
    BTN_HOV  = "linear-gradient(135deg,#1d4ed8,#3b82f6)"
    BTN_SHA  = "rgba(30,64,175,0.25)"
    TAB_ACT  = "#1e40af"
    CHART_BG = "#f8fafc"
    CHART_PL = "#ffffff"
    XRAY_CM  = "gray"

# ─────────────────────────────────────────────────────────────────────
# Global CSS — injected dynamically so theme switch re-renders it
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* ── App background ── */
.stApp {{
    background: {BG};
    color: {TEXT1};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {BG2} !important;
    border-right: 1px solid {BORDER};
}}

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {BG2};
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {TEXT2};
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 500;
    border: none;
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    background: {TAB_ACT} !important;
    color: #ffffff !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {TEXT1} !important;
    background: {BORDER} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    padding-top: 24px;
}}

/* ── ALL buttons — high contrast ── */
.stButton > button {{
    background: {BTN_BG};
    color: {BTN_TXT} !important;
    border: 2px solid {ACCENT2} !important;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 0.01em;
    transition: all 0.2s;
    width: 100%;
    box-shadow: 0 2px 8px {BTN_SHA};
}}
.stButton > button:hover {{
    background: {BTN_HOV};
    transform: translateY(-1px);
    box-shadow: 0 6px 20px {BTN_SHA};
    border-color: #60a5fa !important;
}}
.stButton > button:active {{
    transform: translateY(0);
}}
/* theme-toggle button — smaller, pill shape */
.theme-btn .stButton > button {{
    background: {BG2};
    color: {TEXT2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 500;
    width: auto;
    box-shadow: none;
}}
.theme-btn .stButton > button:hover {{
    background: {BORDER};
    transform: none;
    box-shadow: none;
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px 20px;
}}
[data-testid="stMetricLabel"] {{ color: {TEXT3}; font-size: 12px; }}
[data-testid="stMetricValue"] {{ color: {TEXT1}; font-size: 22px; font-weight: 600; }}

/* ── Alerts ── */
.stAlert {{
    border-radius: 8px;
    border-left-width: 3px;
    font-size: 13px;
    background: {BG2} !important;
}}

/* ── Sliders / selectbox / radio labels ── */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
.stRadio label {{ color: {TEXT2}; font-size: 13px; }}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {BG2};
    border: 1px dashed {BORDER};
    border-radius: 10px;
    padding: 12px;
}}

/* ── Progress bar ── */
.stProgress > div > div {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    border-radius: 4px;
}}

/* ── Cards ── */
.med-card {{
    background: {BG2};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}}
.med-card-title {{
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {ACCENT2};
    margin-bottom: 8px;
}}
.med-card-value {{
    font-size: 26px;
    font-weight: 600;
    color: {TEXT1};
    font-family: 'JetBrains Mono', monospace;
}}
.med-card-sub {{
    font-size: 12px;
    color: {TEXT3};
    margin-top: 4px;
}}

/* ── Diagnosis pills ── */
.diag-normal {{
    display: inline-block;
    background: #052e16;
    color: #4ade80;
    border: 1px solid #166534;
    border-radius: 20px;
    padding: 6px 20px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.05em;
}}
.diag-pneumonia {{
    display: inline-block;
    background: #2d0a0a;
    color: #f87171;
    border: 1px solid #7f1d1d;
    border-radius: 20px;
    padding: 6px 20px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.05em;
}}

/* ── Step badge ── */
.step-badge {{
    display: inline-block;
    background: {BORDER};
    color: {ACCENT2};
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}}

/* ── Section rule ── */
.section-rule {{
    border: none;
    border-top: 1px solid {BORDER};
    margin: 24px 0;
}}

/* ── Images ── */
[data-testid="stImage"] img {{
    border-radius: 8px;
    border: 1px solid {BORDER};
}}

/* ── Caption ── */
.stCaption {{ color: {TEXT3} !important; font-size: 11px !important; }}

/* ── Pyplot ── */
[data-testid="stPyplotGlobalUse"] {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid {BORDER};
}}

/* ── Footer ── */
.footer {{
    margin-top: 60px;
    border-top: 1px solid {BORDER};
    padding: 28px 0 16px 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}}
.footer-left {{
    font-size: 13px;
    color: {TEXT3};
    line-height: 1.6;
}}
.footer-name {{
    font-weight: 600;
    color: {TEXT2};
}}
.footer-link {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {BG2};
    border: 1px solid {BORDER};
    color: {ACCENT2};
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
    text-decoration: none;
    transition: all 0.2s;
}}
.footer-link:hover {{
    background: {ACCENT};
    color: #ffffff;
    border-color: {ACCENT};
}}
.footer-badge {{
    display: inline-block;
    background: {BORDER};
    color: {ACCENT2};
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Matplotlib helpers
# ─────────────────────────────────────────────────────────────────────
def style_ax(ax):
    ax.set_facecolor(CHART_BG)
    ax.tick_params(colors=TEXT3, labelsize=9)
    ax.xaxis.label.set_color(TEXT3)
    ax.yaxis.label.set_color(TEXT3)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)

# ─────────────────────────────────────────────────────────────────────
# Hero header  +  theme toggle
# ─────────────────────────────────────────────────────────────────────
hcol1, hcol2 = st.columns([6, 1])
with hcol1:
    st.markdown(f"""
    <div style="padding:28px 0 20px 0;">
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">
            <div style="background:{BG2}; border:1px solid {ACCENT};
                        border-radius:10px; padding:10px 14px; font-size:22px;">🛡️</div>
            <div>
                <div style="font-size:26px; font-weight:700; color:{TEXT1};
                            letter-spacing:-0.02em;">MedShield AI</div>
                <div style="font-size:12px; color:{TEXT3}; margin-top:2px;
                            letter-spacing:0.04em; text-transform:uppercase;">
                    Adversarially Robust Generative AI · Chest X-Ray Analysis
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hcol2:
    st.markdown('<div style="padding-top:30px;">', unsafe_allow_html=True)
    toggle_label = "☀️  Light" if dark else "🌙  Dark"
    st.markdown('<div class="theme-btn">', unsafe_allow_html=True)
    if st.button(toggle_label, key="theme_toggle"):
        st.session_state["dark_mode"] = not dark
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:0 0 8px 0;">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 0 20px 0;">
        <div style="font-size:11px; font-weight:600; letter-spacing:0.1em;
                    text-transform:uppercase; color:{ACCENT2}; margin-bottom:4px;">
            Configuration
        </div>
        <div style="font-size:12px; color:{TEXT3};">Adjust before running each step</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div style="color:{TEXT2}; font-size:13px; font-weight:600; margin-bottom:6px;">GAN Training</div>', unsafe_allow_html=True)
    epochs = st.slider("Epochs", 5, 50, 10, help="More epochs = better quality fake X-rays")

    st.markdown(f'<hr style="border-color:{BORDER};margin:16px 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{TEXT2}; font-size:13px; font-weight:600; margin-bottom:6px;">Attack Settings</div>', unsafe_allow_html=True)
    epsilon   = st.slider("Attack strength (ε)", 0.01, 0.5, 0.1,
                          help="Higher = stronger attack, more visible noise")
    pgd_steps = st.slider("PGD iterations", 5, 50, 20,
                          help="More steps = stronger but slower PGD attack")

    st.markdown(f'<hr style="border-color:{BORDER};margin:16px 0">', unsafe_allow_html=True)
    st.markdown(f'<div style="color:{TEXT2}; font-size:13px; font-weight:600; margin-bottom:6px;">Defense</div>', unsafe_allow_html=True)
    defense = st.selectbox("Method", ["Gaussian denoising", "Median filter", "Both"])

    st.markdown(f'<hr style="border-color:{BORDER};margin:16px 0">', unsafe_allow_html=True)

    # Pipeline status checklist
    st.markdown(f"""
    <div style="font-size:11px; font-weight:600; letter-spacing:0.1em;
                text-transform:uppercase; color:{ACCENT2}; margin-bottom:12px;">
        Pipeline Status
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("generator"     in st.session_state, "GAN trained"),
        ("classifier"    in st.session_state, "Classifier trained"),
        ("adv_img"       in st.session_state, "Attack executed"),
        ("defended"      in st.session_state, "Defense applied"),
    ]
    rows = "".join(
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">'
        f'<span style="font-size:15px;">{"✅" if done else "⬜"}</span>'
        f'<span style="font-size:13px;color:{"#4ade80" if done else TEXT3};">{label}</span>'
        f'</div>'
        for done, label in steps
    )
    st.markdown(rows, unsafe_allow_html=True)

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
# TAB 1 — TRAIN GAN
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="step-badge">Step 01</div>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{TEXT1};margin-top:4px;">Generative Adversarial Network</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="med-card">
        <div class="med-card-title">How it works</div>
        <div style="font-size:13px; color:{TEXT2}; line-height:1.7;">
            The <b style="color:{TEXT1};">Generator</b> learns to synthesize realistic chest X-rays from random noise.
            The <b style="color:{TEXT1};">Discriminator</b> learns to tell real X-rays from generated ones.
            They compete until the Generator consistently fools the Discriminator.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀  Start GAN Training", key="train_gan"):
        progress_bar = st.progress(0)
        status       = st.empty()

        def update(val):
            progress_bar.progress(val)
            status.markdown(
                f'<div style="font-size:12px;color:{ACCENT2};">Training · {int(val*100)}% complete</div>',
                unsafe_allow_html=True
            )

        with st.spinner(""):
            gen, disc, g_losses, d_losses = train_gan(
                progress_callback=update, epochs_override=epochs
            )
            st.session_state["generator"]     = gen
            st.session_state["discriminator"] = disc

        status.empty()
        st.success("GAN training complete — weights saved to `generator.pth`")

        fig, ax = plt.subplots(figsize=(9, 3), facecolor=CHART_PL)
        style_ax(ax)
        ax.plot(g_losses, color=ACCENT2, linewidth=2, label="Generator",     marker="o", markersize=4)
        ax.plot(d_losses, color="#f59e0b", linewidth=2, label="Discriminator", marker="o", markersize=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Training loss curve", color=TEXT2, fontsize=11)
        ax.legend(framealpha=0, labelcolor=TEXT2, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(f'<hr class="section-rule">', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:11px;color:{TEXT3};letter-spacing:0.05em;text-transform:uppercase;margin-bottom:12px;">Generated X-ray samples</div>', unsafe_allow_html=True)

        noise = torch.randn(16, NOISE_DIM).to(DEVICE)
        gen.eval()
        with torch.no_grad():
            samples = gen(noise).cpu()

        fig2, axes = plt.subplots(2, 8, figsize=(14, 4), facecolor=CHART_BG)
        for i, ax in enumerate(axes.flatten()):
            arr = samples[i].squeeze().numpy()
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            ax.imshow(arr, cmap=XRAY_CM); ax.axis("off")
        plt.tight_layout(pad=0.3)
        st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════════════
# TAB 2 — TRAIN CLASSIFIER
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="step-badge">Step 02</div>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{TEXT1};margin-top:4px;">Pneumonia Classifier</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="med-card">
        <div class="med-card-title">How it works</div>
        <div style="font-size:13px; color:{TEXT2}; line-height:1.7;">
            A <b style="color:{TEXT1};">CNN classifier</b> is trained on real chest X-rays to distinguish
            <span style="color:#4ade80;">NORMAL</span> lungs from
            <span style="color:#f87171;">PNEUMONIA</span> cases.
            This is separate from the GAN — it performs actual medical diagnosis.
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
                f'<div style="font-size:12px;color:{ACCENT2};">Training · {int(val*100)}% complete</div>',
                unsafe_allow_html=True
            )

        with st.spinner(""):
            clf, train_losses, val_accs = train_classifier(
                progress_callback=clf_update, epochs_override=clf_epochs
            )
            st.session_state["classifier"] = clf

        status.empty()
        st.success("Classifier trained — weights saved to `classifier.pth`")

        col1, col2, col3 = st.columns(3)
        col1.metric("Final val accuracy",  f"{val_accs[-1]:.1f}%")
        col2.metric("Final training loss", f"{train_losses[-1]:.4f}")
        col3.metric("Epochs trained",      clf_epochs)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), facecolor=CHART_PL)
        for ax in (ax1, ax2): style_ax(ax)
        ax1.plot(train_losses, color=ACCENT2,    linewidth=2, marker="o", markersize=4)
        ax1.set_title("Training loss",           color=TEXT2, fontsize=11)
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
        ax2.plot(val_accs,     color="#4ade80",  linewidth=2, marker="o", markersize=4)
        ax2.set_title("Validation accuracy (%)", color=TEXT2, fontsize=11)
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════
# TAB 3 — ADVERSARIAL ATTACKS
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="step-badge">Step 03</div>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{TEXT1};margin-top:4px;">Adversarial Attacks</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="med-card">
        <div class="med-card-title">What is an adversarial attack?</div>
        <div style="font-size:13px; color:{TEXT2}; line-height:1.7;">
            Tiny invisible perturbations added to an image — imperceptible to the human eye —
            can cause the model to make completely wrong predictions.
            <b style="color:#f59e0b;">FGSM</b> takes one large step in the gradient direction.
            <b style="color:#f59e0b;">PGD</b> takes many small steps — stronger but slower.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "discriminator" not in st.session_state:
        st.warning("Complete Step 01 — train the GAN first.")
    else:
        attack_type = st.radio("Attack algorithm", ["FGSM", "PGD", "Both"], horizontal=True)

        if st.button("⚔️  Run Attack", key="run_attack"):
            disc = st.session_state["discriminator"]
            gen  = st.session_state["generator"]
            noise    = torch.randn(1, NOISE_DIM).to(DEVICE)
            fake_img = gen(noise).detach()
            label    = torch.zeros(1, 1).to(DEVICE)

            results = {}
            with st.spinner("Generating adversarial examples..."):
                if attack_type in ["FGSM", "Both"]:
                    results["FGSM"] = fgsm_attack(disc, fake_img, label, epsilon=epsilon)
                if attack_type in ["PGD", "Both"]:
                    results["PGD"]  = pgd_attack(disc, fake_img, label, epsilon=epsilon, num_steps=pgd_steps)

            cols = st.columns(1 + len(results))

            def show_xray(col, tensor, caption):
                arr = tensor.squeeze().detach().cpu().numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                col.image(arr, caption=caption, clamp=True, use_container_width=True)

            disc.eval()
            with torch.no_grad():
                sc = disc(fake_img.to(DEVICE)).item()

            with cols[0]:
                show_xray(cols[0], fake_img, "Original (GAN)")
                st.markdown(
                    f'<div class="med-card" style="text-align:center;">'
                    f'<div class="med-card-title">Discriminator score</div>'
                    f'<div class="med-card-value">{sc:.3f}</div>'
                    f'<div class="med-card-sub">1.0 = real · 0.0 = fake</div></div>',
                    unsafe_allow_html=True
                )

            for i, (name, adv_img) in enumerate(results.items()):
                with cols[i + 1]:
                    show_xray(cols[i + 1], adv_img, f"{name} adversarial")
                    with torch.no_grad():
                        adv_sc = disc(adv_img).item()
                    delta = adv_sc - sc
                    color = "#4ade80" if delta > 0 else "#f87171"
                    st.markdown(
                        f'<div class="med-card" style="text-align:center;">'
                        f'<div class="med-card-title">Score after {name}</div>'
                        f'<div class="med-card-value">{adv_sc:.3f}</div>'
                        f'<div class="med-card-sub" style="color:{color};">'
                        f'{"▲" if delta>0 else "▼"} {abs(delta):.3f} shift</div></div>',
                        unsafe_allow_html=True
                    )

            st.session_state["fake_img"] = fake_img
            st.session_state["adv_img"]  = list(results.values())[-1]
            st.session_state["label"]    = label

# ══════════════════════════════════════════════════════════════════════
# TAB 4 — DEFENSE
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="step-badge">Step 04</div>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{TEXT1};margin-top:4px;">Defense Framework</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="med-card">
        <div class="med-card-title">Defense strategy</div>
        <div style="font-size:13px; color:{TEXT2}; line-height:1.7;">
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

            col1, col2 = st.columns(2)

            def show_plain(col, tensor, caption):
                arr = tensor.squeeze().detach().cpu().numpy()
                arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                col.image(arr, caption=caption, clamp=True, use_container_width=True)

            show_plain(col1, adv_img,  "Before defense")
            show_plain(col2, defended, "After defense")

            c1, c2, c3 = st.columns(3)
            c1.metric("Score before defense", f"{adv_sc:.4f}")
            c2.metric("Score after defense",  f"{def_sc:.4f}", delta=f"{def_sc-adv_sc:+.4f}")
            c3.metric("Score shift",          f"{abs(def_sc-adv_sc):.4f}")

            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

            detection = detect_adversarial(disc, adv_img)
            is_adv    = detection["is_adversarial"]
            det_color = "#f87171" if is_adv else "#4ade80"
            det_bg    = "#2d0a0a" if is_adv else "#052e16"
            det_bdr   = "#7f1d1d" if is_adv else "#166534"
            det_label = "⚠️  ADVERSARIAL DETECTED" if is_adv else "✅  IMAGE APPEARS CLEAN"

            st.markdown(
                f'<div style="background:{det_bg};border:1px solid {det_bdr};'
                f'border-radius:10px;padding:18px 22px;">'
                f'<div style="color:{det_color};font-weight:700;font-size:15px;margin-bottom:6px;">{det_label}</div>'
                f'<div style="font-size:12px;color:{TEXT3};font-family:JetBrains Mono,monospace;">'
                f'confidence drop: {detection["confidence_drop"]:.4f} · threshold: {detection["threshold"]}</div>'
                f'</div>',
                unsafe_allow_html=True
            )
            st.session_state["defended"] = defended

# ══════════════════════════════════════════════════════════════════════
# TAB 5 — UPLOAD & DIAGNOSE
# ══════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="step-badge">Step 05</div>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="color:{TEXT1};margin-top:4px;">Upload & Diagnose</h3>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="med-card">
        <div class="med-card-title">How to use</div>
        <div style="font-size:13px; color:{TEXT2}; line-height:1.7;">
            Upload a chest X-ray. The pipeline classifies it as
            <span style="color:#4ade80;">NORMAL</span> or <span style="color:#f87171;">PNEUMONIA</span>,
            then attacks it and shows whether the adversarial perturbation changes the diagnosis —
            demonstrating why robustness matters in medical AI.
        </div>
    </div>
    """, unsafe_allow_html=True)

    missing = []
    if "discriminator" not in st.session_state: missing.append("GAN (Tab 1)")
    if "classifier"    not in st.session_state: missing.append("Classifier (Tab 2)")
    if missing:
        st.warning(f"Complete these steps first: {', '.join(missing)}")

    uploaded = st.file_uploader("Drop a chest X-ray here — JPG or PNG", type=["jpg","jpeg","png"])

    if uploaded:
        img        = Image.open(uploaded).convert("L").resize((64,64))
        transform  = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
        img_tensor = transform(img).unsqueeze(0)
        label      = torch.zeros(1,1)

        col1, col2 = st.columns([1,2])
        with col1:
            st.image(uploaded, caption="Uploaded X-ray", use_container_width=True)
            st.caption(f"Tensor shape: {list(img_tensor.shape)}")
        with col2:
            if "classifier" in st.session_state:
                clf = st.session_state["classifier"]
                pred, conf   = predict_single_image(clf, img_tensor)
                conf_display = conf if pred=="PNEUMONIA" else 1-conf
                pill         = "diag-pneumonia" if pred=="PNEUMONIA" else "diag-normal"
                st.markdown(
                    f'<div class="med-card">'
                    f'<div class="med-card-title">Initial diagnosis</div>'
                    f'<span class="{pill}">{pred}</span>'
                    f'<div style="font-size:13px;color:{TEXT3};margin-top:12px;">'
                    f'Confidence: <b style="color:{TEXT1};">{conf_display:.1%}</b></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown('<hr class="section-rule">', unsafe_allow_html=True)

        if st.button("🚀  Run Full Pipeline", key="run_pipeline") and not missing:
            disc = st.session_state["discriminator"]
            clf  = st.session_state["classifier"]

            with st.spinner("Running adversarial pipeline..."):
                adv     = fgsm_attack(disc, img_tensor, label, epsilon=epsilon)
                def_img = gaussian_denoise(adv)

            st.markdown(f'<div style="font-size:11px;color:{TEXT3};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;">Visual comparison</div>', unsafe_allow_html=True)

            c1,c2,c3 = st.columns(3)
            def show(col, tensor, title):
                arr = tensor.squeeze().detach().cpu().numpy()
                arr = (arr-arr.min())/(arr.max()-arr.min()+1e-8)
                col.image(arr, caption=title, clamp=True, use_container_width=True)

            show(c1, img_tensor, "Original")
            show(c2, adv,        f"After FGSM (ε={epsilon})")
            show(c3, def_img,    "After defense")

            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:11px;color:{TEXT3};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:16px;">Diagnosis comparison</div>', unsafe_allow_html=True)

            pred_orig, conf_orig = predict_single_image(clf, img_tensor)
            pred_adv,  conf_adv  = predict_single_image(clf, adv)
            pred_def,  conf_def  = predict_single_image(clf, def_img)

            def diag_card(col, label_text, pred, conf):
                conf_d = conf if pred=="PNEUMONIA" else 1-conf
                pill   = "diag-pneumonia" if pred=="PNEUMONIA" else "diag-normal"
                col.markdown(
                    f'<div class="med-card" style="text-align:center;">'
                    f'<div class="med-card-title">{label_text}</div>'
                    f'<span class="{pill}">{pred}</span>'
                    f'<div style="font-size:12px;color:{TEXT3};margin-top:10px;">'
                    f'Confidence: <b style="color:{TEXT1};">{conf_d:.1%}</b></div></div>',
                    unsafe_allow_html=True
                )

            d1,d2,d3 = st.columns(3)
            diag_card(d1, "Original",      pred_orig, conf_orig)
            diag_card(d2, "After attack",  pred_adv,  conf_adv)
            diag_card(d3, "After defense", pred_def,  conf_def)

            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            if pred_orig != pred_adv:
                st.markdown(
                    f'<div style="background:#2d0a0a;border:1px solid #7f1d1d;border-radius:10px;padding:16px 20px;">'
                    f'<div style="color:#f87171;font-weight:700;font-size:14px;margin-bottom:4px;">⚠️  Attack changed the diagnosis</div>'
                    f'<div style="color:{TEXT2};font-size:13px;">{pred_orig} → {pred_adv} — This is why adversarial robustness is critical in medical AI systems.</div>'
                    f'</div>', unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background:#052e16;border:1px solid #166534;border-radius:10px;padding:16px 20px;">'
                    f'<div style="color:#4ade80;font-weight:700;font-size:14px;margin-bottom:4px;">✅  Model held its diagnosis</div>'
                    f'<div style="color:{TEXT2};font-size:13px;">Diagnosis remained {pred_orig} despite the adversarial perturbation.</div>'
                    f'</div>', unsafe_allow_html=True
                )

            det = detect_adversarial(disc, adv)
            st.markdown('<hr class="section-rule">', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:11px;color:{TEXT3};letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;">Detection report</div>', unsafe_allow_html=True)
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Adversarial",     "YES" if det["is_adversarial"] else "NO")
            r2.metric("Original score",  det["original_score"])
            r3.metric("Smoothed score",  det["smoothed_score"])
            r4.metric("Confidence drop", det["confidence_drop"])

# ─────────────────────────────────────────────────────────────────────
# Footer — credits
# ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-rule" style="margin-top:60px;">', unsafe_allow_html=True)
st.markdown(f"""
<div class="footer">
    <div class="footer-left">
        <div class="footer-badge">Project</div><br>
        <span class="footer-name">Himanshu Ranjan</span><br>
        <span style="color:{TEXT3};font-size:12px;">
            Adversarially Robust Generative AI · Chest X-Ray Analysis<br>
            Built with PyTorch · Streamlit · Apple Silicon (M4)
        </span>
    </div>
    <a class="footer-link"
       href="https://www.linkedin.com/in/himanshuranjan1241"
       target="_blank">
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
             viewBox="0 0 24 24" fill="currentColor">
            <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037
                     -1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046
                     c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286z
                     M5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782
                     13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542
                     C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729
                     C24 .774 23.2 0 22.222 0h.003z"/>
        </svg>
        Connect on LinkedIn
    </a>
</div>
<div style="text-align:center; padding:12px 0 4px; font-size:11px; color:{TEXT3};">
    MedShield AI · For educational and research purposes only · Not for clinical use
</div>
""", unsafe_allow_html=True)
