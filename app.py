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

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme state ───────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True
dark = st.session_state["dark_mode"]

# ── Token system ──────────────────────────────────────────────────────
if dark:
    BG        = "#070b14"
    BG2       = "#0d1424"
    BG3       = "#111827"
    BORDER    = "#1a2640"
    BORDER2   = "#243351"
    TEXT1     = "#f0f4ff"
    TEXT2     = "#8b9dc3"
    TEXT3     = "#3d5175"
    ACCENT    = "#0ea5e9"
    ACCENT_DK = "#0284c7"
    ACCENT_GL = "rgba(14,165,233,0.12)"
    GLOW      = "rgba(14,165,233,0.25)"
    CHART_BG  = "#070b14"
    CHART_PL  = "#0d1424"
    XRAY_CM   = "bone"
    PULSE_C   = "#0ea5e9"
    TAG_BG    = "rgba(14,165,233,0.10)"
    TAG_CLR   = "#38bdf8"
else:
    BG        = "#f7f9fc"
    BG2       = "#ffffff"
    BG3       = "#eef2f8"
    BORDER    = "#dde3ee"
    BORDER2   = "#c8d2e6"
    TEXT1     = "#0d1424"
    TEXT2     = "#3d5175"
    TEXT3     = "#8b9dc3"
    ACCENT    = "#0284c7"
    ACCENT_DK = "#0369a1"
    ACCENT_GL = "rgba(2,132,199,0.08)"
    GLOW      = "rgba(2,132,199,0.18)"
    CHART_BG  = "#f7f9fc"
    CHART_PL  = "#ffffff"
    XRAY_CM   = "gray"
    PULSE_C   = "#0284c7"
    TAG_BG    = "rgba(2,132,199,0.08)"
    TAG_CLR   = "#0284c7"

# ── CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{BG}; }}
::-webkit-scrollbar-thumb {{ background:{BORDER2}; border-radius:10px; }}

/* ── Base ── */
.stApp {{ background:{BG}; color:{TEXT1}; }}
.block-container {{ padding:0 2.5rem 3rem 2.5rem !important; max-width:1280px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background:{BG2} !important;
    border-right:1px solid {BORDER};
}}
[data-testid="stSidebar"] > div:first-child {{ padding:1.5rem 1.2rem; }}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background:{BG2};
    border-radius:14px;
    padding:5px;
    gap:3px;
    border:1px solid {BORDER};
    box-shadow:0 2px 12px rgba(0,0,0,0.15);
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent;
    color:{TEXT2};
    border-radius:10px;
    padding:9px 20px;
    font-size:13px;
    font-weight:500;
    border:none;
    transition:all 0.2s ease;
    letter-spacing:0.01em;
}}
.stTabs [aria-selected="true"] {{
    background:linear-gradient(135deg,{ACCENT_DK},{ACCENT}) !important;
    color:#fff !important;
    box-shadow:0 2px 10px {GLOW};
}}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {{
    color:{TEXT1} !important;
    background:{BORDER} !important;
}}
.stTabs [data-baseweb="tab-panel"] {{ padding-top:28px; }}

/* ── Primary action buttons ── */
.stButton > button {{
    background:linear-gradient(135deg,{ACCENT_DK},{ACCENT});
    color:#fff !important;
    border:none !important;
    border-radius:10px;
    padding:11px 28px;
    font-weight:600;
    font-size:14px;
    letter-spacing:0.02em;
    transition:all 0.22s ease;
    width:100%;
    box-shadow:0 3px 14px {GLOW};
    position:relative;
    overflow:hidden;
}}
.stButton > button::after {{
    content:'';
    position:absolute;
    inset:0;
    background:linear-gradient(135deg,rgba(255,255,255,0.08),transparent);
    border-radius:10px;
    opacity:0;
    transition:opacity 0.2s;
}}
.stButton > button:hover {{
    transform:translateY(-2px);
    box-shadow:0 8px 24px {GLOW};
    filter:brightness(1.08);
}}
.stButton > button:hover::after {{ opacity:1; }}
.stButton > button:active {{ transform:translateY(0); }}

/* ── Theme pill ── */
.theme-btn .stButton > button {{
    background:{BG3} !important;
    color:{TEXT2} !important;
    border:1px solid {BORDER2} !important;
    border-radius:20px !important;
    padding:5px 14px !important;
    font-size:12px !important;
    font-weight:500 !important;
    width:auto !important;
    box-shadow:none !important;
    letter-spacing:0;
}}
.theme-btn .stButton > button:hover {{
    background:{BORDER} !important;
    transform:none !important;
    box-shadow:none !important;
    filter:none !important;
}}

/* ── Metrics ── */
[data-testid="stMetric"] {{
    background:{BG2};
    border:1px solid {BORDER};
    border-radius:12px;
    padding:16px 20px;
    transition:border-color 0.2s;
}}
[data-testid="stMetric"]:hover {{ border-color:{BORDER2}; }}
[data-testid="stMetricLabel"] {{ color:{TEXT3}; font-size:11px; letter-spacing:0.07em; text-transform:uppercase; }}
[data-testid="stMetricValue"] {{ color:{TEXT1}; font-size:22px; font-weight:700; font-family:'JetBrains Mono',monospace; }}
[data-testid="stMetricDelta"]  {{ font-size:12px; }}

/* ── Alerts ── */
.stAlert {{ border-radius:10px; border-left-width:3px; font-size:13px; background:{BG2} !important; }}

/* ── Form controls ── */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label,
.stRadio > label {{ color:{TEXT2}; font-size:12px; font-weight:500; letter-spacing:0.04em; text-transform:uppercase; }}
[data-testid="stSlider"] [data-testid="stTickBar"] {{ background:{BORDER}; }}
div[data-baseweb="select"] > div {{ background:{BG3} !important; border-color:{BORDER2} !important; color:{TEXT1} !important; border-radius:8px !important; }}
.stRadio [data-testid="stMarkdownContainer"] p {{ color:{TEXT2}; font-size:13px; }}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background:{BG2};
    border:1.5px dashed {BORDER2};
    border-radius:12px;
    padding:20px;
    transition:border-color 0.2s;
}}
[data-testid="stFileUploader"]:hover {{ border-color:{ACCENT}; }}

/* ── Progress ── */
.stProgress > div > div {{
    background:linear-gradient(90deg,{ACCENT_DK},{ACCENT},#38bdf8);
    border-radius:6px;
    transition:width 0.3s ease;
}}
.stProgress > div {{ background:{BORDER}; border-radius:6px; }}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color:{ACCENT} !important; }}

/* ── Images ── */
[data-testid="stImage"] img {{
    border-radius:10px;
    border:1px solid {BORDER};
    transition:border-color 0.2s, box-shadow 0.2s;
}}
[data-testid="stImage"] img:hover {{
    border-color:{ACCENT};
    box-shadow:0 0 0 3px {ACCENT_GL};
}}

/* ── Caption ── */
.stCaption {{ color:{TEXT3} !important; font-size:11px !important; letter-spacing:0.02em; }}

/* ── Pyplot ── */
[data-testid="stPyplotGlobalUse"] {{
    border-radius:12px;
    overflow:hidden;
    border:1px solid {BORDER};
}}

/* ── Custom components ── */
.ms-card {{
    background:{BG2};
    border:1px solid {BORDER};
    border-radius:14px;
    padding:22px 26px;
    margin-bottom:18px;
    transition:border-color 0.2s, box-shadow 0.2s;
}}
.ms-card:hover {{ border-color:{BORDER2}; box-shadow:0 4px 20px rgba(0,0,0,0.12); }}
.ms-card-accent {{
    background:{BG2};
    border:1px solid {BORDER};
    border-top:3px solid {ACCENT};
    border-radius:14px;
    padding:22px 26px;
    margin-bottom:18px;
}}
.ms-eyebrow {{
    font-size:10px;
    font-weight:700;
    letter-spacing:0.12em;
    text-transform:uppercase;
    color:{ACCENT};
    margin-bottom:6px;
}}
.ms-section-label {{
    font-size:10px;
    font-weight:700;
    letter-spacing:0.14em;
    text-transform:uppercase;
    color:{TEXT3};
    margin-bottom:14px;
    padding-bottom:8px;
    border-bottom:1px solid {BORDER};
}}
.ms-rule {{ border:none; border-top:1px solid {BORDER}; margin:28px 0; }}
.ms-tag {{
    display:inline-block;
    background:{TAG_BG};
    color:{TAG_CLR};
    border:1px solid {TAG_CLR}33;
    border-radius:5px;
    padding:2px 8px;
    font-size:10px;
    font-weight:600;
    letter-spacing:0.08em;
    text-transform:uppercase;
    margin-bottom:10px;
}}
.ms-mono {{
    font-family:'JetBrains Mono',monospace;
    font-size:12px;
    color:{TEXT2};
}}
.diag-normal {{
    display:inline-flex; align-items:center; gap:7px;
    background:#022c22; color:#34d399;
    border:1px solid #065f46; border-radius:25px;
    padding:8px 22px; font-size:14px; font-weight:700;
    letter-spacing:0.06em;
}}
.diag-pneumonia {{
    display:inline-flex; align-items:center; gap:7px;
    background:#2d0a0a; color:#f87171;
    border:1px solid #7f1d1d; border-radius:25px;
    padding:8px 22px; font-size:14px; font-weight:700;
    letter-spacing:0.06em;
}}
.stat-row {{
    display:flex; align-items:baseline; gap:8px;
    margin:4px 0;
}}
.stat-val {{
    font-size:28px; font-weight:700;
    font-family:'JetBrains Mono',monospace;
    color:{TEXT1};
    line-height:1;
}}
.stat-unit {{ font-size:12px; color:{TEXT3}; }}
.hero-pulse {{
    display:inline-block;
    width:8px; height:8px;
    border-radius:50%;
    background:{PULSE_C};
    box-shadow:0 0 0 0 {GLOW};
    animation:pulse 2s ease-in-out infinite;
    margin-right:6px;
    vertical-align:middle;
}}
@keyframes pulse {{
    0%,100% {{ box-shadow:0 0 0 0 {GLOW}; }}
    50%      {{ box-shadow:0 0 0 7px rgba(14,165,233,0); }}
}}
.score-bar-track {{
    height:6px; background:{BORDER}; border-radius:6px;
    margin:8px 0 4px;
}}
.footer-wrap {{
    margin-top:64px;
    border-top:1px solid {BORDER};
    padding:32px 0 20px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:16px;
}}
.footer-li-btn {{
    display:inline-flex; align-items:center; gap:8px;
    background:{BG2};
    border:1px solid {BORDER2};
    color:{ACCENT};
    border-radius:10px;
    padding:10px 20px;
    font-size:13px;
    font-weight:600;
    text-decoration:none;
    transition:all 0.2s;
    letter-spacing:0.01em;
}}
.footer-li-btn:hover {{
    background:{ACCENT_DK};
    color:#fff;
    border-color:{ACCENT_DK};
    box-shadow:0 4px 14px {GLOW};
}}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────
def style_ax(ax, facecolor=None):
    fc = facecolor or CHART_BG
    ax.set_facecolor(fc)
    ax.tick_params(colors=TEXT3, labelsize=9)
    ax.xaxis.label.set_color(TEXT3)
    ax.yaxis.label.set_color(TEXT3)
    for s in ax.spines.values():
        s.set_edgecolor(BORDER)

def tensor_to_display(t):
    a = t.squeeze().detach().cpu().numpy()
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

def show_xray(col, tensor, caption, use_cw=True):
    col.image(tensor_to_display(tensor), caption=caption,
              clamp=True, use_container_width=use_cw)

def score_bar(val, label=""):
    pct  = int(val * 100)
    col  = ACCENT if val < 0.5 else "#34d399" if val > 0.8 else "#f59e0b"
    return (f'<div class="ms-mono" style="margin-bottom:2px;">'
            f'{label}: <b style="color:{TEXT1};">{val:.3f}</b></div>'
            f'<div class="score-bar-track">'
            f'<div style="width:{pct}%;height:100%;background:{col};'
            f'border-radius:6px;transition:width 0.4s ease;"></div></div>')

def diag_card_html(title, pred, conf):
    conf_d = conf if pred == "PNEUMONIA" else 1 - conf
    pill   = "diag-pneumonia" if pred == "PNEUMONIA" else "diag-normal"
    dot    = "🔴" if pred == "PNEUMONIA" else "🟢"
    return (f'<div class="ms-card" style="text-align:center;padding:24px 16px;">'
            f'<div class="ms-eyebrow">{title}</div>'
            f'<div style="margin:12px 0 10px;"><span class="{pill}">{dot} {pred}</span></div>'
            f'<div class="ms-mono" style="color:{TEXT2};margin-top:10px;">'
            f'confidence <b style="color:{TEXT1};">{conf_d:.1%}</b></div>'
            f'</div>')

# ── Header ─────────────────────────────────────────────────────────────
hc1, hc2 = st.columns([7, 1])
with hc1:
    st.markdown(f"""
    <div style="padding:32px 0 16px;">
        <div style="display:flex;align-items:center;gap:16px;">
            <div style="background:linear-gradient(135deg,{ACCENT_DK},{ACCENT});
                        border-radius:14px;padding:12px 15px;font-size:24px;
                        box-shadow:0 4px 18px {GLOW};">🛡️</div>
            <div>
                <div style="font-size:28px;font-weight:700;color:{TEXT1};
                            letter-spacing:-0.03em;line-height:1.1;">MedShield AI</div>
                <div style="display:flex;align-items:center;gap:8px;margin-top:5px;">
                    <span class="hero-pulse"></span>
                    <span style="font-size:12px;color:{TEXT3};letter-spacing:0.05em;
                                 text-transform:uppercase;">
                        Adversarially Robust Generative AI · Chest X-Ray Analysis
                    </span>
                </div>
            </div>
        </div>
        <div style="display:flex;gap:8px;margin-top:16px;flex-wrap:wrap;">
            <span class="ms-tag">PyTorch</span>
            <span class="ms-tag">GAN</span>
            <span class="ms-tag">FGSM · PGD</span>
            <span class="ms-tag">CNN Classifier</span>
            <span class="ms-tag">Apple M4</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hc2:
    st.markdown('<div style="padding-top:38px;display:flex;justify-content:flex-end;">', unsafe_allow_html=True)
    lbl = "☀️ Light" if dark else "🌙 Dark"
    st.markdown('<div class="theme-btn">', unsafe_allow_html=True)
    if st.button(lbl, key="theme_toggle"):
        st.session_state["dark_mode"] = not dark
        st.rerun()
    st.markdown('</div></div>', unsafe_allow_html=True)

st.markdown(f'<div style="height:1px;background:linear-gradient(90deg,{ACCENT}44,{BORDER},transparent);margin-bottom:6px;"></div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="margin-bottom:24px;">
        <div style="font-size:16px;font-weight:700;color:{TEXT1};
                    letter-spacing:-0.01em;margin-bottom:2px;">Configuration</div>
        <div style="font-size:11px;color:{TEXT3};">Set parameters before each step</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="ms-eyebrow" style="margin-bottom:8px;">GAN Training</div>', unsafe_allow_html=True)
    epochs = st.slider("Training epochs", 5, 50, 10,
                       help="More epochs → better generated X-rays, longer wait")

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:20px 0 16px;">', unsafe_allow_html=True)
    st.markdown(f'<div class="ms-eyebrow" style="margin-bottom:8px;">Attack Settings</div>', unsafe_allow_html=True)
    epsilon   = st.slider("Strength ε", 0.01, 0.5, 0.1,
                          help="Higher ε = stronger perturbation, more visible noise")
    pgd_steps = st.slider("PGD iterations", 5, 50, 20,
                          help="More iterations = stronger, slower PGD attack")

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:20px 0 16px;">', unsafe_allow_html=True)
    st.markdown(f'<div class="ms-eyebrow" style="margin-bottom:8px;">Defense</div>', unsafe_allow_html=True)
    defense = st.selectbox("Method", ["Gaussian denoising", "Median filter", "Both"])

    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:20px 0 16px;">', unsafe_allow_html=True)
    st.markdown(f'<div class="ms-eyebrow" style="margin-bottom:12px;">Pipeline</div>', unsafe_allow_html=True)

    pipeline = [
        ("generator"      in st.session_state, "01", "GAN trained"),
        ("classifier"     in st.session_state, "02", "Classifier trained"),
        ("adv_img"        in st.session_state, "03", "Attack executed"),
        ("defended"       in st.session_state, "04", "Defense applied"),
    ]
    rows = ""
    for done, num, label in pipeline:
        bg_n  = ACCENT_DK if done else BORDER
        tx_n  = "#fff"    if done else TEXT3
        tx_l  = "#34d399" if done else TEXT2
        rows += (f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
                 f'<div style="width:22px;height:22px;border-radius:6px;background:{bg_n};'
                 f'display:flex;align-items:center;justify-content:center;'
                 f'font-size:9px;font-weight:700;color:{tx_n};flex-shrink:0;">{num}</div>'
                 f'<span style="font-size:13px;color:{tx_l};">{label}</span>'
                 f'</div>')
    st.markdown(rows, unsafe_allow_html=True)

    # Model info card
    st.markdown(f'<hr style="border:none;border-top:1px solid {BORDER};margin:20px 0 16px;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px 16px;">
        <div class="ms-eyebrow" style="margin-bottom:8px;">Device</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:12px;color:{ACCENT};">
            {"⚡ Apple MPS (M4)" if torch.backends.mps.is_available()
             else "🔥 CUDA GPU"   if torch.cuda.is_available()
             else "💻 CPU"}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📊  Train GAN  ",
    "  🫁  Classifier  ",
    "  ⚔️  Attacks  ",
    "  🛡️  Defense  ",
    "  📂  Diagnose  ",
])

# ════════════════════════════════════════════════════════════════════
# TAB 1 — TRAIN GAN
# ════════════════════════════════════════════════════════════════════
with tab1:
    c_info, c_act = st.columns([3, 2], gap="large")

    with c_info:
        st.markdown(f'<div class="ms-tag">Step 01 · Generative Model</div>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color:{TEXT1};font-size:22px;font-weight:700;margin:0 0 16px;letter-spacing:-0.02em;">Train the GAN</h2>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ms-card-accent">
            <div style="font-size:13px;color:{TEXT2};line-height:1.8;">
                A <b style="color:{TEXT1};">Deep Convolutional GAN</b> trained on chest X-ray images.
                Two networks compete: the Generator synthesises fake X-rays from noise,
                while the Discriminator learns to tell them from real scans.
                <br><br>
                <span style="color:{TEXT3};font-size:12px;">
                This trained Discriminator becomes the attack target in Step 03.
                </span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px;">
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px 16px;">
                <div class="ms-eyebrow">Architecture</div>
                <div style="font-size:13px;color:{TEXT1};font-weight:600;">DCGAN</div>
                <div style="font-size:11px;color:{TEXT3};margin-top:2px;">Conv + BatchNorm layers</div>
            </div>
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px 16px;">
                <div class="ms-eyebrow">Input size</div>
                <div style="font-size:13px;color:{TEXT1};font-weight:600;">64 × 64 px</div>
                <div style="font-size:11px;color:{TEXT3};margin-top:2px;">Grayscale chest X-ray</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_act:
        st.markdown(f'<div style="height:28px;"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ms-card" style="padding:20px;">
            <div class="ms-eyebrow">Ready to train</div>
            <div style="font-size:13px;color:{TEXT2};margin:6px 0 16px;line-height:1.6;">
                Epochs set to <b style="color:{TEXT1};">{epochs}</b> in sidebar.
                Training uses your <b style="color:{ACCENT};">
                {"M4 GPU (MPS)" if torch.backends.mps.is_available() else "CPU"}</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀  Start GAN Training", key="train_gan_btn"):
            prog   = st.progress(0)
            status = st.empty()

            def upd(v):
                prog.progress(v)
                status.markdown(
                    f'<div style="font-size:12px;color:{ACCENT};'
                    f'font-family:JetBrains Mono,monospace;">Training · {int(v*100)}%</div>',
                    unsafe_allow_html=True)

            with st.spinner(""):
                gen, disc, g_losses, d_losses = train_gan(
                    progress_callback=upd, epochs_override=epochs)
                st.session_state["generator"]     = gen
                st.session_state["discriminator"] = disc

            status.empty()
            st.success("✅  Training complete — `generator.pth` saved")

    # Loss chart + samples (full width)
    if "generator" in st.session_state and "g_losses" in dir():
        st.markdown(f'<hr class="ms-rule">', unsafe_allow_html=True)
        fc1, fc2 = st.columns([2, 3], gap="large")
        with fc1:
            st.markdown(f'<div class="ms-section-label">Loss Curve</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5, 2.8), facecolor=CHART_PL)
            style_ax(ax)
            ax.plot(g_losses, color=ACCENT,    lw=2, label="Generator",     marker="o", ms=3.5, zorder=3)
            ax.plot(d_losses, color="#f59e0b", lw=2, label="Discriminator", marker="o", ms=3.5, zorder=3)
            ax.set_title("Training loss", color=TEXT2, fontsize=10, pad=8)
            ax.legend(framealpha=0, labelcolor=TEXT2, fontsize=9)
            ax.grid(True, color=BORDER, alpha=0.4, linewidth=0.5)
            plt.tight_layout()
            st.pyplot(fig); plt.close()

    if "generator" in st.session_state:
        gen = st.session_state["generator"]
        st.markdown(f'<div class="ms-section-label" style="margin-top:8px;">Generated X-ray Samples</div>', unsafe_allow_html=True)
        noise = torch.randn(16, NOISE_DIM).to(DEVICE)
        gen.eval()
        with torch.no_grad():
            samples = gen(noise).cpu()
        fig2, axes = plt.subplots(2, 8, figsize=(15, 4.2), facecolor=CHART_BG)
        fig2.subplots_adjust(wspace=0.04, hspace=0.04)
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(tensor_to_display(samples[i]), cmap=XRAY_CM)
            ax.axis("off")
        st.pyplot(fig2); plt.close()

# ════════════════════════════════════════════════════════════════════
# TAB 2 — CLASSIFIER
# ════════════════════════════════════════════════════════════════════
with tab2:
    c_info, c_act = st.columns([3, 2], gap="large")

    with c_info:
        st.markdown(f'<div class="ms-tag">Step 02 · Medical Diagnosis</div>', unsafe_allow_html=True)
        st.markdown(f'<h2 style="color:{TEXT1};font-size:22px;font-weight:700;margin:0 0 16px;letter-spacing:-0.02em;">Pneumonia Classifier</h2>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="ms-card-accent">
            <div style="font-size:13px;color:{TEXT2};line-height:1.8;">
                A <b style="color:{TEXT1};">4-block CNN classifier</b> trained directly on real chest X-rays
                to distinguish healthy lungs from pneumonia cases.
                This is the model that adversarial attacks will attempt to fool in Step 05.
                <br><br>
                <span style="color:{TEXT3};font-size:12px;">
                Classes: <span style="color:#34d399;">NORMAL</span> vs
                <span style="color:#f87171;">PNEUMONIA</span>
                </span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-top:4px;">
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px 16px;">
                <div class="ms-eyebrow">Optimizer</div>
                <div style="font-size:13px;color:{TEXT1};font-weight:600;">Adam</div>
                <div style="font-size:11px;color:{TEXT3};margin-top:2px;">lr = 0.001</div>
            </div>
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px 16px;">
                <div class="ms-eyebrow">Loss</div>
                <div style="font-size:13px;color:{TEXT1};font-weight:600;">BCE</div>
                <div style="font-size:11px;color:{TEXT3};margin-top:2px;">Binary cross-entropy</div>
            </div>
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;padding:14px 16px;">
                <div class="ms-eyebrow">Regularization</div>
                <div style="font-size:13px;color:{TEXT1};font-weight:600;">Dropout</div>
                <div style="font-size:11px;color:{TEXT3};margin-top:2px;">p = 0.5</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c_act:
        st.markdown(f'<div style="height:28px;"></div>', unsafe_allow_html=True)
        clf_epochs = st.slider("Epochs", 5, 30, 10, key="clf_ep")
        if st.button("🫁  Train Classifier", key="train_clf_btn"):
            prog   = st.progress(0)
            status = st.empty()

            def cu(v):
                prog.progress(v)
                status.markdown(
                    f'<div style="font-size:12px;color:{ACCENT};'
                    f'font-family:JetBrains Mono,monospace;">Training · {int(v*100)}%</div>',
                    unsafe_allow_html=True)

            with st.spinner(""):
                clf, tlosses, vaccs = train_classifier(
                    progress_callback=cu, epochs_override=clf_epochs)
                st.session_state["classifier"] = clf
                st.session_state["_tlosses"]   = tlosses
                st.session_state["_vaccs"]     = vaccs

            status.empty()
            st.success("✅  Classifier trained — `classifier.pth` saved")

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Val accuracy",    f"{vaccs[-1]:.1f}%")
            mc2.metric("Training loss",   f"{tlosses[-1]:.4f}")
            mc3.metric("Epochs",          clf_epochs)

    if "_tlosses" in st.session_state:
        tlosses = st.session_state["_tlosses"]
        vaccs   = st.session_state["_vaccs"]
        st.markdown(f'<hr class="ms-rule">', unsafe_allow_html=True)
        st.markdown(f'<div class="ms-section-label">Training Curves</div>', unsafe_allow_html=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), facecolor=CHART_PL)
        for ax in (ax1, ax2):
            style_ax(ax)
            ax.grid(True, color=BORDER, alpha=0.4, linewidth=0.5)
        ax1.plot(tlosses, color=ACCENT,    lw=2, marker="o", ms=3.5)
        ax1.set_title("Training loss",     color=TEXT2, fontsize=10, pad=8)
        ax1.set_xlabel("Epoch")
        ax2.plot(vaccs, color="#34d399",   lw=2, marker="o", ms=3.5)
        ax2.set_title("Validation accuracy (%)", color=TEXT2, fontsize=10, pad=8)
        ax2.set_xlabel("Epoch")
        plt.tight_layout()
        st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════
# TAB 3 — ATTACKS
# ════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="ms-tag">Step 03 · Adversarial Robustness</div>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:{TEXT1};font-size:22px;font-weight:700;margin:0 0 4px;letter-spacing:-0.02em;">Adversarial Attacks</h2>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:24px;">
        <div class="ms-card" style="margin:0;border-left:3px solid #f59e0b;">
            <div class="ms-eyebrow" style="color:#f59e0b;">FGSM</div>
            <div style="font-size:13px;color:{TEXT1};font-weight:600;margin-bottom:4px;">Fast Gradient Sign Method</div>
            <div style="font-size:12px;color:{TEXT2};line-height:1.6;">
                Single-step attack. Computes the gradient of the loss with
                respect to the input and steps in that direction by ε.
                Fast but weaker than iterative methods.
            </div>
        </div>
        <div class="ms-card" style="margin:0;border-left:3px solid #ef4444;">
            <div class="ms-eyebrow" style="color:#ef4444;">PGD</div>
            <div style="font-size:13px;color:{TEXT1};font-weight:600;margin-bottom:4px;">Projected Gradient Descent</div>
            <div style="font-size:12px;color:{TEXT2};line-height:1.6;">
                Iterative attack — essentially repeated FGSM with smaller steps,
                projected back into the ε-ball after each step.
                Considered the strongest first-order attack.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "discriminator" not in st.session_state:
        st.warning("⚠️  Train the GAN in Step 01 first.")
    else:
        a1, a2 = st.columns([1, 2], gap="large")
        with a1:
            attack_type = st.radio("Algorithm", ["FGSM", "PGD", "Both"], index=0)
            st.markdown(f"""
            <div style="background:{BG3};border:1px solid {BORDER};border-radius:10px;
                        padding:12px 16px;margin-top:12px;">
                <div class="ms-eyebrow" style="margin-bottom:6px;">Current settings</div>
                <div class="ms-mono">ε = {epsilon}</div>
                <div class="ms-mono">PGD steps = {pgd_steps}</div>
            </div>
            """, unsafe_allow_html=True)
            run_atk = st.button("⚔️  Run Attack", key="run_attack_btn")

        with a2:
            if run_atk:
                disc = st.session_state["discriminator"]
                gen  = st.session_state["generator"]
                noise    = torch.randn(1, NOISE_DIM).to(DEVICE)
                fake_img = gen(noise).detach()
                label    = torch.zeros(1, 1).to(DEVICE)

                results = {}
                with st.spinner("Crafting adversarial examples…"):
                    if attack_type in ["FGSM", "Both"]:
                        results["FGSM"] = fgsm_attack(disc, fake_img, label, epsilon=epsilon)
                    if attack_type in ["PGD", "Both"]:
                        results["PGD"]  = pgd_attack(disc, fake_img, label,
                                                      epsilon=epsilon, num_steps=pgd_steps)

                disc.eval()
                with torch.no_grad():
                    orig_sc = disc(fake_img.to(DEVICE)).item()

                num_c = 1 + len(results)
                cols  = st.columns(num_c, gap="small")

                with cols[0]:
                    show_xray(cols[0], fake_img, "Original (GAN output)")
                    st.markdown(score_bar(orig_sc, "Discriminator score"), unsafe_allow_html=True)
                    st.markdown(f'<div style="font-size:11px;color:{TEXT3};margin-top:4px;">1.0 = real · 0.0 = fake</div>', unsafe_allow_html=True)

                for i, (name, adv_img) in enumerate(results.items()):
                    with cols[i + 1]:
                        show_xray(cols[i + 1], adv_img, f"{name} adversarial")
                        with torch.no_grad():
                            adv_sc = disc(adv_img).item()
                        delta  = adv_sc - orig_sc
                        d_col  = "#34d399" if delta > 0.05 else "#f87171" if delta < -0.05 else TEXT2
                        st.markdown(score_bar(adv_sc, f"Score after {name}"), unsafe_allow_html=True)
                        st.markdown(
                            f'<div style="font-size:11px;color:{d_col};margin-top:4px;">'
                            f'{"▲" if delta>0 else "▼"} {abs(delta):.3f} shift</div>',
                            unsafe_allow_html=True)

                st.session_state["fake_img"] = fake_img
                st.session_state["adv_img"]  = list(results.values())[-1]
                st.session_state["label"]    = label
            else:
                st.markdown(f"""
                <div style="height:180px;border:1.5px dashed {BORDER2};border-radius:12px;
                            display:flex;align-items:center;justify-content:center;">
                    <div style="text-align:center;color:{TEXT3};">
                        <div style="font-size:28px;margin-bottom:8px;">⚔️</div>
                        <div style="font-size:13px;">Results will appear here</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 4 — DEFENSE
# ════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="ms-tag">Step 04 · Robustness Framework</div>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:{TEXT1};font-size:22px;font-weight:700;margin:0 0 16px;letter-spacing:-0.02em;">Defense Mechanisms</h2>', unsafe_allow_html=True)

    if "adv_img" not in st.session_state:
        st.warning("⚠️  Run an attack in Step 03 first.")
    else:
        d1, d2 = st.columns([1, 2], gap="large")
        with d1:
            st.markdown(f"""
            <div class="ms-card-accent">
                <div class="ms-eyebrow" style="margin-bottom:6px;">Selected method</div>
                <div style="font-size:15px;font-weight:600;color:{TEXT1};margin-bottom:8px;">{defense}</div>
                <div style="font-size:12px;color:{TEXT2};line-height:1.7;">
                    {"Adds Gaussian noise then clips — blurs adversarial high-freq perturbations." if defense == "Gaussian denoising"
                     else "Average pooling smooths the image — effective against pixel-level attacks." if defense == "Median filter"
                     else "Combines both methods in sequence for maximum perturbation removal."}
                </div>
            </div>
            """, unsafe_allow_html=True)
            apply_def = st.button("🛡️  Apply Defense", key="apply_def_btn")

        with d2:
            if apply_def:
                adv_img = st.session_state["adv_img"]
                disc    = st.session_state["discriminator"]

                with st.spinner("Applying defense…"):
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

                ic1, ic2 = st.columns(2)
                show_xray(ic1, adv_img,  "Before defense")
                show_xray(ic2, defended, "After defense")

                st.markdown(f'<hr class="ms-rule" style="margin:16px 0;">', unsafe_allow_html=True)
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Score before", f"{adv_sc:.4f}")
                mc2.metric("Score after",  f"{def_sc:.4f}", delta=f"{def_sc-adv_sc:+.4f}")
                mc3.metric("Recovery",     f"{abs(def_sc-adv_sc):.4f}")

                detection = detect_adversarial(disc, adv_img)
                is_adv    = detection["is_adversarial"]
                bg_det    = "#1a0a00" if is_adv else "#001a0f"
                bdr_det   = "#7f1d1d" if is_adv else "#065f46"
                col_det   = "#f87171" if is_adv else "#34d399"
                icon_det  = "⚠️" if is_adv else "✅"
                lbl_det   = "ADVERSARIAL DETECTED" if is_adv else "IMAGE APPEARS CLEAN"

                st.markdown(f"""
                <div style="background:{bg_det};border:1px solid {bdr_det};
                            border-radius:12px;padding:20px 24px;margin-top:8px;">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                        <span style="font-size:20px;">{icon_det}</span>
                        <span style="font-size:15px;font-weight:700;
                                     color:{col_det};letter-spacing:0.04em;">{lbl_det}</span>
                    </div>
                    <div style="display:flex;gap:24px;flex-wrap:wrap;">
                        <div><div class="ms-eyebrow" style="color:{TEXT3};">Original score</div>
                             <div class="ms-mono" style="color:{TEXT1};">{detection["original_score"]}</div></div>
                        <div><div class="ms-eyebrow" style="color:{TEXT3};">Smoothed score</div>
                             <div class="ms-mono" style="color:{TEXT1};">{detection["smoothed_score"]}</div></div>
                        <div><div class="ms-eyebrow" style="color:{TEXT3};">Conf. drop</div>
                             <div class="ms-mono" style="color:{col_det};">{detection["confidence_drop"]}</div></div>
                        <div><div class="ms-eyebrow" style="color:{TEXT3};">Threshold</div>
                             <div class="ms-mono" style="color:{TEXT1};">{detection["threshold"]}</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.session_state["defended"] = defended
            else:
                st.markdown(f"""
                <div style="height:220px;border:1.5px dashed {BORDER2};border-radius:12px;
                            display:flex;align-items:center;justify-content:center;">
                    <div style="text-align:center;color:{TEXT3};">
                        <div style="font-size:28px;margin-bottom:8px;">🛡️</div>
                        <div style="font-size:13px;">Click "Apply Defense" to see results</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# TAB 5 — DIAGNOSE
# ════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f'<div class="ms-tag">Step 05 · Clinical Demo</div>', unsafe_allow_html=True)
    st.markdown(f'<h2 style="color:{TEXT1};font-size:22px;font-weight:700;margin:0 0 4px;letter-spacing:-0.02em;">Upload & Diagnose</h2>', unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:13px;color:{TEXT3};margin-bottom:20px;">Upload a chest X-ray to run the full adversarial pipeline and see how the diagnosis holds up under attack.</div>', unsafe_allow_html=True)

    missing = []
    if "discriminator" not in st.session_state: missing.append("GAN (Tab 1)")
    if "classifier"    not in st.session_state: missing.append("Classifier (Tab 2)")
    if missing:
        st.warning(f"Complete first: {', '.join(missing)}")

    up1, up2 = st.columns([1, 2], gap="large")
    with up1:
        uploaded = st.file_uploader("Drop a chest X-ray — JPG or PNG",
                                    type=["jpg", "jpeg", "png"])
        if uploaded:
            st.image(uploaded, caption="Uploaded X-ray", use_container_width=True)

    with up2:
        if uploaded and "classifier" in st.session_state:
            img        = Image.open(uploaded).convert("L").resize((64, 64))
            tfm        = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5,), (0.5,))])
            img_tensor = tfm(img).unsqueeze(0)
            label      = torch.zeros(1, 1)

            clf        = st.session_state["classifier"]
            pred, conf = predict_single_image(clf, img_tensor)
            conf_d     = conf if pred == "PNEUMONIA" else 1 - conf
            pill       = "diag-pneumonia" if pred == "PNEUMONIA" else "diag-normal"
            dot        = "🔴" if pred == "PNEUMONIA" else "🟢"

            st.markdown(f"""
            <div class="ms-card-accent">
                <div class="ms-eyebrow">Initial Diagnosis</div>
                <div style="margin:12px 0;">
                    <span class="{pill}">{dot} {pred}</span>
                </div>
                {score_bar(conf_d, "Confidence")}
                <div style="font-size:11px;color:{TEXT3};margin-top:6px;">
                    Tensor shape: {list(img_tensor.shape)}
                </div>
            </div>
            """, unsafe_allow_html=True)

            if not missing:
                if st.button("🚀  Run Full Adversarial Pipeline", key="run_pipeline_btn"):
                    disc = st.session_state["discriminator"]

                    with st.spinner("Running pipeline…"):
                        adv     = fgsm_attack(disc, img_tensor, label, epsilon=epsilon)
                        def_img = gaussian_denoise(adv)

                    st.markdown(f'<hr class="ms-rule">', unsafe_allow_html=True)
                    st.markdown(f'<div class="ms-section-label">Visual Comparison</div>', unsafe_allow_html=True)

                    vc1, vc2, vc3 = st.columns(3, gap="small")
                    show_xray(vc1, img_tensor, "Original")
                    show_xray(vc2, adv,        f"FGSM (ε={epsilon})")
                    show_xray(vc3, def_img,    "Defended")

                    st.markdown(f'<hr class="ms-rule">', unsafe_allow_html=True)
                    st.markdown(f'<div class="ms-section-label">Diagnosis Under Attack</div>', unsafe_allow_html=True)

                    pred_orig, co = predict_single_image(clf, img_tensor)
                    pred_adv,  ca = predict_single_image(clf, adv)
                    pred_def,  cd = predict_single_image(clf, def_img)

                    dc1, dc2, dc3 = st.columns(3, gap="small")
                    dc1.markdown(diag_card_html("Original",      pred_orig, co), unsafe_allow_html=True)
                    dc2.markdown(diag_card_html("After attack",  pred_adv,  ca), unsafe_allow_html=True)
                    dc3.markdown(diag_card_html("After defense", pred_def,  cd), unsafe_allow_html=True)

                    # Impact verdict
                    st.markdown(f'<hr class="ms-rule">', unsafe_allow_html=True)
                    if pred_orig != pred_adv:
                        st.markdown(f"""
                        <div style="background:#200a00;border:1px solid #7c2d12;
                                    border-left:4px solid #ef4444;
                                    border-radius:12px;padding:20px 24px;">
                            <div style="font-size:15px;font-weight:700;color:#f87171;margin-bottom:6px;">
                                ⚠️  Attack successfully changed the diagnosis
                            </div>
                            <div style="font-size:13px;color:{TEXT2};line-height:1.7;">
                                <b style="color:#fbbf24;">{pred_orig}</b> →
                                <b style="color:#f87171;">{pred_adv}</b>
                                — A perturbation invisible to the human eye flipped the clinical outcome.
                                This is precisely why adversarial robustness is non-negotiable
                                in medical AI deployment.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#001a0f;border:1px solid #065f46;
                                    border-left:4px solid #34d399;
                                    border-radius:12px;padding:20px 24px;">
                            <div style="font-size:15px;font-weight:700;color:#34d399;margin-bottom:6px;">
                                ✅  Model held its diagnosis under attack
                            </div>
                            <div style="font-size:13px;color:{TEXT2};line-height:1.7;">
                                Diagnosis remained <b style="color:#34d399;">{pred_orig}</b>
                                despite the adversarial perturbation —
                                demonstrating robust classification behaviour.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Detection report
                    det = detect_adversarial(disc, adv)
                    st.markdown(f'<hr class="ms-rule">', unsafe_allow_html=True)
                    st.markdown(f'<div class="ms-section-label">Detection Report</div>', unsafe_allow_html=True)
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Adversarial",     "YES" if det["is_adversarial"] else "NO")
                    r2.metric("Original score",  det["original_score"])
                    r3.metric("Smoothed score",  det["smoothed_score"])
                    r4.metric("Confidence drop", det["confidence_drop"])

        elif uploaded and "classifier" not in st.session_state:
            st.info("Train the classifier in Tab 2 to see a diagnosis.")

# ── Footer ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer-wrap">
    <div>
        <div style="font-size:10px;font-weight:700;letter-spacing:0.12em;
                    text-transform:uppercase;color:{TEXT3};margin-bottom:8px;">Built by</div>
        <div style="font-size:17px;font-weight:700;color:{TEXT1};
                    letter-spacing:-0.01em;margin-bottom:4px;">Himanshu Ranjan</div>
        <div style="font-size:12px;color:{TEXT3};line-height:1.7;">
            Adversarially Robust Generative AI · Chest X-Ray Analysis<br>
            PyTorch · Streamlit · Apple Silicon M4
        </div>
    </div>
    <div style="display:flex;flex-direction:column;align-items:flex-end;gap:10px;">
        <a class="footer-li-btn"
           href="https://www.linkedin.com/in/himanshuranjan1241"
           target="_blank">
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15"
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
        <div style="font-size:10px;color:{TEXT3};text-align:right;">
            For educational & research use only<br>Not intended for clinical deployment
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
