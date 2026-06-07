import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model.gan      import Generator, Discriminator
from model.train    import train_gan, NOISE_DIM, DEVICE
from attacks.fgsm   import fgsm_attack
from attacks.pgd    import pgd_attack
from defense.defend import gaussian_denoise, median_filter_defense, detect_adversarial
from xai.gradcam    import compute_saliency_map, plot_comparison

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Adversarially Robust Generative AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Adversarially Robust Generative AI")
st.markdown("""
This tool trains a GAN on image data, attacks it with adversarial examples,
defends against those attacks, and explains model decisions using XAI.
""")

# ─────────────────────────────────────────────
# Sidebar – controls
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
epochs    = st.sidebar.slider("Training epochs",  5, 50, 10)
epsilon   = st.sidebar.slider("Attack strength (ε)", 0.01, 0.5, 0.1)
pgd_steps = st.sidebar.slider("PGD steps", 5, 50, 20)
defense   = st.sidebar.selectbox("Defense method", ["Gaussian denoising", "Median filter", "Both"])

# ─────────────────────────────────────────────
# Tab layout
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Train GAN",
    "⚔️ Adversarial Attacks",
    "🛡️ Defense",
    "🔍 Explainability (XAI)",
    "📁 Upload Your Image"
])

# ─────────────────────────── TAB 1: TRAIN ──────────────────────────────
with tab1:
    st.header("Step 1: Train the GAN")
    st.info("The GAN learns to generate realistic images. The Generator creates fakes; the Discriminator tries to catch them.")

    if st.button("🚀 Start Training"):
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def update_progress(val):
            progress_bar.progress(val)
            status_text.text(f"Training: {int(val*100)}% complete")

        with st.spinner("Training in progress..."):
            gen, disc, g_losses, d_losses = train_gan(
                progress_callback=update_progress,
                epochs_override=epochs
            )
            st.session_state['generator']     = gen
            st.session_state['discriminator'] = disc

        st.success("✅ Training complete! Model saved.")

        # Plot losses
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(g_losses, label="Generator loss",     color="purple")
        ax.plot(d_losses, label="Discriminator loss", color="teal")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.set_title("Training loss curve")
        st.pyplot(fig)

        # Show generated samples
        st.subheader("Generated images (samples from trained Generator)")
        noise    = torch.randn(16, NOISE_DIM).to(DEVICE)
        gen.eval()
        with torch.no_grad():
            samples = gen(noise).cpu()

        fig2, axes = plt.subplots(2, 8, figsize=(14, 4))
        for i, ax in enumerate(axes.flatten()):
            ax.imshow(samples[i].squeeze(), cmap="gray")
            ax.axis("off")
        st.pyplot(fig2)

# ─────────────────────── TAB 2: ATTACKS ───────────────────────────────
with tab2:
    st.header("Step 2: Adversarial Attacks")
    st.warning("Attacks craft tiny invisible noise that fools the discriminator into thinking fake images are real.")

    if 'discriminator' not in st.session_state:
        st.info("Please train the model first (Tab 1).")
    else:
        disc = st.session_state['discriminator']
        gen  = st.session_state['generator']

        # Generate a fake image to attack
        noise    = torch.randn(1, NOISE_DIM).to(DEVICE)
        fake_img = gen(noise).detach()
        label    = torch.zeros(1, 1).to(DEVICE)  # label: fake

        attack_type = st.radio("Choose attack:", ["FGSM", "PGD", "Both"])

        if st.button("⚔️ Run Attack"):
            results = {}
            if attack_type in ["FGSM", "Both"]:
                results["FGSM"] = fgsm_attack(disc, fake_img, label, epsilon=epsilon)
            if attack_type in ["PGD", "Both"]:
                results["PGD"]  = pgd_attack(disc, fake_img, label, epsilon=epsilon, num_steps=pgd_steps)

            cols = st.columns(1 + len(results))
            with cols[0]:
                st.image(
                    ((fake_img.squeeze().detach().cpu().numpy() + 1) / 2),
                    caption="Original fake image", clamp=True
                )

            for i, (name, adv_img) in enumerate(results.items()):
                disc.eval()
                with torch.no_grad():
                    orig_score = disc(fake_img).item()
                    adv_score  = disc(adv_img).item()

                with cols[i + 1]:
                    st.image(
                        ((adv_img.squeeze().detach().cpu().numpy() + 1) / 2),
                        caption=f"{name} adversarial image", clamp=True
                    )
                    st.metric("Discriminator score (original)", f"{orig_score:.3f}")
                    st.metric("Discriminator score (adversarial)", f"{adv_score:.3f}",
                              delta=f"{adv_score - orig_score:+.3f}")

            st.session_state['fake_img'] = fake_img
            st.session_state['adv_img']  = list(results.values())[-1]
            st.session_state['label']    = label

# ─────────────────────── TAB 3: DEFENSE ──────────────────────────────
with tab3:
    st.header("Step 3: Defense Mechanisms")
    st.success("Defense tries to clean adversarial noise before the model sees the image.")

    if 'adv_img' not in st.session_state:
        st.info("Please run an attack first (Tab 2).")
    else:
        adv_img = st.session_state['adv_img']
        disc    = st.session_state['discriminator']

        if st.button("🛡️ Apply Defense"):
            if defense == "Gaussian denoising":
                defended = gaussian_denoise(adv_img)
            elif defense == "Median filter":
                defended = median_filter_defense(adv_img)
            else:
                defended = median_filter_defense(gaussian_denoise(adv_img))

            disc.eval()
            with torch.no_grad():
                adv_score  = disc(adv_img).item()
                def_score  = disc(defended).item()

            col1, col2 = st.columns(2)
            col1.image(
                ((adv_img.squeeze().detach().cpu().numpy() + 1) / 2),
                caption="Adversarial image"
            )
            col2.image(
                ((defended.squeeze().detach().cpu().numpy() + 1) / 2),
                caption="After defense"
            )

            st.metric("Score before defense", f"{adv_score:.4f}")
            st.metric("Score after defense",  f"{def_score:.4f}",
                      delta=f"{def_score - adv_score:+.4f}")

            detection = detect_adversarial(disc, adv_img)
            st.subheader("🔎 Adversarial Detection Result")
            st.json(detection)
            if detection["is_adversarial"]:
                st.error("⚠️ This image was detected as adversarial!")
            else:
                st.success("✅ Image appears clean.")

            st.session_state['defended'] = defended

# ─────────────────────── TAB 4: XAI ──────────────────────────────────
with tab4:
    st.header("Step 4: Explainability (XAI)")
    st.info("Saliency maps show which pixels the discriminator focuses on when judging an image.")

    if 'defended' not in st.session_state:
        st.info("Please complete Tab 3 first.")
    else:
        if st.button("🔍 Generate Saliency Maps"):
            fake_img = st.session_state['fake_img']
            adv_img  = st.session_state['adv_img']
            defended = st.session_state['defended']
            label    = st.session_state['label']
            disc     = st.session_state['discriminator']

            sal_orig = compute_saliency_map(disc, fake_img, label)
            sal_adv  = compute_saliency_map(disc, adv_img,  label)

            fig = plot_comparison(fake_img, adv_img, defended, sal_orig, sal_adv)
            st.pyplot(fig)

            st.markdown("""
            **How to read the saliency maps:**
            - 🔴 **Bright/hot areas** = pixels the model cares about most  
            - ⚫ **Dark areas** = pixels the model largely ignores  
            - Notice how adversarial attacks shift the model's attention to noisy/irrelevant regions
            """)

# ─────────────────────── TAB 5: UPLOAD ───────────────────────────────
with tab5:
    st.header("Step 5: Upload Your Own Image")
    st.info("Upload any grayscale image (medical scan, security footage frame, etc.) to test the pipeline.")

    uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded and 'discriminator' in st.session_state:
        from PIL import Image
        
        # Fix: resize to 64x64 to match pneumonia dataset GAN input size
        img = Image.open(uploaded).convert("L").resize((64, 64))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        img_tensor = transform(img).unsqueeze(0)  # shape: [1, 1, 64, 64]
        label      = torch.zeros(1, 1)

        col1, col2 = st.columns(2)
        col1.image(uploaded, caption="Uploaded image", width=200)

        # Show tensor shape for debugging (you can remove this later)
        st.caption(f"Image tensor shape: {img_tensor.shape}")

        if st.button("🚀 Run full pipeline on this image"):
            disc = st.session_state['discriminator']
            adv  = fgsm_attack(disc, img_tensor, label, epsilon=epsilon)
            def_ = gaussian_denoise(adv)
            sal  = compute_saliency_map(disc, img_tensor, label)
            sal_adv = compute_saliency_map(disc, adv, label)

            fig = plot_comparison(img_tensor, adv, def_, sal, sal_adv)
            st.pyplot(fig)
            det = detect_adversarial(disc, adv)
            st.json(det)