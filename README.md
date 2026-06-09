# 🛡️ MedShield AI — Adversarially Robust Generative AI

> A research project that trains a GAN on chest X-ray images, attacks it with adversarial techniques, defends against those attacks, and classifies X-rays as **NORMAL** or **PNEUMONIA** — all through a clean interactive interface.

---

## 📌 What This Project Does

In simple terms:

1. **Trains a GAN** — an AI that learns to generate fake chest X-ray images
2. **Trains a Classifier** — a separate AI that looks at real X-rays and says *"this is Normal"* or *"this has Pneumonia"*
3. **Attacks the model** — adds tiny invisible noise to fool the AI into making wrong decisions
4. **Defends against attacks** — cleans the noisy image before the AI sees it
5. **Shows you everything** — through a professional interactive web interface built with Streamlit

This project answers a real and important question:
> *"Can a medical AI be tricked by invisible noise — and how do we stop that?"*

---

## 🖼️ Screenshots

> Run the project and explore the interface across all 5 tabs.

| Tab | What you see |
|-----|-------------|
| 📊 Train GAN | Loss curve + generated X-ray samples |
| 🫁 Classifier | Training accuracy curve + val accuracy |
| ⚔️ Attacks | Original vs adversarial image + score shift |
| 🛡️ Defense | Before/after defense + detection result |
| 📂 Diagnose | NORMAL/PNEUMONIA diagnosis + PDF download |

---

## 🧠 How It Works — The Simple Explanation

### What is a GAN?
A **GAN (Generative Adversarial Network)** has two parts:
- The **Generator** — tries to create fake X-rays that look real
- The **Discriminator** — tries to catch the fakes

They compete with each other until the Generator gets so good that even the Discriminator can't tell the difference.

### What is an Adversarial Attack?
Imagine someone adds a tiny amount of invisible static to an image. You cannot see it. But the AI sees something completely different and makes the wrong diagnosis. That is an adversarial attack.

- **FGSM (Fast Gradient Sign Method)** — one big step of noise in the worst possible direction
- **PGD (Projected Gradient Descent)** — many small steps, much stronger attack

### What is the Defense?
Before the image reaches the AI, we clean it:
- **Gaussian Denoising** — smooths out the invisible noise
- **Median Filter** — average-pools the image to remove sharp perturbations
- **Both** — applies both methods one after the other

### What is the Classifier?
A separate **CNN (Convolutional Neural Network)** trained only on real chest X-rays. It learns the difference between healthy lungs and lungs with pneumonia.

---

## 🗂️ Project Structure

```
adversarial_ai_project/
│
├── app.py                    ← Main Streamlit interface (run this)
│
├── model/
│   ├── __init__.py
│   ├── gan.py                ← GAN architecture (Generator + Discriminator)
│   ├── train.py              ← GAN training loop
│   ├── classifier.py         ← CNN classifier architecture
│   └── train_classifier.py  ← Classifier training loop
│
├── attacks/
│   ├── __init__.py
│   ├── fgsm.py               ← Fast Gradient Sign Method attack
│   └── pgd.py                ← Projected Gradient Descent attack
│
├── defense/
│   ├── __init__.py
│   └── defend.py             ← Denoising + adversarial detection
│
├── utils/
│   ├── __init__.py
│   └── data_loader.py        ← Loads the chest X-ray dataset
│
└── data/
    └── chest_xray/           ← Dataset goes here (not uploaded to GitHub)
        ├── train/
        │   ├── NORMAL/
        │   └── PNEUMONIA/
        ├── val/
        └── test/
```

---

## 🛠️ Tech Stack

| Technology | What it is used for |
|------------|---------------------|
| **Python 3.9** | Main programming language |
| **PyTorch** | Building and training all AI models |
| **Streamlit** | Building the interactive web interface |
| **Matplotlib** | Plotting loss curves and training charts |
| **ReportLab** | Generating the downloadable PDF report |
| **Pillow (PIL)** | Loading and resizing uploaded images |
| **Torchvision** | Image transforms and dataset utilities |
| **Apple MPS (M4)** | GPU acceleration on Apple Silicon Mac |

---

## 📦 Installation — Step by Step

### Step 1 — Make sure you have Python 3.9
```bash
python3 --version
```

### Step 2 — Clone the repository
```bash
git clone https://github.com/yourusername/adversarial-robust-generative-ai.git
cd adversarial-robust-generative-ai
```

### Step 3 — Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal line.

### Step 4 — Install all required packages
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install streamlit matplotlib numpy pillow scikit-learn
pip install torchattacks grad-cam reportlab
```

### Step 5 — Download the dataset
Go to this link and download the chest X-ray dataset:

👉 [Chest X-Ray Images (Pneumonia) — Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

After downloading, unzip and place the folder so your structure looks like:
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    └── test/
```

### Step 6 — Run the app
```bash
streamlit run app.py
```

Your browser will automatically open at `http://localhost:8501`

---

## 🚀 How to Use the App

Follow the tabs in order — each step builds on the previous one.

### Tab 1 — Train GAN 📊
- Set the number of epochs using the slider in the sidebar (start with **10**)
- Click **Start GAN Training**
- Watch the progress bar complete
- See the loss curve and generated fake X-ray samples

### Tab 2 — Train Classifier 🫁
- Set the number of epochs (start with **10–15**)
- Click **Train Classifier**
- Watch the validation accuracy improve
- Aim for **75% or higher** for reliable diagnosis

### Tab 3 — Adversarial Attacks ⚔️
- Choose your attack type: **FGSM**, **PGD**, or **Both**
- Adjust attack strength (ε) in the sidebar
- Click **Run Attack**
- See how the model's confidence score changes after the attack

### Tab 4 — Defense 🛡️
- Choose your defense method in the sidebar
- Click **Apply Defense**
- See the cleaned image and whether the attack was detected

### Tab 5 — Upload & Diagnose 📂
- Upload any chest X-ray image (JPG or PNG)
- The app immediately shows **NORMAL** or **PNEUMONIA**
- Click **Run Full Adversarial Pipeline** to see if the attack changes the diagnosis
- Download the full **PDF report** of all results

---

## 🔑 Key Concepts Explained Simply

| Term | Simple meaning |
|------|---------------|
| **GAN** | Two AIs competing — one fakes, one detects |
| **Generator** | The AI that creates fake X-ray images |
| **Discriminator** | The AI that decides if an image is real or fake |
| **Adversarial attack** | Invisible noise added to trick the AI |
| **FGSM** | One-step attack — fast and simple |
| **PGD** | Multi-step attack — stronger and harder to defend |
| **Epsilon (ε)** | How strong the attack is (0.01 = weak, 0.5 = strong) |
| **Defense** | Cleaning the image before the AI sees it |
| **CNN** | A type of AI designed specifically to understand images |
| **MPS** | Apple Silicon GPU — makes training faster on M1/M2/M3/M4 Macs |

---

## ⚠️ Important Notes

- This project is **for educational and research purposes only**
- The classifier is **not suitable for real medical diagnosis**
- Training accuracy depends heavily on how many epochs you run
- If your classifier accuracy is below 70%, run more epochs before testing

---

## 👤 Author

**Himanshu Ranjan**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/himanshuranjan1241)

---

## 📜 License

This project is open source and available for educational use.

---

<div align="center">
    <sub>Built with PyTorch · Streamlit · Apple Silicon M4</sub>
</div>
