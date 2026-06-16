import streamlit as st
import os

st.set_page_config(page_title="MedShield AI", page_icon="🩺", layout="wide")

st.title("🩺 MedShield AI")
st.success("App is running on Streamlit Cloud!")
st.write(f"Python path: {os.sys.version}")

import torch
st.write(f"PyTorch version: {torch.__version__}")
st.write(f"Device available: {'MPS' if torch.backends.mps.is_available() else 'CUDA' if torch.cuda.is_available() else 'CPU'}")
