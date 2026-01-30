import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
IMG_SIZE = 224

CLASS_NAMES = {
    0: "Healthy",
    1: "Early caries",
    2: "Advanced caries"
}

MODEL_PATH = "vgg16_tl_clahe_3class.h5"   # make sure this file is in same folder

# ---------------------------------------------------------
# Load model (cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_caries_model():
    model = load_model(MODEL_PATH, compile=False)
    # compile so we can see metrics if needed (not used for inference itself)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ---------------------------------------------------------
# Image preprocessing – must match Colab pipeline
# ---------------------------------------------------------
def preprocess_image(pil_image):
    """
    Takes a PIL.Image and returns a NumPy array ready for the model:
    - convert to grayscale
    - resize to 224x224
    - apply CLAHE
    - slight Gaussian blur
    - normalize to [0, 1]
    - stack to 3 channels
    - add batch dimension
    """
    # Convert PIL -> OpenCV (RGB -> BGR)
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

    # CLAHE (same settings as in Colab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Optional denoising / smoothing
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Normalize to [0, 1]
    gray = gray.astype("float32") / 255.0

    # Expand to (H, W, 1) then repeat to (H, W, 3)
    gray = np.expand_dims(gray, axis=-1)           # (224, 224, 1)
    rgb = np.repeat(gray, 3, axis=-1)             # (224, 224, 3)

    # Add batch dimension: (1, 224, 224, 3)
    rgb = np.expand_dims(rgb, axis=0)

    return rgb


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Dental Caries Detection",
        layout="centered",
        page_icon="🦷"
    )

    # ---------- Header (centered) ----------
    col_h1, col_h2, col_h3 = st.columns([1, 8, 1])
    with col_h2:
        st.title("Dental Caries Detection on Radiographic Images")
        st.write(
            "Upload an intra-oral radiographic image to obtain an automated "
            "prediction of whether the image is **healthy**, shows **early caries**, "
            "or **advanced caries**."
        )

    st.markdown("---")

    # ---------- Main content (also centered) ----------
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        st.subheader("Upload a radiographic image (JPG or PNG)")
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is None:
            st.info("Please upload a radiographic image to begin.")
            return

        # Show image
        pil_image = Image.open(uploaded_file)
        st.image(pil_image, caption="Uploaded Image", use_column_width=True)

        # Analyse button
        analyse = st.button("Analyse Image")

        if analyse:
            with st.spinner("Analysing image…"):
                model = load_caries_model()
                input_tensor = preprocess_image(pil_image)
                probs = model.predict(input_tensor)[0]  # (3,)
                pred_class = int(np.argmax(probs))
                label = CLASS_NAMES[pred_class]

            # ---------- Prediction block ----------
            # Bigger title, text on one line
            st.markdown(f"## Prediction: {label}")

            st.markdown("**Class probabilities:**")
            for i, name in CLASS_NAMES.items():
                st.write(f"{name}: `{probs[i]:.3f}`")
                st.progress(float(probs[i]))



if __name__ == "__main__":
    main()
