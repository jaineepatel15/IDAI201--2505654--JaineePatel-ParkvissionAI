"""
ParkVision AI - Intelligent Urban Parking Analytics & Space Optimisation Platform
Streamlit web app: upload a parking lot image, detect slot-level occupancy,
view smart overlays, utilisation analytics, and recommendations.
"""

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ----------------------------------------------------------------------------
# CONFIG - adjust these to match your training setup if needed
# ----------------------------------------------------------------------------
MODEL_PATH = "models/parkvision_model.h5"   # path to your trained model
IMG_SIZE = (224, 224)                       # must match training input size
# Class order must match train_ds.class_names from training (alphabetical by default)
CLASS_NAMES = ["empty", "occupied"]         # index 0 = empty, index 1 = occupied
OCCUPIED_INDEX = CLASS_NAMES.index("occupied")

st.set_page_config(page_title="ParkVision AI", page_icon="🚗", layout="wide")


# ----------------------------------------------------------------------------
# MODEL LOADING (cached so it only loads once per session)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ----------------------------------------------------------------------------
# CORE LOGIC: slot grid detection + classification
# ----------------------------------------------------------------------------
def split_into_slots(image: Image.Image, rows: int, cols: int):
    """Split the parking lot image into a rows x cols grid of slot crops.
    Returns a list of (crop_image, (x1, y1, x2, y2)) tuples."""
    w, h = image.size
    slot_w, slot_h = w // cols, h // rows
    slots = []
    for r in range(rows):
        for c in range(cols):
            x1, y1 = c * slot_w, r * slot_h
            x2 = x1 + slot_w if c < cols - 1 else w
            y2 = y1 + slot_h if r < rows - 1 else h
            crop = image.crop((x1, y1, x2, y2))
            slots.append((crop, (x1, y1, x2, y2)))
    return slots


def classify_slot(model, crop: Image.Image):
    """Run one slot crop through the model. Returns (label, confidence)."""
    img = crop.resize(IMG_SIZE).convert("RGB")
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr, verbose=0)[0]

    # Handle both single-sigmoid-output and softmax-two-output models
    if pred.shape[0] == 1:
        occupied_prob = float(pred[0])
        label = "occupied" if occupied_prob >= 0.5 else "empty"
        confidence = occupied_prob if label == "occupied" else 1 - occupied_prob
    else:
        idx = int(np.argmax(pred))
        label = CLASS_NAMES[idx]
        confidence = float(pred[idx])

    return label, confidence


def draw_overlays(image: Image.Image, results):
    """Draw color-coded bounding boxes: green = empty, red = occupied."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)

    for label, confidence, (x1, y1, x2, y2) in results:
        color = (220, 50, 50) if label == "occupied" else (40, 180, 90)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        tag = f"{label[:3].upper()} {confidence*100:.0f}%"
        draw.rectangle([x1, y1, x1 + 8 * len(tag) + 6, y1 + 18], fill=color)
        draw.text((x1 + 3, y1 + 2), tag, fill="white")

    return annotated


def compute_insights(results):
    """Turn slot predictions into utilisation metrics and recommendations."""
    total = len(results)
    occupied = sum(1 for label, _, _ in results if label == "occupied")
    available = total - occupied
    occupancy_pct = (occupied / total * 100) if total > 0 else 0

    if occupancy_pct < 40:
        congestion = "Low"
    elif occupancy_pct <= 75:
        congestion = "Moderate"
    else:
        congestion = "High"

    if occupancy_pct >= 90:
        recommendation = "🚫 Parking nearly full — try another location."
    elif occupancy_pct >= 75:
        recommendation = "⚠️ Limited spots left — proceed but expect a search."
    else:
        recommendation = "✅ Slots available — proceed to park."

    return {
        "total": total,
        "occupied": occupied,
        "available": available,
        "occupancy_pct": occupancy_pct,
        "congestion": congestion,
        "recommendation": recommendation,
    }


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🚗 ParkVision AI")
st.caption("Intelligent Urban Parking Analytics & Space Optimisation Platform")

with st.sidebar:
    st.header("⚙️ Settings")
    st.write("Adjust the slot grid to roughly match the parking layout in your image.")
    rows = st.slider("Rows of slots", min_value=1, max_value=10, value=4)
    cols = st.slider("Columns of slots", min_value=1, max_value=10, value=6)
    st.divider()
    st.write("**About**")
    st.write(
        "Upload a parking lot image. The model classifies each grid cell as "
        "occupied or empty, then generates real-time availability insights."
    )

uploaded_file = st.file_uploader(
    "Upload a parking lot image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Could not load model from '{MODEL_PATH}'. Error: {e}")
        st.stop()

    with st.spinner("Analyzing parking slots..."):
        slots = split_into_slots(image, rows, cols)
        results = []
        for crop, box in slots:
            label, confidence = classify_slot(model, crop)
            results.append((label, confidence, box))

        annotated_image = draw_overlays(image, results)
        insights = compute_insights(results)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detected Slot Status")
        st.image(annotated_image, use_container_width=True)

    st.subheader("📊 Real-Time Availability Insights")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Slots", insights["total"])
    m2.metric("Occupied", insights["occupied"])
    m3.metric("Available", insights["available"])
    m4.metric("Occupancy", f"{insights['occupancy_pct']:.1f}%")

    congestion_color = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
    st.write(
        f"**Congestion Level:** {congestion_color[insights['congestion']]} "
        f"{insights['congestion']}"
    )
    st.info(insights["recommendation"])

    # Download annotated result
    from io import BytesIO
    buf = BytesIO()
    annotated_image.save(buf, format="PNG")
    st.download_button(
        "⬇️ Download annotated image",
        data=buf.getvalue(),
        file_name="parkvision_result.png",
        mime="image/png",
    )
else:
    st.info("👆 Upload a parking lot image to get started.")