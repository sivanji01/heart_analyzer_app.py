import streamlit as st
import numpy as np
import cv2
import pandas as pd
from skimage import measure, morphology
from PIL import Image

st.set_page_config(page_title="Heart Image Analyzer", layout="wide")
st.title("🫀 Heart Image Analyzer + Comparison")

# Upload section
col1, col2 = st.columns(2)
with col1:
    patient_file = st.file_uploader("📤 Upload Patient Heart Image", type=['jpg', 'jpeg', 'png'], key="patient")
with col2:
    healthy_file = st.file_uploader("📤 Upload Healthy Reference Image", type=['jpg', 'jpeg', 'png'], key="healthy")

def analyze_image(uploaded_file):
    if uploaded_file:
        image = Image.open(uploaded_file).convert('L')
        img = np.array(image)

        # Preprocessing
        img_blur = cv2.medianBlur(img, 5)
        img_eq = cv2.equalizeHist(img_blur)
        
        # Segmentation
        _, binary = cv2.threshold(img_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        binary = morphology.remove_small_objects(binary > 0, 500)

        # Label regions
        labels = measure.label(binary)
        props = measure.regionprops(labels)
        if not props:
            return img, img_eq, binary, 0, 0

        # Get largest region (heart)
        largest = max(props, key=lambda x: x.area)
        outer = largest.coords

        # Erosion for inner region (for wall thickness)
        inner = morphology.erosion(binary, morphology.disk(10))
        inner_labels = measure.label(inner)
        inner_props = measure.regionprops(inner_labels)
        inner_coords = inner_props[0].coords if inner_props else None

        # Wall thickness calculation
        if inner_coords is not None:
            dists = [np.min(np.sqrt(np.sum((inner_coords - pt) ** 2, axis=1))) for pt in outer]
            wall_thickness = np.mean(dists)
        else:
            wall_thickness = 0

        # Valve detection
        valve_count = sum(1 for p in props if p.perimeter > 0 and (4 * np.pi * p.area) / (p.perimeter ** 2) > 0.7 and p.area < 2000)

        return img, img_eq, binary, wall_thickness, valve_count

# If both images are uploaded
if patient_file and healthy_file:
    # Analyze both
    p_img, p_eq, p_mask, p_thick, p_valves = analyze_image(patient_file)
    h_img, h_eq, h_mask, h_thick, h_valves = analyze_image(healthy_file)

    # Display grayscale
    st.subheader("🖼 Patient vs. Healthy Image (Grayscale)")
    st.image([p_img, h_img], caption=["Patient", "Healthy"], width=300)

    # Preprocessed
    st.subheader("🩻 Preprocessed Image")
    st.image([p_eq, h_eq], caption=["Patient", "Healthy"], width=300)

    # Segmented
    st.subheader("🧠 Segmented Mask")
    st.image([p_mask.astype(np.uint8) * 255, h_mask.astype(np.uint8) * 255], caption=["Patient", "Healthy"], width=300)

    # Calculations
    wall_diff = p_thick - h_thick
    wall_pct = (wall_diff / h_thick * 100) if h_thick else 0

    valve_diff = p_valves - h_valves
    valve_pct = (valve_diff / h_valves * 100) if h_valves else 0

    # Comparison
    st.subheader("📊 Comparison Metrics")
    st.markdown(f"""
    **Wall Thickness**  
    - Patient: `{p_thick:.2f} px`  
    - Healthy: `{h_thick:.2f} px`  
    - **Change**: `{wall_diff:.2f} px` ({wall_pct:+.1f}%)
    
    **Valve Count**  
    - Patient: `{p_valves}`  
    - Healthy: `{h_valves}`  
    - **Change**: `{valve_diff}` ({valve_pct:+.1f}%)
    """)

    # Visual chart
    st.subheader("📈 Visual Comparison")
    df = pd.DataFrame({
        'Metric': ['Wall Thickness', 'Valve Count'],
        'Patient': [p_thick, p_valves],
        'Healthy': [h_thick, h_valves]
    })

    st.bar_chart(df.set_index('Metric'))

    # Diagnostic Insight
    st.subheader("🩺 Diagnostic Insight")
    if wall_pct > 20:
        st.warning("⬆️ Wall thickness significantly increased — possible hypertrophy.")
    elif wall_pct < -20:
        st.warning("⬇️ Wall thickness significantly reduced — possible wall thinning.")
    else:
        st.info("✅ Wall thickness within a normal range.")

    if valve_diff != 0:
        st.info(f"🔍 Valve count differs by {valve_diff}. Further evaluation is recommended.")
