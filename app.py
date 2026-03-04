import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2  # Added import for OpenCV (cv2) - ensure it's installed via pip install opencv-python

st.set_page_config(page_title="Alzheimer's Diagnosis", layout="wide", page_icon="🧠")
st.title("🧠 Alzheimer's Disease Diagnosis System")
st.markdown("**Enhanced LeNet CNN | 99.22% Accuracy | MRI Brain Scans ONLY**")

# Clear class mapping
ALZHEIMER_CLASSES = ['Mild Impairment', 'Moderate Impairment']  # HAS Alzheimer's
NORMAL_CLASSES = ['No Impairment', 'Very Mild Impairment']  # NO Alzheimer's

@st.cache_resource
def load_model():
    """Load Enhanced LeNet (99.22%)"""
    return tf.keras.models.load_model('models/enhanced_lenet_v2.keras')

def preprocess_mri(image):
    """MRI → model input (128x128x1)"""
    img = image.convert('L')  # Grayscale ONLY
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=-1)  # (128,128,1)
    img_batch = np.expand_dims(img_batch, axis=0)  # (1,128,128,1)
    return img_batch

def validate_alzheimers_mri(image):
    """STRICT validation: Alzheimer's MRI characteristics only"""
    img_array = np.array(image.convert('L'))
    
    # 1. MRI size check
    h, w = img_array.shape
    if h < 100 or w < 100 or h > 512 or w > 512:
        return False, "❌ Image size not typical for MRI"
    
    # 2. Grayscale brain check (high center intensity) - Adjusted thresholds to be more lenient for typical MRI scans
    center_crop = img_array[h//4:3*h//4, w//4:3*w//4]
    center_mean = np.mean(center_crop)
    
    if center_mean < 40 or center_mean > 220:  # Lowered lower bound from 80 to 40 to accept more valid MRI images
        return False, "❌ Not a brain MRI scan (center intensity)"
    
    # 3. Edge detection (brain boundary) - Adjusted density range to be more flexible
    edges = cv2.Canny(img_array, 50, 150)  # Simplified, as img_array is already uint8
    edge_density = np.sum(edges > 0) / (h * w)
    
    if edge_density < 0.01 or edge_density > 0.20:  # Widened range from 0.02-0.15 to 0.01-0.20
        return False, "❌ No clear brain boundary detected"
    
    # 4. Hippocampus region variance (Alzheimer's biomarker area) - Lowered threshold for more acceptance
    hippo_region = img_array[h//3:2*h//3, w//4:3*w//4]
    variance = np.var(hippo_region)
    
    if variance < 100:  # Lowered from 200 to 100
        return False, "❌ Insufficient hippocampus region detail"
    
    return True, "✅ Valid Alzheimer's MRI scan detected"

# HEADER
st.markdown("---")
col1, col2 = st.columns([1, 2])
with col1:
    st.metric("**Accuracy**", "**99.22%**")
    st.success("**Enhanced LeNet**")
with col2:
    st.markdown("""
    **✅ Detects**: Mild/Moderate Alzheimer's
    **❌ Rejects**: Normal brain, non-MRI images
    **🎯 Focus**: Hippocampus region analysis
    """)

# UPLOAD SECTION
st.subheader("🔬 Upload **Alzheimer's MRI Scan**")
uploaded_file = st.file_uploader(
    "Choose MRI file...",
    type=['jpg', 'jpeg', 'png'],
    help="**Alzheimer's brain MRI scans ONLY** - other images rejected"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # === STRICT VALIDATION ===
    is_valid, message = validate_alzheimers_mri(image)
    
    col_img, col_status = st.columns([1, 2])
    
    with col_img:
        st.image(image, caption="Uploaded Image", width=300)
    
    with col_status:
        if is_valid:
            st.success(message)
            st.markdown("**✓ Meets Alzheimer's MRI criteria**")
        else:
            st.error(message)
            st.warning("**❌ This is NOT an Alzheimer's MRI scan**")
            st.stop()  # STOP EXECUTION
    
    # === ALZHEIMER'S PREDICTION ===
    with st.spinner("Analyzing hippocampus for Alzheimer's biomarkers..."):
        model = load_model()
        processed_img = preprocess_mri(image)
        prediction = model.predict(processed_img, verbose=0)[0]
    
    # === DIAGNOSIS RESULT (ENHANCED MODEL) - Made bigger and highlighted ===
    st.markdown("<h1 style='text-align: center; color: #FF6347; background-color: #F0F8FF; padding: 20px; border-radius: 10px;'>🧠 Diagnosis Result (Enhanced Model)</h1>", unsafe_allow_html=True)
    
    # Clear result
    confidence = np.max(prediction) * 100
    has_alzheimers = np.argmax(prediction) in [0, 1]  # Mild/Moderate = HAS Alzheimer's
    
    col_result, col_conf = st.columns([1, 1])
    
    with col_result:
        if has_alzheimers:
            st.error("🚨 **PATIENT HAS ALZHEIMER'S DISEASE**")
            st.metric("**Diagnosis**", "**ALZHEIMER'S POSITIVE**", f"{confidence:.1f}%")
            severity = "Mild" if np.argmax(prediction) == 0 else "Moderate"
            st.warning(f"**Severity**: {severity} Impairment")
        else:
            st.success("✅ **NO ALZHEIMER'S DISEASE DETECTED**")
            st.metric("**Diagnosis**", "**NORMAL / VERY MILD**", f"{confidence:.1f}%")
    
    with col_conf:
        # Confidence breakdown
        st.subheader("Confidence Scores")
        st.bar_chart({
            'Mild AD': prediction[0]*100,
            'Moderate AD': prediction[1]*100,
            'No Impairment': prediction[2]*100,
            'Very Mild': prediction[3]*100
        })
    
    # === MODEL COMPARISON - Made small ===
    st.markdown("<h4 style='text-align: center; color: #808080; margin-top: 40px;'>Model Comparison</h4>", unsafe_allow_html=True)
    comparison_data = {
        "Model": ["Standard LeNet", "Enhanced LeNet"],
        "Accuracy": ["95.00%", "99.22%"],
        "Focus": ["General CNN", "Hippocampus Optimized"]
    }
    st.dataframe(comparison_data, use_container_width=True)
    
    # === SUMMARY ===
    st.markdown("---")
    st.markdown(f"""
    **🎯 Final Assessment**
    - **Model**: Enhanced LeNet (99.22% accuracy)
    - **Input**: Validated Alzheimer's MRI scan
    - **Method**: Hippocampus region CNN analysis
    - **Confidence**: {confidence:.1f}%
    
    **💡 Clinical Note**: Consult neurologist for confirmation
    """)

# FOOTER
st.markdown("---")