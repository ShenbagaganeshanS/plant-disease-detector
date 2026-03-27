import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import signal

st.set_page_config(page_title="Plant AI", layout="centered")


st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    background: linear-gradient(135deg, #d4fc79, #96e6a1);
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1b5e20;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #2e7d32;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* Floating animation */
@keyframes float1 {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
    100% { transform: translateY(0px); }
}

@keyframes float2 {
    0% { transform: translateX(0px); }
    50% { transform: translateX(8px); }
    100% { transform: translateX(0px); }
}

@keyframes float3 {
    0% { transform: translateY(0px); }
    50% { transform: translateY(8px); }
    100% { transform: translateY(0px); }
}

@keyframes float4 {
    0% { transform: translateX(0px); }
    50% { transform: translateX(-8px); }
    100% { transform: translateX(0px); }
}

/* Slogan style */
.slogan {
    position: fixed;
    padding: 10px 16px;
    border-radius: 15px;
    font-size: 13px;
    font-weight: 600;
    backdrop-filter: blur(10px);
    background: rgba(255, 255, 255, 0.4);
    color: #1b5e20;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    z-index: 9999;
}

/* Positions + animation */
.tl {
    top: 80px;
    left: 20px;
    animation: float1 6s ease-in-out infinite;
}

.tr {
    top: 80px;
    right: 20px;
    animation: float2 7s ease-in-out infinite;
}

.bl {
    bottom: 20px;
    left: 20px;
    animation: float3 6.5s ease-in-out infinite;
}

.br {
    bottom: 20px;
    right: 20px;
    animation: float4 7.5s ease-in-out infinite;
}

</style>
""", unsafe_allow_html=True)
st.markdown('<div class="slogan tl">🌱 Life begins with plants</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan tr">🌍 Grow green, live clean</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan bl">💧 Plants care for you — care for them</div>', unsafe_allow_html=True)
st.markdown('<div class="slogan br">🌿 Small plant, big impact</div>', unsafe_allow_html=True)

st.markdown("""
<style>

/* Stop button styling */
.stop-btn button {
    background-color: #c62828 !important;
    color: white !important;
    border-radius: 25px;
    padding: 10px 25px;
    font-size: 14px;
    border: none;
    transition: 0.3s;
}

.stop-btn button:hover {
    background-color: #b71c1c !important;
    transform: scale(1.05);
}

</style>
""", unsafe_allow_html=True)


model = tf.keras.models.load_model("model_finetuned")


with open("class_names.json", "r") as f:
    class_names = json.load(f)


solutions = {
    "Pepper__bell___Bacterial_spot": "Use copper-based sprays and remove infected leaves.",
    "Pepper__bell___healthy": "Plant is healthy. Maintain proper care.",
    "Potato___Early_blight": "Use fungicide and remove affected leaves.",
    "Potato___Late_blight": "Improve drainage and apply fungicide.",
    "Potato___healthy": "Healthy plant. Continue watering regularly.",
    "Tomato_Bacterial_spot": "Avoid wet leaves and apply bactericides.",
    "Tomato_Early_blight": "Remove infected leaves and use fungicide.",
    "Tomato_Late_blight": "Avoid overhead watering and apply fungicide.",
    "Tomato_Leaf_Mold": "Ensure good ventilation and reduce humidity.",
    "Tomato_Septoria_leaf_spot": "Remove affected leaves and apply fungicide.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use neem oil or insecticidal soap.",
    "Tomato__Target_Spot": "Apply fungicide and remove infected leaves.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Control whiteflies and remove infected plants.",
    "Tomato__Tomato_mosaic_virus": "Remove infected plants immediately.",
    "Tomato_healthy": "Plant is healthy. No action needed."
}


st.markdown('<div class="title">🌿 Plant Disease Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a leaf image and get instant diagnosis</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "png", "jpeg"])


def predict(image):
    image = image.resize((160, 160))
    img = np.array(image) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    predicted_class = np.argmax(pred)

    class_name = class_names[predicted_class]
    readable = class_name.replace("_", " ").replace("__", " - ")

    return class_name, readable


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, width=250)

    class_name, readable = predict(image)

    st.markdown("<br>", unsafe_allow_html=True)

    
    st.markdown(
        f"<div style='text-align:center; font-size:26px; font-weight:bold; color:#2e7d32;'>🌱 {readable}</div>",
        unsafe_allow_html=True
    )

    solution = solutions.get(class_name, "No solution available.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        "<div style='text-align:center; font-size:20px;'>💡 Suggested Solution</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div style='text-align:center; font-size:15px; color:#444;'>{solution}</div>",
        unsafe_allow_html=True
    )


st.markdown("---")

st.markdown(
    """
    <div style='text-align:center;'>
        <img src="https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExb2lxYm5sbzkycXN3OGN5b2w1NWU0b3VtMXphMXV5cTk4eWthNDFiaiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ORZf4AfURr6UY9ymNU/giphy.gif" width="100">
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)


col1, col2, col3 = st.columns([2,1,2])
with col2:
    st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
    
    import os, signal
    if st.button("🛑 Stop App"):
        os.kill(os.getpid(), signal.SIGTERM)
    
    st.markdown('</div>', unsafe_allow_html=True)
st.markdown(
    "<center style='color:gray;'>AI-powered Plant Health Detection 🌿</center>",
    unsafe_allow_html=True
)
