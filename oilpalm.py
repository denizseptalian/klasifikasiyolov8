import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import os
import base64
from io import BytesIO
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi Buah Sawit", layout="centered")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

def predict_image(model, image):
    image = np.array(image.convert("RGB"))
    results = model(image)
    return results

def draw_results(image, results):
    img = np.array(image.convert("RGB"))
    class_counts = Counter()

    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            label = f"{names[class_id]}: {box.conf[0]:.2f}"
            class_counts[names[class_id]] += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img, class_counts

st.title("📷 Deteksi dan Klasifikasi Kematangan Buah Sawit")

if os.path.exists("Buah-Kelapa-Sawit.jpg"):
    st.image("Buah-Kelapa-Sawit.jpg", use_container_width=True)

option = st.radio("Pilih metode input gambar:", ("Upload Gambar", "Gunakan Kamera (Flip)"))
image = None

if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

elif option == "Gunakan Kamera (Flip)":
    st.markdown("### Kamera Flip (Depan ↔ Belakang)")

    # Kontrol tombol flip
    flip = st.button("🔄 Flip Kamera")
    capture = st.button("📸 Ambil Gambar")

    # Simpan status flip kamera
    if "flip_state" not in st.session_state:
        st.session_state.flip_state = True

    if flip:
        st.session_state.flip_state = not st.session_state.flip_state

    facing_mode = "environment" if st.session_state.flip_state else "user"

    html_code = f"""
    <video id="video" autoplay playsinline style="width: 100%; border:1px solid gray;"></video><br>
    <canvas id="canvas" style="display: none;"></canvas>
    <script>
    let stream;

    async function startCamera() {{
        if (stream) {{
            stream.getTracks().forEach(track => track.stop());
        }}
        const constraints = {{
            video: {{
                facingMode: {{ exact: "{facing_mode}" }}
            }}
        }};
        try {{
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            document.getElementById('video').srcObject = stream;
        }} catch (err) {{
            alert("Tidak dapat mengakses kamera: " + err.message);
        }}
    }}

    function takePhoto() {{
        const canvas = document.getElementById('canvas');
        const video = document.getElementById('video');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);
        const dataUrl = canvas.toDataURL('image/png');
        const input = window.parent.document.querySelector("input#camera_image_input");
        input.value = dataUrl;
        input.dispatchEvent(new Event("input", {{ bubbles: true }}));
    }}

    window.onload = startCamera;
    {'takePhoto();' if capture else ''}
    </script>
    """

    st.components.v1.html(html_code, height=500)

    base64_img = st.text_input("📷 Hasil Kamera", key="camera_image_input", label_visibility="collapsed")

    if base64_img:
        header, encoded = base64_img.split(",", 1)
        decoded = base64.b64decode(encoded)
        image = Image.open(BytesIO(decoded))
        st.image(image, caption="Gambar dari Kamera", use_container_width=True)

# Prediksi
if image and st.button("Prediksi"):
    with st.spinner("Sedang memproses prediksi..."):
        model = load_model()
        results = predict_image(model, image)
        processed_image, class_counts = draw_results(image, results)

        st.image(processed_image, caption="Hasil Deteksi", use_container_width=True)

        st.subheader("Jumlah Objek Terdeteksi:")
        for class_name, count in class_counts.items():
            st.write(f"- **{class_name}**: {count}")
