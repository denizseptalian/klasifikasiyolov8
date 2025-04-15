import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter
import base64
from io import BytesIO

st.set_page_config(page_title="Deteksi Sawit", layout="centered")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best2.pt")

def predict_image(model, image):
    image_np = np.array(image.convert("RGB"))
    results = model(image_np)
    return results

def draw_results(image, results):
    image_np = np.array(image.convert("RGB"))
    class_counts = Counter()

    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            label = f"{names[class_id]}: {box.conf[0]:.2f}"

            class_counts[names[class_id]] += 1
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image_np, class_counts

# App title
st.title("📷 Deteksi dan Klasifikasi Kematangan Buah Sawit")

option = st.radio("Pilih metode input gambar:", ("Upload Gambar", "Gunakan Kamera"))

image = None

if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

elif option == "Gunakan Kamera":
    st.markdown("### Kamera Belakang")

    camera_code = """
    <div>
        <video id="video" autoplay playsinline width="100%" style="border:1px solid gray;"></video>
        <button onclick="takePhoto()" style="margin-top:10px;">📸 Ambil Gambar</button>
        <canvas id="canvas" style="display:none;"></canvas>
        <input type="hidden" id="imgData" name="imgData">
    </div>

    <script>
        async function startCamera() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: { exact: "environment" } },
                    audio: false
                });
                const video = document.getElementById('video');
                video.srcObject = stream;
            } catch (err) {
                alert("Gagal mengakses kamera belakang: " + err.message);
            }
        }

        function takePhoto() {
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataURL = canvas.toDataURL('image/png');
            const imgData = document.getElementById('imgData');
            imgData.value = dataURL;

            // Kirim ke Streamlit input
            const streamlitInput = window.parent.document.querySelector('input[data-testid="stTextInput"]');
            streamlitInput.value = dataURL;
            streamlitInput.dispatchEvent(new Event("input", { bubbles: true }));
        }

        window.onload = startCamera;
    </script>
    """
    st.components.v1.html(camera_code, height=520)

    base64_img = st.text_input("📷 Gambar kamera:", label_visibility="collapsed")

    if base64_img:
        try:
            header, encoded = base64_img.split(",", 1)
            decoded = base64.b64decode(encoded)
            image = Image.open(BytesIO(decoded))
            st.image(image, caption="📷 Gambar dari Kamera", use_container_width=True)

            # Prediksi otomatis
            with st.spinner("🔍 Memproses gambar..."):
                model = load_model()
                results = predict_image(model, image)
                img_with_boxes, class_counts = draw_results(image, results)

                st.image(img_with_boxes, caption="📊 Hasil Deteksi", use_container_width=True)

                st.subheader("Jumlah Objek Terdeteksi:")
                for name, count in class_counts.items():
                    st.write(f"- **{name}**: {count}")
        except Exception as e:
            st.error(f"Gagal membaca gambar: {e}")

if image and option == "Upload Gambar":
    if st.button("🔍 Prediksi"):
        with st.spinner("Sedang memproses..."):
            model = load_model()
            results = predict_image(model, image)
            img_with_boxes, class_counts = draw_results(image, results)

            st.image(img_with_boxes, caption="📊 Hasil Deteksi", use_container_width=True)
            st.subheader("Jumlah Objek Terdeteksi:")
            for name, count in class_counts.items():
                st.write(f"- **{name}**: {count}")
