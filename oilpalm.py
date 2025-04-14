import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import os
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

option = st.radio("Pilih metode input gambar:", ("Upload Gambar", "Gunakan Kamera (default)"))
image = None

if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

elif option == "Gunakan Kamera (default)":
    st.info("Gunakan kamera depan atau belakang sesuai pengaturan browser.")
    camera_input = st.camera_input("Ambil gambar dari kamera")
    if camera_input:
        image = Image.open(camera_input)
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
