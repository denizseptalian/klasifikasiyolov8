import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Klasifikasi Buah Sawit", layout="centered")

@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

def predict_image(model, image):
    image_array = np.array(image.convert("RGB"))
    results = model.predict(source=image_array, conf=0.25)
    return results

def draw_results(image, results):
    img = np.array(image.convert("RGB"))
    class_counts = Counter()

    class_colors = {
        "Masak": (0, 165, 255),     # Orange
        "Mengkal": (0, 255, 255),   # Yellow
        "Mentah": (0, 0, 0),        # Black
    }

    for result in results:
        boxes = result.boxes
        names = result.names

        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            class_name = names[class_id]
            label = f"{class_name}: {conf:.2f}"
            class_counts[class_name] += 1
            color = class_colors.get(class_name, (0, 255, 0))

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return img, class_counts

# UI
st.title("🌴 Deteksi & Klasifikasi Buah Sawit")
st.markdown("Unggah gambar atau gunakan kamera untuk mendeteksi buah kelapa sawit berdasarkan tingkat kematangan.")
st.image("Buah-Kelapa-Sawit.jpg", use_container_width=True)

model = load_model()

tab1, tab2 = st.tabs(["📁 Upload Gambar", "📷 Buka Kamera"])

with tab1:
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

        if st.button("🔍 Predict dari Upload"):
            results = predict_image(model, image)
            processed_image, class_counts = draw_results(image, results)

            st.image(processed_image, caption="Hasil Deteksi", use_container_width=True)
            st.subheader("📊 Jumlah Kelas Terdeteksi:")
            for cls, count in class_counts.items():
                st.write(f"- {cls}: {count}")

with tab2:
    camera_image = st.camera_input("Ambil Gambar dari Kamera")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Gambar dari Kamera", use_container_width=True)

        if st.button("🔍 Predict dari Kamera"):
            results = predict_image(model, image)
            processed_image, class_counts = draw_results(image, results)

            st.image(processed_image, caption="Hasil Deteksi", use_container_width=True)
            st.subheader("📊 Jumlah Kelas Terdeteksi:")
            for cls, count in class_counts.items():
                st.write(f"- {cls}: {count}")
