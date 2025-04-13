import streamlit as st
# Harus di sini (PALING ATAS)
st.set_page_config(page_title="YOLOv8 Klasifikasi Buah Sawit", layout="centered")

import numpy as np
import cv2
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
import supervision as sv

# Load YOLOv8 model (ganti 'best.pt' dengan path model kamu)
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Fungsi anotasi menggunakan supervision
def annotate_image(image: np.ndarray, results) -> tuple:
    detections = sv.Detections.from_ultralytics(results[0])

    class_names = results[0].names
    class_colors = {
        "Masak": sv.Color.ORANGE,
        "Mengkal": sv.Color.YELLOW,
        "Mentah": sv.Color.BLACK
    }

    color_lookup = [
        class_colors.get(class_names[class_id], sv.Color.WHITE)
        for class_id in detections.class_id
    ]

    box_annotator = sv.BoxAnnotator(
        thickness=4,
        text_thickness=2,
        text_scale=1.1,
        text_padding=5,
        text_color=sv.Color.WHITE,
        text_background=sv.Color.BLACK
    )

    labels = [class_names[class_id] for class_id in detections.class_id]

    annotated_img = box_annotator.annotate(
        scene=image.copy(),
        detections=detections,
        labels=labels,
        color=color_lookup
    )

    return annotated_img, detections


# Streamlit UI
st.title("📸 Klasifikasi Buah Sawit Menggunakan YOLOv8")
st.markdown("Upload gambar buah sawit dan deteksi tingkat kematangannya secara otomatis.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    st.image(image, caption="Gambar Asli", use_column_width=True)

    with st.spinner("🚀 Mendeteksi..."):
        # Simpan sementara gambar
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_path = tmp_file.name
            image.save(tmp_path)

        # Prediksi menggunakan YOLOv8
        results = model(tmp_path)

        # Hapus file sementara
        os.remove(tmp_path)

        # Anotasi
        annotated_img, detections = annotate_image(image_np, results)

        st.image(annotated_img, caption="Hasil Deteksi", use_column_width=True)

        # Hitung jumlah buah berdasarkan kelas
        class_names = results[0].names
        count_dict = {}
        for class_id in detections.class_id:
            label = class_names[class_id]
            count_dict[label] = count_dict.get(label, 0) + 1

        st.subheader("📊 Jumlah Buah per Kategori:")
        for k, v in count_dict.items():
            st.markdown(f"- **{k}**: {v} buah")
