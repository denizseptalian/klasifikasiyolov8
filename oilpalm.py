import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import base64
from io import BytesIO
from ultralytics import YOLO

st.set_page_config(page_title="Deteksi Buah Sawit", layout="centered")

# Load model hanya sekali
@st.cache_resource
def load_model():
    return YOLO("best2.pt")  # Ganti dengan path modelmu

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
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img, class_counts


# Inisialisasi session_state
if "camera_image" not in st.session_state:
    st.session_state["camera_image"] = ""

# Judul Aplikasi
st.title("📷 Deteksi dan Klasifikasi Kematangan Buah Sawit")
st.markdown("Pilih metode input gambar:")

# Pilih metode input
option = st.radio("", ["Upload Gambar", "Gunakan Kamera"])

image = None

# Upload gambar manual
if option == "Upload Gambar":
    uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

# Kamera langsung (belakang)
elif option == "Gunakan Kamera":
    st.markdown("### Kamera Belakang (Environment)")

    camera_code = """
    <div>
        <video id="video" autoplay playsinline width="100%" style="border:1px solid gray;"></video>
        <button onclick="takePhoto()" style="margin-top:10px;">📸 Ambil Gambar</button>
        <canvas id="canvas" style="display:none;"></canvas>
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

            // Kirim data ke Streamlit
            const inputBox = window.parent.document.querySelector('textarea[data-testid="stTextArea"]');
            inputBox.value = dataURL;
            inputBox.dispatchEvent(new Event("input", { bubbles: true }));
        }

        window.onload = startCamera;
    </script>
    """

    # Tampilkan kamera dan tombol ambil
    st.components.v1.html(camera_code, height=500)

    # Input tersembunyi untuk base64 image dari kamera
    base64_img = st.text_area("Hidden Camera Input", value=st.session_state["camera_image"], label_visibility="collapsed")

    if base64_img and base64_img.startswith("data:image"):
        st.session_state["camera_image"] = base64_img  # Simpan base64 image

        try:
            header, encoded = base64_img.split(",", 1)
            decoded = base64.b64decode(encoded)
            image = Image.open(BytesIO(decoded))
            st.image(image, caption="📷 Gambar dari Kamera", use_container_width=True)

        except Exception as e:
            st.error(f"Gagal memproses gambar: {e}")

# Jalankan prediksi jika ada gambar
if image:
    with st.spinner("🔍 Memproses gambar..."):
        model = load_model()
        results = predict_image(model, image)
        img_with_boxes, class_counts = draw_results(image, results)

        st.image(img_with_boxes, caption="📊 Hasil Deteksi", use_container_width=True)
        st.subheader("Jumlah Objek Terdeteksi:")
        for name, count in class_counts.items():
            st.write(f"- **{name}**: {count}")
