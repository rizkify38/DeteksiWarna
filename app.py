import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase # Import ini

# --- 1. Konfigurasi Aplikasi Streamlit ---
st.set_page_config(
    page_title="Real-time Color Detector",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Real-time Color Detector with YOLOv8")
st.write("Deteksi warna secara real-time menggunakan model YOLOv8 yang telah dilatih.")

# Inisialisasi session_state (sekarang tidak digunakan untuk kontrol start/stop utama)
# Namun tetap berguna jika ada fitur lain yang memerlukan state ini
if 'run_camera_status' not in st.session_state: # Ganti nama key agar lebih jelas
    st.session_state.run_camera_status = False

# --- 2. Muat Model YOLOv8 ---
@st.cache_resource # Cache resource untuk menghindari model dimuat berulang kali
def load_yolo_model_with_checks(): # Ganti nama fungsi untuk kejelasan
    # PASTIKAN JALUR INI BENAR UNTUK DEPLOYMENT DI STREAMLIT CLOUD
    # Jika best.pt ada di root folder repo GitHub Anda: 'best.pt'
    # Jika best.pt ada di runs/detect/train3/weights/best.pt di repo GitHub Anda: 'runs/detect/train3/weights/best.pt'
    model_path_in_repo = 'best.pt' # <--- SESUAIKAN JALUR INI!
    
    st.info(f"Mencoba memuat model dari jalur: {model_path_in_repo}")
    
    if not os.path.exists(model_path_in_repo):
        st.error(f"FATAL ERROR: Model tidak ditemukan di jalur: {model_path_in_repo}. Harap pastikan model 'best.pt' berada di lokasi yang benar di repositori GitHub Anda.")
        st.stop() # Hentikan aplikasi jika model tidak ditemukan

    try:
        model = YOLO(model_path_in_repo)
        st.success(f"Model YOLOv8 berhasil dimuat.") # Hapus detail path dari pesan sukses agar tidak ambigu
        return model
    except Exception as e:
        st.error(f"FATAL ERROR: Gagal memuat model YOLOv8 dari {model_path_in_repo}. Error: {e}")
        st.stop() # Hentikan aplikasi jika model gagal dimuat

model = load_yolo_model_with_checks()

# --- 3. Konfigurasi Deteksi (Optional: Tambahkan slider untuk user) ---
# Set nilai default slider lebih rendah untuk pengujian awal
confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05) # Default 0.25
iou_threshold = st.slider("IoU Threshold (NMS)", 0.0, 1.0, 0.5, 0.05) # Default 0.5

# --- 4. Stream Kamera Real-time dengan streamlit-webrtc ---
st.subheader("Live Camera Feed")

# Definisi Video Transformer Class
class VideoProcessor(VideoTransformerBase):
    def __init__(self, model_instance, conf_thresh, iou_thresh):
        self.model = model_instance
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.frame_count = 0 # Tambahkan penghitung frame untuk debugging

    def transform(self, frame):
        self.frame_count += 1
        # Debugging: Konfirmasi frame diterima
        # st.sidebar.text(f"Frame {self.frame_count} received.") 

        img = frame.to_ndarray(format="bgr24")

        # Lakukan Inferensi
        results = self.model.predict(img, conf=self.conf_thresh, iou=self.iou_thresh, verbose=False)

        # Gambar Bounding Boxes dan Label
        annotated_frame = results[0].plot() # YOLOv8 .plot() menghasilkan BGR numpy array

        # Debugging: Cek apakah ada deteksi
        num_detections = len(results[0].boxes)
        # st.sidebar.text(f"Frame {self.frame_count}: Detections: {num_detections}")

        return annotated_frame # Kembalikan numpy array BGR

# Panggil webrtc_streamer
webrtc_streamer(
    key="color_detector_camera",
    video_processor_factory=lambda: VideoProcessor(model, confidence_threshold, iou_threshold),
    rtc_configuration={
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:stun2.l.google.com:19302"]},
            {"urls": ["stun:stun3.l.google.com:19302"]},
            {"urls": ["stun:stun4.l.google.com:19302"]},
            {"urls": ["stun:stun.services.mozilla.com"]},
            {"urls": ["stun:stun.nextcloud.com:3478"]},
            {"urls": ["stun:stun.stunprotocol.org:3478"]},
            {"urls": ["stun:stun.voipbuster.com:3478"]},
        ]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True # Perbaikan deprecation warning
)

st.info("Aplikasi siap. Izinkan akses kamera di browser Anda. Jika deteksi tidak muncul, coba turunkan 'Confidence Threshold' atau 'IoU Threshold' di slider.")

# Hapus bagian tombol Mulai/Hentikan yang sudah tidak relevan
# Komentari atau hapus semua kode ini:
# col1, col2 = st.columns(2)
# with col1:
#     start_button = st.button("Mulai Deteksi Kamera")
# with col2:
#     stop_button = st.button("Hentikan Deteksi Kamera")
# if start_button:
#     st.session_state.run_camera = True
# elif stop_button:
#     st.session_state.run_camera = False
# frame_placeholder = st.empty()
# if st.session_state.run_camera:
#     # ... (seluruh loop while cap.isOpened() dan logicnya) ...
# else:
#     st.info("Tekan 'Mulai Deteksi Kamera' untuk memulai deteksi real-time dari webcam Anda.")