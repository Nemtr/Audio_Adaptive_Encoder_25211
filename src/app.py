import streamlit as st
import os
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from classifier import extract_features
from encoder import compress_audio

# Thiết lập UI
st.set_page_config(page_title="Smart Audio Compressor", layout="wide")
st.title("🎧 Smart Audio Compressor")
st.markdown("Hệ thống nén âm thanh thích ứng nội dung (Content-Adaptive)")

# Hàm vẽ Waveform
def plot_waveform(audio_path, title):
    y, sr = librosa.load(audio_path, sr=22050)
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax, color='#1E88E5')
    ax.set_title(title)
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Biên độ")
    return fig

# Giao diện Upload File
uploaded_file = st.file_uploader("Kéo & Thả file âm thanh (.wav, .mp3)", type=['wav', 'mp3'])

if uploaded_file is not None:
    st.success(f"Tải lên thành công: {uploaded_file.name}")
    
    temp_dir = "data/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_input = os.path.join(temp_dir, uploaded_file.name)
    temp_output = os.path.join(temp_dir, f"compressed_{uploaded_file.name.split('.')[0]}.mkv") 
    
    with open(temp_input, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    # --- PHẦN 1: AUDIO PLAYER GỐC ---
    st.subheader("🔊 Nghe thử file gốc")
    st.audio(temp_input)

    with st.spinner("🧠 AI đang phân tích và xử lý..."):
        features = extract_features(temp_input)
        if features is not None:
            model = joblib.load("models/audio_classifier.pkl")
            prediction = model.predict([features])[0]
            
            st.info(f"🤖 **AI nhận diện: {prediction}**")
            
            success, orig_size, comp_size = compress_audio(temp_input, temp_output, prediction)
            
            if success:
                st.divider()
                st.subheader("📊 Kết quả báo cáo (Dashboard)")
                
                # --- PHẦN 2: CHỈ SỐ & BIỂU ĐỒ ---
                col_text, col_chart = st.columns([1, 1])
                
                with col_text:
                    st.metric("Dung lượng Gốc", f"{orig_size:.2f} MB")
                    st.metric("Sau khi nén", f"{comp_size:.2f} MB", f"-{((orig_size-comp_size)/orig_size)*100:.1f}%")
                    st.download_button("⬇️ Tải file nén", open(temp_output, "rb"), file_name=os.path.basename(temp_output))
                
                with col_chart:
                    # Vẽ biểu đồ cột so sánh dung lượng
                    chart_data = pd.DataFrame({
                        "Trạng thái": ["Gốc", "Nén"],
                        "Dung lượng (MB)": [orig_size, comp_size]
                    })
                    st.bar_chart(chart_data, x="Trạng thái", y="Dung lượng (MB)")

                # --- PHẦN 3: WAVEFORM & AUDIO NÉN ---
                st.divider()
                col_orig, col_comp = st.columns(2)
                
                with col_orig:
                    st.write("**Sóng âm file Gốc**")
                    st.pyplot(plot_waveform(temp_input, "Waveform Gốc"))
                
                with col_comp:
                    st.write("**Sóng âm file Nén**")
                    # Lưu ý: file mkv có thể cần định dạng lại để web player đọc được
                    st.audio(temp_output) 
                    st.pyplot(plot_waveform(temp_output, "Waveform Sau Nén"))
            else:
                # THÊM ĐOẠN NÀY VÀO ĐỂ BẮT LỖI
                st.error("❌ Hệ thống nén FFmpeg đã gặp sự cố!")
                st.code(orig_size) # Khi lỗi, hàm của chúng ta trả về nội dung lỗi ở biến thứ 2