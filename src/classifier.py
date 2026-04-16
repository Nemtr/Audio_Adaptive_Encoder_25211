import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib # Thư viện dùng để lưu mô hình AI

# Cấu hình đường dẫn
DATA_DIR = "data"
MODEL_PATH = "models/audio_classifier.pkl"

def extract_features(file_path):
    """Trích xuất 30 đặc trưng: Mean và Std của MFCC, ZCR, Centroid"""
    try:
        y, sr = librosa.load(file_path, sr=22050, duration=30.0)
        
        # 1. MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0) # <-- Lấy thêm độ biến thiên
        
        # 2. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr.T, axis=0)
        zcr_std = np.std(zcr.T, axis=0) # <-- Lấy thêm độ biến thiên
        
        # 3. Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent.T, axis=0)
        cent_std = np.std(cent.T, axis=0) # <-- Lấy thêm độ biến thiên
        
        # Gộp tất cả (13 + 13 + 1 + 1 + 1 + 1 = 30 chiều dữ liệu)
        return np.hstack((mfccs_mean, mfccs_std, zcr_mean, zcr_std, cent_mean, cent_std))
        
    except Exception as e:
        print(f"Lỗi đọc file {file_path}: {e}")
        return None

def build_dataset():
    """Quét thư mục data để tạo bộ dữ liệu huấn luyện"""
    X = [] # Chứa các ma trận số (Features)
    y = [] # Chứa nhãn (Labels: 'Speech' hoặc 'Music')
    
    classes = ['speech', 'music', 'noise']
    
    for cls in classes:
        folder_path = os.path.join(DATA_DIR, cls)
        if not os.path.exists(folder_path):
            print(f"⚠️ Cảnh báo: Không tìm thấy thư mục {folder_path}")
            continue
            
        print(f"📂 Đang trích xuất dữ liệu từ thư mục: {cls}...")
        for filename in os.listdir(folder_path):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                file_path = os.path.join(folder_path, filename)
                features = extract_features(file_path)
                if features is not None:
                    X.append(features)
                    # Gán nhãn viết hoa chữ cái đầu cho đẹp (Speech/Music)
                    y.append(cls.capitalize()) 
                    
    return np.array(X), np.array(y)

def train_model():
    print("🚀 BẮT ĐẦU HUẤN LUYỆN AI...")
    X, y = build_dataset()
    
    if len(X) == 0:
        print("❌ Không có dữ liệu để huấn luyện. Hãy kiểm tra lại thư mục data!")
        return

    # Chia dữ liệu: 80% để học, 20% để thi thử (test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Chọn thuật toán Random Forest (Rừng ngẫu nhiên - Rất mạnh cho dữ liệu dạng bảng)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    print("🧠 Đang cho AI học...")
    model.fit(X_train, y_train)
    
    # Làm bài kiểm tra
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Huấn luyện xong! Độ chính xác trên tập test: {acc * 100:.2f}%")
    
    # Lưu "bộ não" vào két sắt
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, MODEL_PATH)
    print(f"💾 Đã lưu mô hình AI tại: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()