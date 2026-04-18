README - Audio Adaptive Encoder Project

1. Project Overview
This project, Content-Adaptive Audio Encoder (Project Code: 25211), is an intelligent system designed to optimize audio compression by automatically classifying input content (Speech, Music, or Noise) and routing it to the most efficient codec and bitrate configuration.

The system utilizes Machine Learning (Random Forest) for classification and FFmpeg for high-performance encoding, all wrapped in a user-friendly Streamlit web interface.

2. Key Features
AI-Driven Classification: Automatically identifies audio types using MFCC, Zero Crossing Rate, and Spectral Centroid features.

Dynamic Encoding:
    Speech: Opus @ 32kbps (Up to 97% reduction).
    Music: AAC @ 128kbps (High-fidelity stereo preservation).
    Noise: Opus @ 64kbps (Efficient environmental sound storage).

Interactive Dashboard: Real-time waveform visualization and compression ratio reporting.

3. System Requirements
Operating System: Ubuntu (recommended) or Linux-based systems.
Python Version: 3.10+
External Tools: FFmpeg (must be installed in the system PATH).

4. Installation & Setup Instructions
Step 1: Install System Dependencies
Open your terminal and install Python, Pip, and FFmpeg:
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg -y

Step 2: Clone the Repository
git clone <your-repo-link>
cd Audio_Adaptive_Encoder_25211

Step 3: Setup Virtual Environment
python3 -m venv .venv
source .venv/bin/activate

Step 4: Install Python Libraries
pip install --upgrade pip
pip install -r requirements.txt

Note: If you don't have requirements.txt, manually install: pip install librosa pydub scikit-learn streamlit matplotlib joblib

Step 5: Install Soundfile (Optional but Recommended)
To avoid errors with librosa loading certain audio formats:
sudo apt install libsndfile1

5. How to Run
Ensure your virtual environment is active (.venv), then run the Streamlit app:
streamlit run src/app.py
Once the command is executed, open your browser at http://localhost:8501.

6. Project Structure
├── data/           # Dataset for training (Music, Noise, Speech)
├── models/         # Trained Random Forest model (.pkl)
├── src/            # Source code
│   ├── app.py          # Main UI and Streamlit logic
│   ├── classifier.py   # Feature extraction and AI prediction
│   └── encoder.py      # FFmpeg routing and encoding logic
├── requirements.txt
└── README.md


7. Authors
Trần Lê Hải Nam (20213580) - Lead Developer for System Design & Backend.
Nguyễn Hữu Mạnh (20224288) - Lead Developer for Machine Learning & Evaluation.
Supervisor: TS. Phạm Văn Tiến

