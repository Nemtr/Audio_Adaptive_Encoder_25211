import subprocess
import os

def compress_audio(input_file, output_file, ai_label):
    """
    Nhận nhãn từ AI và gọi FFmpeg để nén file.
    Trả về: (Thành công hay không, Dung lượng gốc, Dung lượng nén)
    """
    command = ['ffmpeg', '-y', '-i', input_file]
    
    # Logic Thích ứng nội dung (Content-Adaptive)
    if ai_label == "Speech":
        command.extend(['-c:a', 'libopus', '-b:a', '32k'])
    elif ai_label == "Music":
        command.extend(['-c:a', 'aac', '-b:a', '128k'])
    elif ai_label == "Noise":
        command.extend(['-c:a', 'libopus', '-b:a', '64k'])
    else: 
        command.extend(['-c:a', 'libmp3lame', '-b:a', '128k'])
        
    command.append(output_file)
    
    try:
        # Chạy lệnh FFmpeg ngầm
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Tính toán dung lượng (MB)
        orig_size = os.path.getsize(input_file) / (1024 * 1024)
        comp_size = os.path.getsize(output_file) / (1024 * 1024)
        return True, orig_size, comp_size
        
    except Exception as e:
        return False, str(e), 0