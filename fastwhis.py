import time
import os
import torch
from faster_whisper import WhisperModel

class FastSpeechToText:
    def __init__(self, model_size="small"):
        print(f"[*] Đang khởi tạo Faster-Whisper (Kích thước: {model_size})")
        
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16" 
            print(f"[+] Faster-Whisper: Đã phát hiện GPU. Chạy bằng CUDA (float16).")
        else:
            self.device = "cpu"
            self.compute_type = "int8" 
            print(f"[-] Faster-Whisper: Không có GPU. Chạy trên CPU (int8).")

        start_time = time.time()
        try:
            self.model = WhisperModel(
                "phowhisper-small-ct2", 
                device=self.device, 
                compute_type=self.compute_type
            )
        except Exception as e:
            print(f"[!] Lỗi khi tải mô hình STT: {e}")
            raise e
            
        print(f"[+] Khởi tạo STT thành công! Mất {time.time() - start_time:.2f} giây.\n")
        
    def transcribe(self, audio_path) -> str:
        try:
            start_infer = time.time()
            segments, info = self.model.transcribe(
                audio_path, 
                language="vi", 
                beam_size=1,
                condition_on_previous_text=False
            )
            
            text_result = "".join([segment.text + " " for segment in segments])
            print(f"[STT Latency] Thời gian xử lý: {time.time() - start_infer:.4f} giây")
            return text_result.strip()
            
        except Exception as e:
            print(f"[!] Lỗi: {str(e)}")
            return ""
