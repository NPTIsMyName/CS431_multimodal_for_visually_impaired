import time
import os
import torch
from faster_whisper import WhisperModel


class FastSpeechToText:
    def __init__(self, model_size="base"):
        """
        Khởi tạo Faster-Whisper với khả năng TỰ ĐỘNG PHÁT HIỆN GPU/CPU.
        """
        print(f"[*] Đang khởi tạo Faster-Whisper (Kích thước: {model_size})")

        # TỰ ĐỘNG NHẬN DIỆN THIẾT BỊ CHO NHẬN DIỆN GIỌNG NÓI
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"  # Dùng float16 trên GPU cho tốc độ siêu nhanh
            print("[+] Faster-Whisper: Đã phát hiện GPU. Chạy bằng CUDA (float16).")
        else:
            self.device = "cpu"
            self.compute_type = "int8"  # Ép nén mô hình để tính toán siêu tốc trên CPU
            print("[-] Faster-Whisper: Không có GPU. Chạy trên CPU (int8).")

        start_time = time.time()

        try:
            # Truyền tham số tự động vào WhisperModel
            self.model = WhisperModel(
                "phowhisper-base-ct2",
                device=self.device,
                compute_type=self.compute_type
            )
        except Exception as e:
            print(f"[!] Lỗi khi tải mô hình STT: {e}")
            raise e

        load_time = time.time() - start_time
        print(f"[+] Khởi tạo STT thành công! Mất {load_time:.2f} giây.\n")

    def transcribe(self, audio_path) -> str:
        """
        Nhận diện âm thanh siêu tốc.
        """
        try:
            start_infer = time.time()
            print("[*] Đang xử lý file âm thanh...")

            segments, info = self.model.transcribe(
                audio_path,
                language="vi",
                beam_size=1,
                condition_on_previous_text=False
            )

            # Gộp text từ generator
            text_result = ""
            for segment in segments:
                text_result += segment.text + " "

            latency = time.time() - start_infer
            print(f"[STT Latency] Thời gian xử lý: {latency:.4f} giây")

            return text_result.strip()

        except Exception as e:
            print(f"[!] Lỗi: {str(e)}")
            return ""


# HÀM MAIN TEST ĐỘC LẬP
if __name__ == "__main__":
    stt_module = FastSpeechToText(model_size="small")

    sample_audio_path = "test_data/hay_mo_ta_canh_vat_truoc_mat.wav"

    if os.path.exists(sample_audio_path):
        print(f"[*] Bắt đầu suy luận file: {sample_audio_path}")
        transcribed_text = stt_module.transcribe(sample_audio_path)

        print("\n" + "=" * 50)
        print("VĂN BẢN KẾT QUẢ (FASTER-WHISPER):")
        print(transcribed_text if transcribed_text else "[Không có văn bản]")
        print("=" * 50)
    else:
        print(f"[!] Không tìm thấy file {sample_audio_path} để test.")