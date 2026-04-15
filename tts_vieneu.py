import time
import sounddevice as sd
import numpy as np
from vieneu import Vieneu


class FastTextToSpeech:
    def __init__(self, voice_index: int = 0):
        """
        Khởi tạo VieNeu TTS (CPU GGUF)
        """
        print("[*] Đang khởi tạo engine VieNeu-TTS...")
        start_time = time.time()

        try:
            self.tts = Vieneu()

            # Danh sách giọng
            self.available_voices = self.tts.list_preset_voices()

            print("\n[*] Danh sách giọng đọc:")
            for idx, (desc, name) in enumerate(self.available_voices):
                print(f"   [{idx}] - {desc} (ID: {name})")

            if self.available_voices:
                safe_index = voice_index if voice_index < len(self.available_voices) else 0
                _, self.current_voice_id = self.available_voices[safe_index]
                self.voice_data = self.tts.get_preset_voice(self.current_voice_id)

                print(f"\n[+] Giọng mặc định: {self.available_voices[safe_index][0]}")
            else:
                self.voice_data = None
                print("\n[+] Dùng giọng mặc định hệ thống")

        except Exception as e:
            print(f"[!] Lỗi khi tải VieNeu: {e}")
            raise e

        load_time = time.time() - start_time
        print(f"[+] TTS ready! ({load_time:.2f}s)\n")

    def speak(self, text: str):
        """
        Sinh và phát audio trực tiếp
        """
        try:
            start_infer = time.time()
            print(f"[*] Text: '{text}'")

            # Infer
            if self.voice_data:
                audio_result = self.tts.infer(text=text, voice=self.voice_data)
            else:
                audio_result = self.tts.infer(text=text)

            latency = time.time() - start_infer
            print(f"[TTS Latency] {latency:.4f}s")

            # Parse output
            audio_array = None
            sample_rate = 24000

            if isinstance(audio_result, tuple):
                if isinstance(audio_result[0], int):
                    sample_rate, audio_array = audio_result
                else:
                    audio_array, sample_rate = audio_result

            elif isinstance(audio_result, dict):
                audio_array = audio_result.get("audio") or audio_result.get("wav")
                sample_rate = audio_result.get("sr", sample_rate)

            else:
                audio_array = np.array(audio_result)

            if audio_array is None:
                raise ValueError("Không lấy được audio từ TTS output")

            # Mono
            if audio_array.ndim > 1:
                audio_array = audio_array.squeeze()

            # Ensure float32 (sounddevice yêu cầu)
            audio_array = audio_array.astype(np.float32)

            print("[*] Đang phát...")
            sd.play(audio_array, sample_rate)
            sd.wait()

        except Exception as e:
            print(f"[!] Lỗi phát audio: {e}")

    def close(self):
        """Giải phóng tài nguyên"""
        if hasattr(self, "tts"):
            self.tts.close()


# TEST
if __name__ == "__main__":
    tts_module = FastTextToSpeech(voice_index=0)

    test_text = "Cẩn thận, trước mặt bạn là bức tường cách khoảng hai mét."
    tts_module.speak(test_text)

    time.sleep(1)
    tts_module.speak("Vui lòng rẽ trái để tránh chướng ngại vật.")

    tts_module.close()