import time
import os
import base64
import traceback
from PIL import Image
import io

from fastwhis import FastSpeechToText
from tts_vieneu import FastTextToSpeech

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

system_rules = """Bạn là một trợ lý AI xuất sắc trong việc phân tích hình ảnh (VQA).
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên hình ảnh được cung cấp một cách ngắn gọn, chính xác bằng Tiếng Việt một cách tự nhiên.

QUY TẮC QUAN TRỌNG:
1. Bạn phải CỐ GẮNG HẾT SỨC để trả lời dựa trên bất kỳ chi tiết nào bạn có thể nhận diện được trong ảnh,một cách ngắn gọn , xúc tích.
2. CHỈ KHI hình ảnh THỰC SỰ QUÁ MỜ, nhòe, nhiễu nặng hoặc tối đen đến mức MỘT CON NGƯỜI cũng không thể đoán được bất cứ thông tin gì liên quan đến câu hỏi,thì bạn phải trả lời đúng một câu: "Ảnh mờ, bạn vui lòng chụp lại ảnh rõ hơn."
3. Nếu ảnh chỉ hơi mờ hoặc thiếu sáng nhưng vẫn có thể suy luận được, tuyệt đối không phàn toàn, hãy đưa ra câu trả lời tốt nhất của bạn.
4. Trả lời ngắn gọn, súc tích trong một câu, không giải thích dài dòng, không thêm thông tin không cần thiết, chỉ trả lời đúng trọng tâm câu hỏi.
5. Luôn trả lời bằng tiếng Việt, không được trả lời bằng ngôn ngữ khác, không được trả lời bằng tiếng Anh, không được trả lời bằng tiếng Trung, chỉ được trả lời bằng tiếng Việt.
"""

class VisionLanguageModel:
    def __init__(
        self,
        model_path: str = r"C:\Users\npt31\OneDrive\Desktop\CS431_multimodal_for_visually_impaired\models\Qwen3-VL-2B-Finetuned-BF16-Q4_K_M.gguf",
        mmproj_path: str = r"C:\Users\npt31\OneDrive\Desktop\CS431_multimodal_for_visually_impaired\models\mmproj-Qwen3-VL-2B-Finetuned-BF16.gguf",
    ):
        """
        VLM chạy bằng llama.cpp (CPU)
        """
        self.model_path = model_path
        self.mmproj_path = mmproj_path

        print("\n[*] Initializing Qwen3-VL 2B GGUF...")
        start_time = time.time()

        if not os.path.exists(self.model_path) or not os.path.exists(self.mmproj_path):
            raise FileNotFoundError(f"Thiếu model hoặc mmproj tại thư mục: \nModel: {self.model_path}\nProj: {self.mmproj_path}")

        try:
            # Vẫn sử dụng Llava15ChatHandler làm cầu nối đánh giá ảnh cho kiến trúc LLaVA/Qwen-VL GGUF
            self.chat_handler = Llava15ChatHandler(
                clip_model_path=self.mmproj_path,
                verbose=False,
            )

            self.llm = Llama(
                model_path=self.model_path,
                chat_handler=self.chat_handler,
                n_ctx=4096,  # Tăng lên 4096 vì Qwen-VL cần nhiều token cho hình ảnh hơn
                n_gpu_layers=0,
                n_threads=os.cpu_count(),
                verbose=False,
            )

        except Exception as e:
            print(f"[!] Load model lỗi: {e}")
            raise e

        print(f"[+] VLM ready ({time.time() - start_time:.2f}s)\n")

    def _image_to_base64(self, image_path: str, max_size: int = 896) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            
            # Ép cạnh dài nhất về max_size pixel, giữ nguyên tỷ lệ
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            buffered = io.BytesIO()
            # Tăng quality để ảnh rõ nét, hỗ trợ Qwen đọc text/chi tiết nhỏ
            image.save(buffered, format="JPEG", quality=95)

            encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"

        except Exception as e:
            print(f"[!] Lỗi xử lý ảnh: {e}")
            raise e

    def analyze(self, image_path: str, user_question: str) -> str:
        try:
            start_infer = time.time()
            print("[*] VLM analyzing...")

            base64_image = self._image_to_base64(image_path)

            # Đưa System Prompt vào cấu trúc messages chuẩn
            messages = [
                {
                    "role": "system",
                    "content": system_rules
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": base64_image}},
                        {"type": "text", "text": user_question},
                    ],
                }
            ]

            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=128,
                temperature=0.2,
                repeat_penalty=1.2,
            )

            latency = time.time() - start_infer
            print(f"[VLM Latency] {latency:.4f}s")

            if "choices" not in response or len(response["choices"]) == 0:
                return "Tôi không chắc mình hiểu đúng hình ảnh."

            content = response["choices"][0]["message"].get("content", "")
            return content.strip()

        except Exception as e:
            print(f"[!] VLM error: {e}")
            traceback.print_exc()
            return "Xin lỗi, tôi không thể nhìn rõ cảnh vật lúc này."

# TEST PIPELINE
if __name__ == "__main__":
    print("=" * 60)
    print("VIBEBLIND ASSISTANT PIPELINE")
    print("=" * 60)

    stt_module = FastSpeechToText(model_size="tiny")

    vlm_module = VisionLanguageModel() # Sẽ tự động lấy đường dẫn mặc định từ class

    tts_module = FastTextToSpeech(voice_index=0)

    audio_input_path = "test_data/hay_mo_ta_canh_vat_truoc_mat.wav"
    image_input_path = "test_data/truck_and_human_ahead.jpg"

    print("\n" + "=" * 60)
    print("PIPELINE START")
    print("=" * 60)

    total_start = time.time()

    print("\n>>> STEP 1: STT")
    question_text = stt_module.transcribe(audio_input_path)
    print(f"-> Question: '{question_text}'")

    if question_text:
        print("\n>>> STEP 2: VLM")
        answer_text = vlm_module.analyze(
            image_path=image_input_path,
            user_question=question_text,
        )
        print(f"-> Answer: '{answer_text}'")

        print("\n>>> STEP 3: TTS")
        tts_module.speak(answer_text)
    else:
        print("[!] Không nhận diện được câu hỏi")

    print("\n" + "=" * 60)
    print(f"[TOTAL LATENCY] {time.time() - total_start:.2f}s")
    print("=" * 60)