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


class VisionLanguageModel:
    def __init__(
        self,
        model_path: str = "models/Vintern-1B-v3_5-Finetuned-Q5_K_M.gguf",
        mmproj_path: str = "models/mmproj-Vintern-1B-v3_5-Finetuned.gguf",
    ):
        """
        VLM chạy bằng llama.cpp (CPU)
        """
        self.model_path = model_path
        self.mmproj_path = mmproj_path

        print("\n[*] Initializing Vintern 1B GGUF...")
        start_time = time.time()

        if not os.path.exists(self.model_path) or not os.path.exists(self.mmproj_path):
            raise FileNotFoundError("Thiếu model.gguf hoặc mmproj.gguf")

        try:
            self.chat_handler = Llava15ChatHandler(
                clip_model_path=self.mmproj_path,
                verbose=False,
            )

            self.llm = Llama(
                model_path=self.model_path,
                chat_handler=self.chat_handler,
                n_ctx=2048,
                n_gpu_layers=0,
                n_threads=os.cpu_count(),
                verbose=False,
            )

        except Exception as e:
            print(f"[!] Load model lỗi: {e}")
            raise e

        print(f"[+] VLM ready ({time.time() - start_time:.2f}s)\n")

    def _image_to_base64(self, image_path: str) -> str:
        try:
            image = Image.open(image_path).convert("RGB")

            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")

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

            messages = [
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
                temperature=0.0,
                repeat_penalty=1.2,
            )

            latency = time.time() - start_infer
            print(f"[VLM Latency] {latency:.4f}s")

            # Safe extract
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
    print("MULTIMODAL ASSISTANT")
    print("=" * 60)

    stt_module = FastSpeechToText(model_size="tiny")

    vlm_module = VisionLanguageModel(
        model_path="models/Vintern-1B-v3_5-Finetuned-Q5_K_M.gguf",
        mmproj_path="models/mmproj-Vintern-1B-v3_5-Finetuned.gguf",
    )

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