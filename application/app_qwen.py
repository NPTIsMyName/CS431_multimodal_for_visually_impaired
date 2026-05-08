import time
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig, AutoConfig
from PIL import Image

class VisionLanguageModel:
    def __init__(self, model_id: str = "NPTIsMyName/qwen3-vl-2b-it-ft-bf16", quantization: str = None):
        """
        Khởi tạo VLM Qwen3-VL 2B hỗ trợ chọn linh hoạt Quantization và tự động vá lỗi Config.
        :param model_id: ID mô hình trên HuggingFace
        :param quantization: '4bit', '8bit' hoặc None (không nén)
        """
        self.model_id = model_id
        self.quantization = quantization
        print(f"\n[*] Đang kiểm tra phần cứng hệ thống để setup Qwen (Quantization: {quantization})...")
        
        quantization_config = None

        # CẤU HÌNH THIẾT BỊ VÀ KIỂU DỮ LIỆU
        if torch.cuda.is_available():
            self.device = "cuda"
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
                print(f"[+] GPU hỗ trợ BFloat16.")
            else:
                self.dtype = torch.float16
                print(f"[+] Đã phát hiện GPU hỗ trợ Float16.")
                
            # CẤU HÌNH LƯỢNG TỬ HÓA LINH HOẠT
            if self.quantization == "4bit":
                print("[+] Đang kích hoạt chế độ nén 4-bit (NF4)...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",           
                    bnb_4bit_compute_dtype=self.dtype,   
                    bnb_4bit_use_double_quant=True       
                )
            elif self.quantization == "8bit":
                print("[+] Đang kích hoạt chế độ nén 8-bit...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            else:
                print("[+] Không sử dụng nén (Tải mô hình gốc).")
        else:
            self.device = "cpu"
            self.dtype = torch.float32 
            print(f"[-] Không tìm thấy GPU. Chạy bằng CPU (Float32) - Không hỗ trợ 4-bit/8-bit.")

        print(f"[*] Đang khởi tạo VLM: {self.model_id} trên {self.device.upper()}")
        start_time = time.time()
        
        try:
            # --- BẮT ĐẦU VÁ LỖI CONFIG ---
            model_config = AutoConfig.from_pretrained(self.model_id)
            if getattr(model_config, "rope_scaling", None) is None:
                print("[!] Cảnh báo: Tham số rope_scaling bị None, đang tự động vá thành dict rỗng...")
                model_config.rope_scaling = {}
            # --- KẾT THÚC VÁ LỖI ---

            # Tải mô hình với config đã được vá
            if quantization_config:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    config=model_config,
                    quantization_config=quantization_config,
                    device_map=self.device
                )
            else:
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_id,
                    config=model_config,
                    torch_dtype=self.dtype,
                    device_map=self.device
                )
                
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
        except Exception as e:
            print(f"[!] Lỗi khi tải mô hình VLM: {e}")
            raise e
            
        print(f"[+] Khởi tạo VLM thành công! Mất {time.time() - start_time:.2f} giây.\n")
    
    def analyze(self, image_path: str, user_question: str) -> str:
        """
        Nhận ảnh và câu hỏi -> Trả về câu trả lời.
        """
        try:
            start_infer = time.time()
            print(f"[*] VLM đang phân tích ảnh và suy luận...")

            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            
            # giảm size ảnh
            max_size = (512, 512)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)

            system_rules = """Bạn là một trợ lý AI xuất sắc trong việc phân tích hình ảnh (VQA).
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên hình ảnh được cung cấp một cách ngắn gọn, chính xác bằng Tiếng Việt một cách tự nhiên.

QUY TẮC QUAN TRỌNG:
1. Bạn phải CỐ GẮNG HẾT SỨC để trả lời dựa trên bất kỳ chi tiết nào bạn có thể nhận diện được trong ảnh,một cách ngắn gọn , xúc tích.
2. CHỈ KHI hình ảnh THỰC SỰ QUÁ MỜ, nhòe, nhiễu nặng hoặc tối đen đến mức MỘT CON NGƯỜI cũng không thể đoán được bất cứ thông tin gì liên quan đến câu hỏi,thì bạn phải trả lời đúng một câu: "Ảnh mờ, bạn vui lòng chụp lại ảnh rõ hơn."
3. Nếu ảnh chỉ hơi mờ hoặc thiếu sáng nhưng vẫn có thể suy luận được, tuyệt đối không phàn nàn, hãy đưa ra câu trả lời tốt nhất của bạn.
4. Trả lời ngắn gọn, súc tích trong một câu, không giải thích dài dòng, không thêm thông tin không cần thiết, chỉ trả lời đúng trọng tâm câu hỏi.
5. Luôn trả lời bằng tiếng Việt, không được trả lời bằng ngôn ngữ khác, không được trả lời bằng tiếng Anh, không được trả lời bằng tiếng Trung, chỉ được trả lời bằng tiếng Việt.
"""
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_rules}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": user_question},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    use_cache=True,
                    temperature=None,
                    top_p=None,
                    top_k=None
                )
                
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            latency = time.time() - start_infer
            print(f"[VLM Latency] Thời gian VLM suy nghĩ: {latency:.4f} giây")
            
            return output_text.strip()
            
        except Exception as e:
            print(f"[!] Lỗi trong quá trình VLM suy luận: {str(e)}")
            import traceback
            traceback.print_exc()
            return "Xin lỗi, tôi không thể nhìn rõ cảnh vật lúc này."