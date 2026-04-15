HỆ THỐNG TRỢ LÝ ĐA PHƯƠNG THỨC CHO NGƯỜI KHIẾM THỊ TRONG SINH HOẠT THƯỜNG NGÀY

Kiến trúc: Có thể chạy Offline - Tối ưu hóa cho CPU

--- HƯỚNG DẪN CÀI ĐẶT ---

1. CHUẨN BỊ MÔI TRƯỜNG
   Yêu cầu hệ thống cài đặt Python 3.10 trở lên. Khuyến nghị sử dụng môi trường ảo (.venv) để tránh xung đột thư viện.

Mở terminal và chạy:

# Trên Windows

python -m venv .venv
.venv\Scripts\activate

# Trên Linux/Mac

python3 -m venv .venv
source .venv/bin/activate

2. CÀI ĐẶT THƯ VIỆN
   Cài đặt toàn bộ các dependencies cần thiết:

pip install -r requirements.txt

3. CHUẨN BỊ MÔ HÌNH (QUAN TRỌNG)

Hệ thống được tối ưu để chạy trên CPU, do đó bạn cần chuẩn bị trước các file mô hình.

[Bước 3.1] Tối ưu hóa mô hình Giọng nói (STT) sang CTranslate2 (int8)

Chạy lần lượt:

# Tải model gốc

huggingface-cli download vinai/PhoWhisper-tiny --local-dir temp_model

# Convert sang CTranslate2 int8

ct2-transformers-converter --model temp_model --output_dir phowhisper-tiny-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization int8

Sau khi hoàn tất:

* Thư mục "phowhisper-tiny-ct2" sẽ được tạo
* Có thể xóa "temp_model" để tiết kiệm dung lượng

[Bước 3.2] Chuẩn bị mô hình Hình ảnh (VLM)

* Tạo thư mục: models/
* Tải và đặt vào đó 2 file:

  * model.gguf (ví dụ: Vintern-1B-v3_5-Finetuned-Q6_K.gguf)
  * mmproj.gguf (ví dụ: mmproj-Vintern-1B-v3_5-Finetuned.gguf)

--- HƯỚNG DẪN CHẠY HỆ THỐNG ---

CÁCH 1: CHẠY PIPELINE QUA TERMINAL (CLI)

Dùng để test luồng xử lý STT → VLM → TTS:

python vintern_pipeline.py

CÁCH 2: CHẠY GIAO DIỆN WEB (KHUYÊN DÙNG)

python app_vintern.py

Sau đó truy cập:
http://127.0.0.1:7860

--- CHÚ Ý (DISCLAIMER) ---

* Ưu tiên CPU:
  Hệ thống sử dụng GGUF và int8 để tối ưu hiệu năng trên CPU.
  Lỗi decode ảnh từ C++ đã được xử lý thông qua PIL.

* Chạy Offline:
  Chỉ cần Internet khi tải model (Bước 3).
  Sau đó hệ thống có thể chạy hoàn toàn offline.

* Tự động nhận diện GPU (STT):
  Faster-Whisper tự động sử dụng GPU nếu có (CUDA), nếu không sẽ fallback CPU.

* Nâng cấp GPU cho VLM:
  Mở file vintern_pipeline.py
  Tìm tham số:
  n_gpu_layers=0
  Đổi thành:
  n_gpu_layers=-1
  để sử dụng GPU cho mô hình hình ảnh.

--- GHI CHÚ ---

* Không đưa file model vào GitHub (do giới hạn dung lượng)
* Nên lưu model trên Google Drive hoặc HuggingFace và cung cấp link trong README
