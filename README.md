HỆ THỐNG TRỢ LÝ ĐA PHƯƠNG THỨC CHO NGƯỜI KHIẾM THỊ TRONG SINH HOẠT THƯỜNG NGÀY

Kiến trúc: Có thể chạy Offline - Tối ưu hóa cho CPU

--- HƯỚNG DẪN CÀI ĐẶT ---

1. CHUẨN BỊ MÔI TRƯỜNG
Yêu cầu hệ thống cài đặt Python 3.10 trở lên. Khuyến nghị sử dụng môi trường ảo (.venv) để tránh xung đột thư viện.

Mở terminal và chạy:
# Tạo và kích hoạt môi trường ảo (Trên Windows)
python -m venv .venv
.venv\Scripts\activate

# Tạo và kích hoạt môi trường ảo (Trên Linux/Mac)
python3 -m venv .venv
source .venv/bin/activate


2. CÀI ĐẶT THƯ VIỆN
Cài đặt toàn bộ các dependencies cần thiết (Bao gồm llama-cpp-python, faster-whisper, gradio, pillow, sounddevice...):

pip install -r requirements.txt


3. CHUẨN BỊ MÔ HÌNH (QUAN TRỌNG)
Vì hệ thống được tối ưu để chạy trên CPU, bạn CẦN chuẩn bị file trọng số (weights) theo đúng định dạng trước khi chạy:

[Bước 3.1] Tối ưu hóa mô hình Giọng nói (STT) sang CTranslate2 (int8):
Để giảm thời gian nhận diện từ vài giây xuống dưới 1s trên CPU, bạn cần tải và nén mô hình. 
(Lưu ý: Chúng ta sẽ tải model về máy trước khi convert để tránh lỗi "403 Forbidden" từ Hugging Face).

Mở terminal và chạy lần lượt 2 lệnh sau (mất khoảng 1-2 phút, yêu cầu có Internet):

# Lệnh 1: Tải model gốc về thư mục tạm "temp_model"
huggingface-cli download vinai/PhoWhisper-tiny --local-dir temp_model

# Lệnh 2: Nén model từ thư mục tạm sang định dạng CTranslate2 (int8)
ct2-transformers-converter --model temp_model --output_dir phowhisper-tiny-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization int8

-> Sau khi lệnh chạy xong, thư mục "phowhisper-tiny-ct2" sẽ xuất hiện trong project. Bạn có thể xóa thư mục "temp_model" đi để giải phóng ổ cứng.

-> Sau khi xong, thư mục "phowhisper-tiny-ct2" sẽ xuất hiện trong project.

[Bước 3.2] Tải mô hình Hình ảnh (VLM) định dạng GGUF:
Tạo một thư mục tên là "models" trong thư mục gốc của project.
Tải 2 file sau từ kho lưu trữ về và đặt vào thư mục "models":
1. model.gguf (VD: Vintern-1B-v3_5-Finetuned-Q6_K.gguf)
2. mmproj.gguf (VD: mmproj-Vintern-1B-v3_5-Finetuned.gguf)


--- HƯỚNG DẪN CHẠY HỆ THỐNG ---

Bạn có 2 cách để khởi chạy và kiểm tra hệ thống:

CÁCH 1: KIỂM TRA LUỒNG PIPELINE QUA TERMINAL (CLI)
Dùng để test nội bộ luồng STT -> VLM -> TTS xem có hoạt động trơn tru không. Nó sẽ tự lấy file âm thanh và hình ảnh trong thư mục "test_data/".
> python vintern_pipeline.py

CÁCH 2: KHỞI CHẠY GIAO DIỆN WEB (GRADIO UI) - [KHUYÊN DÙNG]
Mở giao diện Web App hoàn chỉnh có tính năng Warm-up (khởi động nóng) và hỗ trợ thu âm/chụp ảnh trực tiếp từ trình duyệt.
> python app_vintern.py
-> Truy cập vào đường link http://127.0.0.1:7860 hiện ra trên terminal để sử dụng.


--- CHÚ Ý (DISCLAIMER) ---

* Ưu tiên CPU: Dự án này mặc định dùng định dạng GGUF và int8, ép hệ thống sử dụng toàn bộ số lõi CPU để tính toán. Lỗi giải mã hình ảnh (stb_image) ngầm định của C++ đã được khắc phục hoàn toàn nhờ lớp đệm PIL.
* Chạy hoàn toàn Offline: Bạn chỉ cần Internet ở Bước 3 để tải mô hình. Ở các lần sử dụng sau, rút dây mạng hệ thống vẫn hoạt động 100%.
* Tự động nhận diện GPU (STT): Module Faster-Whisper trong code (fastwhis.py) ĐÃ được thiết lập tự động tìm GPU. Nếu máy có Card NVIDIA, nó tự chạy CUDA (float16); nếu không, nó tự fallback về CPU (int8). Bạn không cần sửa code.
* Nâng cấp GPU cho VLM: Nếu bạn muốn mô hình nhìn ảnh chạy bằng Card rời, hãy mở file "vintern_pipeline.py", tìm biến `n_gpu_layers=0` trong lúc khởi tạo Llama và đổi thành `n_gpu_layers=-1` (offload toàn bộ layer lên VRAM).