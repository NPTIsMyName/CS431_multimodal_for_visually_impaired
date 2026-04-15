# Hệ Thống Trợ Lý Đa Phương Thức Cho Người Khiếm Thị

> Kiến trúc: **Chạy Offline** • **Tối ưu CPU**

---

## Hướng dẫn Cài đặt

### 1. Chuẩn bị Môi trường

Yêu cầu Python 3.10 trở lên. Khuyến nghị dùng môi trường ảo (.venv) để tránh xung đột thư viện.

```bash
# Trên Windows
python -m venv .venv
.venv\Scripts\activate

# Trên Linux/Mac
python3 -m venv .venv
source .venv/bin/activate
```

---

### 2. Cài đặt Thư viện

Cài toàn bộ dependencies cần thiết:

```bash
pip install -r requirements.txt
```

---

## Chuẩn bị Mô hình (QUAN TRỌNG)

Vì hệ thống tối ưu cho CPU, bạn cần chuẩn bị đúng định dạng model trước khi chạy.

---

### 1. Tối ưu hóa mô hình Giọng nói (STT)

Để giảm thời gian nhận diện xuống dưới 1 giây trên CPU:

```bash
# Tải model gốc
huggingface-cli download vinai/PhoWhisper-tiny --local-dir temp_model

# Convert sang CTranslate2 (int8)
ct2-transformers-converter \
  --model temp_model \
  --output_dir phowhisper-tiny-ct2 \
  --copy_files tokenizer.json preprocessor_config.json \
  --quantization int8
```

Sau khi hoàn tất:

* Thư mục `phowhisper-tiny-ct2/` sẽ xuất hiện
* Có thể xóa `temp_model/` để tiết kiệm dung lượng

---

### 2. Mô hình Hình ảnh (VLM)

Tạo thư mục:

```bash
models/
```

Đặt vào đó 2 file:

```text
models/
├── model.gguf
└── mmproj.gguf
```

Ví dụ:

* `Vintern-1B-v3_5-Finetuned-Q6_K.gguf`
* `mmproj-Vintern-1B-v3_5-Finetuned.gguf`

---

## Hướng dẫn Chạy Hệ thống

### 1. Chạy Pipeline (CLI)

Dùng để test luồng STT → VLM → TTS:

```bash
python vintern_pipeline.py
```

Hệ thống sẽ:

* Tự lấy dữ liệu từ thư mục `test_data/`
* Nhận diện giọng nói
* Phân tích hình ảnh
* Trả lời bằng giọng nói

---

### 2. Chạy Web App (Khuyên dùng)

```bash
python app_vintern.py
```

Truy cập:

```
http://127.0.0.1:7860
```

---

## Chú ý (Disclaimer)

* Ưu tiên CPU:

  * Sử dụng GGUF và int8
  * Tận dụng toàn bộ lõi CPU
  * Lỗi decode ảnh (stb_image) đã được xử lý bằng PIL

* Offline hoàn toàn:

  * Chỉ cần Internet khi tải model
  * Sau đó hệ thống chạy 100% offline

* Tự động nhận diện GPU (STT):

  * Có NVIDIA GPU → dùng CUDA (float16)
  * Không có → fallback CPU (int8)

* Nâng cấp GPU cho VLM (tuỳ chọn):

  * Mở file `vintern_pipeline.py`
  * Tìm:

    ```
    n_gpu_layers=0
    ```
  * Đổi thành:

    ```
    n_gpu_layers=-1
    ```
