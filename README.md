# Hệ Thống Trợ Lý Đa Phương Thức Cho Người Khiếm Thị (Multimodal Assistant)

> Kiến trúc: **Chạy Offline** • **Tối ưu CPU** • **Hỗ trợ Finetuning**

Hệ thống trợ lý ảo đa phương thức hỗ trợ người khiếm thị nhận diện môi trường xung quanh thông qua giọng nói (STT), phân tích hình ảnh (VLM) và phản hồi bằng giọng nói (TTS). Dự án hiện đã được tái cấu trúc, hỗ trợ quá trình finetune mô hình cũng như chạy ứng dụng suy luận (inference) tối ưu cho CPU.

---

## 📂 Cấu trúc thư mục (Project Structure)

```text
CS431_multimodal_for_visually_impaired/
├── application/          # Chứa mã nguồn chạy ứng dụng suy luận (Web App & CLI Pipeline)
├── finetuning/           # [MỚI] Các script dùng để huấn luyện/finetune mô hình (VLM/STT)
├── models/               # Thư mục chứa các model weights (GGUF, CTranslate2)
├── test_data/            # Dữ liệu âm thanh/hình ảnh để test hệ thống
└── README.md             # Tài liệu dự án này
```

---

## 🚀 Hướng dẫn Finetuning (Tính năng mới)

Dự án hiện đã được bổ sung các script để finetune mô hình nhằm tăng cường độ chính xác cho các tác vụ cụ thể.

### 1. Chuẩn bị dữ liệu
*(Thêm hướng dẫn định dạng dữ liệu đầu vào cho finetuning tại đây: ví dụ chuẩn bị file JSONL, cách format prompt cho VLM, v.v.)*

### 2. Chạy Huấn luyện
*(Cập nhật các câu lệnh thực tế dùng để chạy finetuning của bạn)*
```bash
# Ví dụ:
# cd finetuning
# python train.py --model_path ../models/Qwen3-VL --data_path data/train.jsonl
```

---

## 💻 Hướng dẫn Cài đặt & Sử dụng Ứng dụng

Dưới đây là hướng dẫn cài đặt và chạy hệ thống pipeline suy luận (Inference) từ thư mục `application/`.

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

**Cài đặt Thư viện:**
```bash
pip install -r application/requirements.txt
```

### 2. Chuẩn bị Mô hình (QUAN TRỌNG)

Hệ thống suy luận được tối ưu cho CPU nên cần chuẩn bị model đúng định dạng trước khi chạy.

**A. Tối ưu hóa mô hình Giọng nói (STT - PhoWhisper)**
```bash
huggingface-cli download vinai/PhoWhisper-tiny --local-dir temp_model
ct2-transformers-converter --model temp_model --output_dir models/phowhisper-tiny-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization int8
```

**B. Mô hình Hình ảnh (VLM - Qwen3-VL)**
Đặt 2 file `.gguf` vào thư mục `models/`:
* `mmproj-Qwen3-VL-2B-Finetuned-BF16.gguf`
* `Qwen3-VL-2B-Finetuned-BF16-Q4_K_M.gguf`

### 3. Chạy Hệ thống

**Chạy Web App (Khuyên dùng):**
```bash
cd application
python app_qwen_gguf.py
```
Truy cập giao diện Web tại: `http://127.0.0.1:7860`

**Chạy Pipeline (CLI):** Hệ thống sẽ tự lấy dữ liệu từ `test_data/`, nhận diện giọng nói, phân tích ảnh và trả lời.
```bash
cd application
python qwen_pipeline_gguf.py
```

---

## ⚠️ Chú ý (Disclaimer)
* **Ưu tiên CPU:** Hệ thống suy luận sử dụng GGUF và int8, tận dụng toàn bộ lõi CPU. Lỗi decode ảnh (stb_image) đã được xử lý bằng PIL.
* **Offline hoàn toàn:** Chỉ cần Internet khi tải model lần đầu, sau đó hệ thống chạy 100% offline.
* **Tự động nhận diện GPU (STT):** Có NVIDIA GPU → dùng CUDA (float16) | Không có → fallback CPU (int8).
* **Chuyển đổi GPU cho VLM (Tuỳ chọn):** Mở file `application/qwen_pipeline_gguf.py`, tìm `n_gpu_layers=0` và đổi thành `n_gpu_layers=-1`.