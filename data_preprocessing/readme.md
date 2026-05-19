## 📂 Cấu trúc Repository

Bao gồm 3 Jupyter Notebook chính được thực thi theo thứ tự sau:

### 1. `gen_data.ipynb`
Notebook này chịu trách nhiệm tương tác với model (Pixtral) để sinh câu trả lời AI cho các hình ảnh đầu vào.
* **Đầu vào:** Các file metadata chứa hình ảnh và câu hỏi (`metadata_ans_0.json` - tập không thể trả lời và `metadata_ans_1.json` - tập có thể trả lời).
* **Xử lý:** Chạy vòng lặp qua từng mẫu dữ liệu để prompt model sinh ra câu trả lời tương ứng cho hình ảnh.
* **Đầu ra:** Các file `pixtral_ans_0.json` và `pixtral_ans_1.json` chứa câu hỏi gốc và câu trả lời do AI tạo ra.

### 2. `translate_q.ipynb`
Notebook này dùng để dịch và đa dạng hóa (paraphrase) các câu hỏi từ tiếng Anh sang tiếng Việt.
* **Đầu vào:** Các file `pixtral_ans_0.json` và `pixtral_ans_1.json`.
* **Xử lý:** Sử dụng thư viện `deep_translator` kết hợp với một từ điển ánh xạ (`paraphrase_map`) để dịch các câu hỏi phổ biến (VD: "what is this?" -> "Cái gì đây?", "Đây là vật gì?", v.v.) sang nhiều biến thể tiếng Việt khác nhau.
* **Đầu ra:** Các file `pixtral_ans_0_translated.json` và `pixtral_ans_1_translated.json`.

### 3. `EDA.ipynb`
Notebook cuối cùng thực hiện Phân tích Khám phá Dữ liệu (Exploratory Data Analysis) trên các file đã được dịch tiếng Việt.
* **Đầu vào:** `pixtral_ans_0_translated.json` và `pixtral_ans_1_translated.json`.
* **Xử lý:** * Chuyển đổi JSON thành cấu trúc Pandas DataFrame.
  * Tính toán thống kê độ dài số từ của câu hỏi và câu trả lời (ngắn nhất, dài nhất, trung bình).
  * Trực quan hóa dữ liệu bằng `matplotlib` và `seaborn` (phân phối độ dài văn bản, thống kê tần suất các loại câu trả lời).
* **Mục tiêu:** Cung cấp cái nhìn tổng quan về cấu trúc ngôn ngữ và độ nhiễu của dữ liệu để phục vụ cho các bước tiền xử lý và fine-tuning model tiếp theo.

## 🛠 Yêu cầu hệ thống (Dependencies)

Để chạy trơn tru các file notebook này, bạn cần cài đặt Python 3.x và các thư viện cần thiết. Nếu bạn đang sử dụng môi trường Linux, bạn có thể dễ dàng mở Terminal và cài đặt thông qua `pip`:

```bash
pip install pandas numpy matplotlib seaborn deep-translator tqdm
